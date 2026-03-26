#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂反解控制系统
基于当前位置进行六方向移动控制
整合读取角度、正运动学、逆运动学和角度设置功能
"""

import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import threading
import math
from typing import Tuple, List, Optional

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入运动学模块
from forword_kinematics_01 import GenkiArmForwardKinematics
from inverse_kinematics_02 import GenkiArmInverseKinematics


class ServoController:
    """舵机控制器类 - 整合读取和写入功能"""
    
    def __init__(self, port="COM4", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        
        # 寄存器地址常量
        self.ADDR_PRESENT_POSITION = 56
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_TORQUE_ENABLE = 40
        
        # 指令常量
        self.INST_READ = 2
        self.INST_WRITE = 3
        
        # 通信状态常量
        self.COMM_SUCCESS = 0
        self.COMM_RX_TIMEOUT = -6
        self.COMM_RX_CORRUPT = -7
        
        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            print(f"成功打开串口 {port}")
        except serial.SerialException as e:
            print(f"错误: 无法打开串口 {port}: {e}")
            sys.exit(1)

    def close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"串口 {self.port_name} 已关闭")

    def _calculate_checksum(self, data):
        """计算校验和"""
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id, instruction, parameters=None):
        """发送数据包"""
        if not self.serial_port or not self.serial_port.is_open:
            return False
        if parameters is None:
            parameters = []
        
        length = len(parameters) + 2
        packet_core = [servo_id, length, instruction] + parameters
        checksum = self._calculate_checksum(packet_core)
        packet = bytes([0xFF, 0xFF] + packet_core + [checksum])
        
        try:
            self.serial_port.reset_input_buffer()
            self.serial_port.write(packet)
            self.serial_port.flush()
            return True
        except Exception:
            return False

    def _read_packet(self):
        """读取数据包"""
        start_time = time.time()
        packet = []
        
        while (time.time() - start_time) < self.serial_port.timeout:
            if self.serial_port.in_waiting > 0:
                byte = self.serial_port.read(1)
                if not byte:
                    continue
                byte = byte[0]

                if not packet and byte != 0xFF:
                    continue
                
                packet.append(byte)

                if len(packet) >= 2 and packet[-2:] == [0xFF, 0xFF]:
                    if len(packet) > 2:
                        packet = [0xFF, 0xFF]
                    continue

                if len(packet) > 4:
                    pkt_len = packet[3]
                    if len(packet) == pkt_len + 4:
                        core_data = packet[2:-1]
                        calculated_checksum = self._calculate_checksum(core_data)
                        if calculated_checksum == packet[-1]:
                            return self.COMM_SUCCESS, packet[4], packet[5:-1]
                        else:
                            return self.COMM_RX_CORRUPT, 0, []
        
        return self.COMM_RX_TIMEOUT, 0, []

    def get_servo_angle(self, servo_id):
        """读取舵机角度"""
        if not self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 2]):
            return None

        result, error, data = self._read_packet()

        if result != self.COMM_SUCCESS or error != 0:
            return None
        
        if data and len(data) >= 2:
            position = data[0] | (data[1] << 8)
            angle = ((position - 1024.0) / (3072.0 - 1024.0)) * 180.0 - 90.0
            angle = max(-90.0, min(90.0, angle))
            return angle
        
        return None

    def _write_register(self, servo_id, address, value, size=2):
        """写入寄存器"""
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        
        return self._send_packet(servo_id, self.INST_WRITE, params)

    def enable_torque(self, servo_id):
        """使能舵机扭矩"""
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """设置舵机角度 (-90 到 90 度)"""
        # 将角度映射到位置值 (1024 到 3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)


class ArmInverseController:
    """机械臂反解控制系统"""
    
    def __init__(self, port="COM4", baudrate=1000000):
        """初始化控制系统"""
        # 初始化各个模块
        self.servo_controller = ServoController(port, baudrate)
        self.forward_kinematics = GenkiArmForwardKinematics()
        self.inverse_kinematics = GenkiArmInverseKinematics()
        
        # 舵机ID列表
        self.servo_ids = list(range(1, 7))
        
        # 移动步长 (米)
        self.move_step = 0.01  # 1cm
        
        # 当前状态
        self.current_joint_angles = [0.0] * 6  # 弧度
        self.current_position = [0.0, 0.0, 0.0]  # 米
        
        # 使能所有舵机
        self._enable_all_servos()
        
        # 读取初始状态
        self.update_current_state()
        
        print("机械臂反解控制系统初始化完成")
        print(f"当前末端位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")

    def _enable_all_servos(self):
        """使能所有舵机"""
        print("正在使能所有舵机...")
        for servo_id in self.servo_ids:
            self.servo_controller.enable_torque(servo_id)
            time.sleep(0.05)
        print("所有舵机已使能")

    def read_current_joint_angles(self):
        """读取当前所有关节角度"""
        angles_deg = []
        for servo_id in self.servo_ids:
            angle = self.servo_controller.get_servo_angle(servo_id)
            if angle is not None:
                angles_deg.append(angle)
            else:
                print(f"警告: 无法读取舵机 {servo_id} 的角度")
                angles_deg.append(0.0)  # 使用默认值
        
        # 转换为弧度
        angles_rad = [math.radians(angle) for angle in angles_deg]
        return angles_rad

    def calculate_current_position(self):
        """基于当前关节角度计算末端位置"""
        try:
            position = self.forward_kinematics.forward_kinematics(self.current_joint_angles)
            return position.tolist()
        except Exception as e:
            print(f"正运动学计算错误: {e}")
            return [0.0, 0.0, 0.0]

    def update_current_state(self):
        """更新当前状态（关节角度和末端位置）"""
        self.current_joint_angles = self.read_current_joint_angles()
        self.current_position = self.calculate_current_position()

    def move_to_position(self, target_position, verbose=False):
        """移动到目标位置"""
        try:
            # 使用逆运动学求解
            joint_angles, success, info = self.inverse_kinematics.inverse_kinematics_multiple_attempts(
                target_position, num_attempts=3, verbose=verbose
            )
            
            if success:
                # 发送角度到舵机
                angles_deg = [math.degrees(angle) for angle in joint_angles]
                
                for i, servo_id in enumerate(self.servo_ids):
                    if not self.servo_controller.set_servo_angle(servo_id, angles_deg[i]):
                        print(f"警告: 无法设置舵机 {servo_id} 角度")
                
                # 等待移动完成
                time.sleep(0.5)
                
                # 更新当前状态
                self.update_current_state()
                
                return True, info['final_error']
            else:
                print(f"逆运动学求解失败: 误差 {info['final_error']:.6f}m")
                return False, info['final_error']
                
        except Exception as e:
            print(f"移动到位置时发生错误: {e}")
            return False, float('inf')

    def move_up(self, step=None):
        """向上移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[2] += step  # Z轴向上
        
        print(f"向上移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def move_down(self, step=None):
        """向下移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[2] -= step  # Z轴向下
        
        print(f"向下移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def move_left(self, step=None):
        """向左移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[1] += step  # Y轴向左
        
        print(f"向左移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def move_right(self, step=None):
        """向右移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[1] -= step  # Y轴向右
        
        print(f"向右移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def move_forward(self, step=None):
        """向前移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[0] += step  # X轴向前
        
        print(f"向前移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def move_backward(self, step=None):
        """向后移动"""
        if step is None:
            step = self.move_step
        
        target_position = self.current_position.copy()
        target_position[0] -= step  # X轴向后
        
        print(f"向后移动 {step*1000:.1f}mm...")
        success, error = self.move_to_position(target_position)
        
        if success:
            print(f"移动成功! 新位置: x={self.current_position[0]:.4f}m, y={self.current_position[1]:.4f}m, z={self.current_position[2]:.4f}m")
        else:
            print(f"移动失败! 误差: {error:.6f}m")
        
        return success

    def get_status_info(self):
        """获取当前状态信息"""
        angles_deg = [math.degrees(angle) for angle in self.current_joint_angles]
        
        info = {
            'joint_angles_deg': angles_deg,
            'joint_angles_rad': self.current_joint_angles,
            'end_position': self.current_position,
            'move_step_mm': self.move_step * 1000
        }
        
        return info

    def close(self):
        """关闭控制系统"""
        self.servo_controller.close()
        print("机械臂控制系统已关闭")


def create_control_gui(controller):
    """创建控制GUI界面"""
    
    # 创建图形界面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('机械臂反解控制系统', fontsize=16)
    
    # 左侧：控制按钮
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('方向控制')
    ax1.axis('off')
    
    # 右侧：状态显示
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('当前状态')
    ax2.axis('off')
    
    # 创建控制按钮
    button_size = 0.08
    button_spacing = 0.12
    
    # 按钮位置 (相对于figure)
    buttons = {}
    
    # 上下左右前后按钮布局
    button_configs = [
        ('up', [0.15, 0.7, button_size, button_size], '↑\n上'),
        ('down', [0.15, 0.5, button_size, button_size], '↓\n下'),
        ('left', [0.05, 0.6, button_size, button_size], '←\n左'),
        ('right', [0.25, 0.6, button_size, button_size], '→\n右'),
        ('forward', [0.15, 0.8, button_size, button_size], '↗\n前'),
        ('backward', [0.15, 0.4, button_size, button_size], '↙\n后'),
    ]
    
    # 创建按钮
    for name, pos, label in button_configs:
        ax_button = plt.axes(pos)
        button = Button(ax_button, label)
        buttons[name] = button
    
    # 状态文本
    status_text = ax2.text(0.1, 0.8, '', fontsize=10, verticalalignment='top', 
                          transform=ax2.transAxes, fontfamily='sans-serif')
    
    def update_status_display():
        """更新状态显示"""
        try:
            print("正在更新状态显示...")
            controller.update_current_state()
            info = controller.get_status_info()
            
            status_str = f"""当前末端位置:
X: {info['end_position'][0]:8.4f} m
Y: {info['end_position'][1]:8.4f} m  
Z: {info['end_position'][2]:8.4f} m

当前关节角度:
关节1: {info['joint_angles_deg'][0]:6.1f}°
关节2: {info['joint_angles_deg'][1]:6.1f}°
关节3: {info['joint_angles_deg'][2]:6.1f}°
关节4: {info['joint_angles_deg'][3]:6.1f}°
关节5: {info['joint_angles_deg'][4]:6.1f}°
关节6: {info['joint_angles_deg'][5]:6.1f}°

移动步长: {info['move_step_mm']:.1f} mm"""
            
            status_text.set_text(status_str)
            fig.canvas.draw()
            print("状态显示更新完成")
        except Exception as e:
            print(f"更新状态显示出错: {e}")
            status_text.set_text(f"状态更新失败: {e}")
            fig.canvas.draw()
    
    # 按钮回调函数
    def on_up_click(event):
        print("点击了向上按钮")
        try:
            controller.move_up()
            update_status_display()
        except Exception as e:
            print(f"向上移动出错: {e}")
    
    def on_down_click(event):
        print("点击了向下按钮")
        try:
            controller.move_down()
            update_status_display()
        except Exception as e:
            print(f"向下移动出错: {e}")
    
    def on_left_click(event):
        print("点击了向左按钮")
        try:
            controller.move_left()
            update_status_display()
        except Exception as e:
            print(f"向左移动出错: {e}")
    
    def on_right_click(event):
        print("点击了向右按钮")
        try:
            controller.move_right()
            update_status_display()
        except Exception as e:
            print(f"向右移动出错: {e}")
    
    def on_forward_click(event):
        print("点击了向前按钮")
        try:
            controller.move_forward()
            update_status_display()
        except Exception as e:
            print(f"向前移动出错: {e}")
    
    def on_backward_click(event):
        print("点击了向后按钮")
        try:
            controller.move_backward()
            update_status_display()
        except Exception as e:
            print(f"向后移动出错: {e}")
    
    # 绑定按钮事件
    buttons['up'].on_clicked(on_up_click)
    buttons['down'].on_clicked(on_down_click)
    buttons['left'].on_clicked(on_left_click)
    buttons['right'].on_clicked(on_right_click)
    buttons['forward'].on_clicked(on_forward_click)
    buttons['backward'].on_clicked(on_backward_click)
    
    # 初始状态显示
    update_status_display()
    
    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("机械臂反解控制系统")
    print("=" * 60)
    
    controller = None
    try:
        # 初始化控制器
        controller = ArmInverseController(port="COM4", baudrate=1000000)
        
        # 显示初始状态
        info = controller.get_status_info()
        print(f"\n初始状态:")
        print(f"末端位置: x={info['end_position'][0]:.4f}m, y={info['end_position'][1]:.4f}m, z={info['end_position'][2]:.4f}m")
        print(f"关节角度: {[f'{angle:.1f}°' for angle in info['joint_angles_deg']]}")
        
        # 创建GUI
        print("\n正在启动控制界面...")
        fig = create_control_gui(controller)
        
        print("控制界面已启动!")
        print("使用按钮控制机械臂移动，关闭窗口退出程序")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if controller:
            controller.close()
        print("程序结束")


if __name__ == "__main__":
    main()