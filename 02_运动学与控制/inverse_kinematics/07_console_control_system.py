#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂反解控制系统 - 控制台版本
基于当前位置进行六方向移动控制
使用键盘操作，无GUI界面
"""

import serial
import time
import sys
import numpy as np
import math
import threading
import msvcrt  # Windows下的键盘输入
from typing import Tuple, List, Optional

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
            print(f"无法打开串口 {port}: {e}")
            self.serial_port = None

    def close(self):
        """关闭串口"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")

    def _calculate_checksum(self, data):
        """计算校验和"""
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id, instruction, parameters=None):
        """发送数据包"""
        if not self.serial_port or not self.serial_port.is_open:
            return self.COMM_RX_TIMEOUT
        
        if parameters is None:
            parameters = []
        
        length = len(parameters) + 2
        packet = [0xFF, 0xFF, servo_id, length, instruction] + parameters
        checksum = self._calculate_checksum(packet[2:])
        packet.append(checksum)
        
        try:
            self.serial_port.write(bytes(packet))
            return self.COMM_SUCCESS
        except Exception as e:
            print(f"发送数据包失败: {e}")
            return self.COMM_RX_TIMEOUT

    def _read_packet(self):
        """读取响应数据包"""
        if not self.serial_port or not self.serial_port.is_open:
            return self.COMM_RX_TIMEOUT, []
        
        try:
            # 等待并读取包头
            header = self.serial_port.read(2)
            if len(header) != 2 or header != b'\xff\xff':
                return self.COMM_RX_TIMEOUT, []
            
            # 读取ID和长度
            id_length = self.serial_port.read(2)
            if len(id_length) != 2:
                return self.COMM_RX_TIMEOUT, []
            
            servo_id, length = id_length
            
            # 读取剩余数据
            remaining_data = self.serial_port.read(length)
            if len(remaining_data) != length:
                return self.COMM_RX_TIMEOUT, []
            
            # 验证校验和
            packet_data = [servo_id, length] + list(remaining_data[:-1])
            expected_checksum = self._calculate_checksum(packet_data)
            actual_checksum = remaining_data[-1]
            
            if expected_checksum != actual_checksum:
                return self.COMM_RX_CORRUPT, []
            
            return self.COMM_SUCCESS, list(remaining_data[1:-1])  # 返回参数部分
            
        except Exception as e:
            print(f"读取数据包失败: {e}")
            return self.COMM_RX_TIMEOUT, []

    def get_servo_angle(self, servo_id):
        """读取舵机角度"""
        # 发送读取位置命令
        result = self._send_packet(servo_id, self.INST_READ, [self.ADDR_PRESENT_POSITION, 2])
        if result != self.COMM_SUCCESS:
            return None
        
        # 读取响应
        comm_result, data = self._read_packet()
        if comm_result != self.COMM_SUCCESS or len(data) < 2:
            return None
        
        # 解析位置数据 (小端序)
        position = data[0] + (data[1] << 8)
        # 转换为角度 (假设0-1023对应0-300度)
        angle = (position / 1023.0) * 300.0 - 150.0
        return angle

    def _write_register(self, servo_id, address, value, size=2):
        """写入寄存器"""
        if size == 2:
            # 16位值，小端序
            low_byte = value & 0xFF
            high_byte = (value >> 8) & 0xFF
            parameters = [address, low_byte, high_byte]
        else:
            parameters = [address, value & 0xFF]
        
        return self._send_packet(servo_id, self.INST_WRITE, parameters)

    def enable_torque(self, servo_id):
        """使能舵机扭矩"""
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """设置舵机角度"""
        # 角度范围限制 (-150 到 150 度)
        angle = max(-150, min(150, angle))
        # 转换为位置值 (0-1023)
        position = int((angle + 150.0) / 300.0 * 1023)
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position)


class ArmConsoleController:
    """机械臂控制台控制器"""
    
    def __init__(self, port="COM4", baudrate=1000000):
        # 初始化串口控制器
        self.servo_controller = ServoController(port, baudrate)
        
        # 初始化运动学
        self.forward_kinematics = GenkiArmForwardKinematics()
        self.inverse_kinematics = GenkiArmInverseKinematics()
        
        # 控制参数
        self.move_step = 0.01  # 移动步长，单位：米 (10mm)
        self.current_position = np.array([0.0, 0.0, 0.0])  # 当前末端位置
        self.current_joint_angles = np.zeros(6)  # 当前关节角度
        
        # 使能所有舵机
        self._enable_all_servos()
        
        # 设置初始姿态（避免奇异点）
        self._set_initial_pose()
        
        # 读取初始状态
        self.update_current_state()
        
        print("机械臂控制台控制器初始化完成")

    def _set_initial_pose(self):
        """设置初始姿态，避免奇异点"""
        print("设置初始姿态...")
        
        # 设置一个更好的工作配置，提高各轴可操作性
        # 这个姿态让机械臂处于一个更均衡的工作状态
        initial_angles = [0.0, -0.3, 0.6, 0.0, -0.6, 0.0]  # 弧度
        
        try:
            # 转换为舵机角度并设置
            for i, angle in enumerate(initial_angles):
                servo_angle = math.degrees(angle) + 90  # 转换为舵机角度 (0-180)
                servo_angle = max(0, min(180, servo_angle))  # 限制范围
                self.servo_controller.set_servo_angle(i + 1, servo_angle)
            
            # 等待运动完成
            time.sleep(2.0)
            print("初始姿态设置完成")
            
        except Exception as e:
            print(f"设置初始姿态失败: {e}")
            print("使用默认姿态")

    def _enable_all_servos(self):
        """使能所有舵机"""
        print("正在使能所有舵机...")
        for servo_id in range(1, 7):
            result = self.servo_controller.enable_torque(servo_id)
            if result == self.servo_controller.COMM_SUCCESS:
                print(f"舵机 {servo_id} 使能成功")
            else:
                print(f"舵机 {servo_id} 使能失败")

    def read_current_joint_angles(self):
        """读取当前关节角度"""
        angles = []
        print("正在读取关节角度...")
        for servo_id in range(1, 7):
            angle = self.servo_controller.get_servo_angle(servo_id)
            if angle is not None:
                angle_rad = math.radians(angle)
                angles.append(angle_rad)  # 转换为弧度
                print(f"舵机 {servo_id}: {angle:.2f}° ({angle_rad:.4f} rad)")
            else:
                print(f"读取舵机 {servo_id} 角度失败，使用默认值0")
                angles.append(0.0)  # 使用默认值
        
        self.current_joint_angles = np.array(angles)
        print(f"当前关节角度: {[f'{a:.4f}' for a in angles]} (弧度)")
        return self.current_joint_angles

    def calculate_current_position(self):
        """计算当前末端位置"""
        try:
            position = self.forward_kinematics.forward_kinematics(self.current_joint_angles)
            if position is not None and len(position) >= 3:
                self.current_position = np.array(position[:3])  # 只取前3个元素(x,y,z)
                print(f"正运动学计算成功: 位置 = [{self.current_position[0]:.4f}, {self.current_position[1]:.4f}, {self.current_position[2]:.4f}]")
            else:
                print(f"正运动学返回无效结果: {position}")
                # 使用默认位置
                self.current_position = np.array([0.0, 0.0, 0.3])  # 默认高度30cm
            return self.current_position
        except Exception as e:
            print(f"正运动学计算失败: {e}")
            # 使用默认位置
            self.current_position = np.array([0.0, 0.0, 0.3])  # 默认高度30cm
            return self.current_position

    def update_current_state(self):
        """更新当前状态"""
        self.read_current_joint_angles()
        self.calculate_current_position()

    def move_to_position(self, target_position, verbose=True):
        """移动到目标位置"""
        try:
            print(f"目标位置: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]")
            print(f"当前位置: [{self.current_position[0]:.4f}, {self.current_position[1]:.4f}, {self.current_position[2]:.4f}]")
            
            # 使用逆运动学求解
            joint_angles, success, info = self.inverse_kinematics.inverse_kinematics(
                target_position, self.current_joint_angles, verbose=verbose
            )
            
            if not success:
                error = info.get('final_error', float('inf'))
                if verbose:
                    print(f"逆运动学求解失败，误差: {error:.6f}m")
                return False, error
            
            print(f"逆运动学求解成功，误差: {info.get('final_error', 0):.6f}m")
            print(f"目标关节角度: {[f'{math.degrees(a):.2f}°' for a in joint_angles]}")
            
            # 设置关节角度
            for i, angle in enumerate(joint_angles):
                servo_id = i + 1
                angle_deg = math.degrees(angle)
                result = self.servo_controller.set_servo_angle(servo_id, angle_deg)
                if result != self.servo_controller.COMM_SUCCESS:
                    if verbose:
                        print(f"设置舵机 {servo_id} 角度失败")
            
            # 等待运动完成
            time.sleep(0.5)
            
            # 更新当前状态
            self.update_current_state()
            
            # 计算误差
            error = np.linalg.norm(self.current_position - target_position)
            
            return error < 0.02, error  # 2cm误差容限
            
        except Exception as e:
            if verbose:
                print(f"移动到目标位置失败: {e}")
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

    def print_status(self):
        """打印当前状态"""
        print("\n" + "="*50)
        print("当前机械臂状态:")
        print(f"末端位置: x={self.current_position[0]:8.4f}m, y={self.current_position[1]:8.4f}m, z={self.current_position[2]:8.4f}m")
        print("关节角度:")
        for i, angle in enumerate(self.current_joint_angles):
            print(f"  关节{i+1}: {math.degrees(angle):6.1f}°")
        print(f"移动步长: {self.move_step*1000:.1f}mm")
        print("="*50)

    def set_move_step(self, step_mm):
        """设置移动步长"""
        self.move_step = step_mm / 1000.0  # 转换为米
        print(f"移动步长设置为: {step_mm:.1f}mm")

    def close(self):
        """关闭控制器"""
        if self.servo_controller:
            self.servo_controller.close()


def print_help():
    """打印帮助信息"""
    print("\n" + "="*60)
    print("机械臂控制台控制系统 - 操作说明")
    print("="*60)
    print("方向控制:")
    print("  w/W - 向上移动")
    print("  s/S - 向下移动")
    print("  a/A - 向左移动")
    print("  d/D - 向右移动")
    print("  q/Q - 向前移动")
    print("  e/E - 向后移动")
    print("\n其他操作:")
    print("  p/P - 打印当前状态")
    print("  r/R - 刷新当前状态")
    print("  +   - 增加移动步长")
    print("  -   - 减少移动步长")
    print("  h/H - 显示帮助")
    print("  ESC - 退出程序")
    print("="*60)


def main():
    """主函数"""
    print("=" * 60)
    print("机械臂反解控制系统 - 控制台版本")
    print("=" * 60)
    
    controller = None
    try:
        # 初始化控制器
        controller = ArmConsoleController(port="COM4", baudrate=1000000)
        
        # 显示初始状态
        controller.print_status()
        
        # 显示帮助信息
        print_help()
        
        print("\n系统就绪，等待键盘输入...")
        
        # 主控制循环
        while True:
            try:
                # 等待键盘输入
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    
                    if key == '\x1b':  # ESC键
                        print("\n退出程序...")
                        break
                    elif key in ['w']:
                        controller.move_up()
                    elif key in ['s']:
                        controller.move_down()
                    elif key in ['a']:
                        controller.move_left()
                    elif key in ['d']:
                        controller.move_right()
                    elif key in ['q']:
                        controller.move_forward()
                    elif key in ['e']:
                        controller.move_backward()
                    elif key in ['p']:
                        controller.print_status()
                    elif key in ['r']:
                        print("刷新状态...")
                        controller.update_current_state()
                        controller.print_status()
                    elif key == '+':
                        new_step = controller.move_step * 1000 + 1
                        controller.set_move_step(min(50, new_step))  # 最大50mm
                    elif key == '-':
                        new_step = controller.move_step * 1000 - 1
                        controller.set_move_step(max(1, new_step))   # 最小1mm
                    elif key in ['h']:
                        print_help()
                    else:
                        print(f"未知命令: {key}, 按 h 查看帮助")
                
                time.sleep(0.01)  # 短暂延时，避免CPU占用过高
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"\n控制循环出错: {e}")
                continue
        
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if controller:
            controller.close()
        print("程序结束")


if __name__ == "__main__":
    main()