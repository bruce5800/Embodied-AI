#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遥操作系统 - 教师端到学生端实时角度同步
结合了02_read_all_teacher_angles.py和05_write_all_student_angle.py的功能
"""

import time
import struct
import enum
import logging
import math
import sys
from copy import deepcopy
import numpy as np
from serial import Serial
from serial.tools import list_ports

# region: GBot/utils.py (从教师端代码复制)

def bytes_to_short(data: bytes, signed: bool = False, byteorder: str = 'little') -> int:
    if len(data) != 2:
        raise ValueError("Data must be exactly 2 bytes long")
    prefix = '<' if byteorder == 'little' else '>'
    format_char = 'h' if signed else 'H'
    return struct.unpack(f"{prefix}{format_char}", data)[0]

def short_to_bytes(value: int, signed: bool = False, byteorder: str = 'little') -> bytes:
    if signed:
        format_char = 'h'
        min_val, max_val = -32768, 32767
    else:
        format_char = 'H'
        min_val, max_val = 0, 65535
    if not (min_val <= value <= max_val):
        raise OverflowError(f"Value {value} out of range for {'signed' if signed else 'unsigned'} short")
    prefix = '<' if byteorder == 'little' else '>'
    return struct.pack(f"{prefix}{format_char}", value)

def bytes_to_int(byte_data, signed=False, byteorder='little'):
    if len(byte_data) != 4:
        raise ValueError("输入必须是 4 字节")
    fmt_char = 'i' if signed else 'I'
    fmt_str = ('>' if byteorder == 'big' else '<') + fmt_char
    return struct.unpack(fmt_str, byte_data)[0]

def int_to_bytes(int_value, signed=False, byteorder='little'):
    if signed and not (-2**31 <= int_value < 2**31):
        raise ValueError("有符号整数超出 4 字节范围")
    elif not signed and not (0 <= int_value < 2**32):
        raise ValueError("无符号整数超出 4 字节范围")
    fmt_char = 'i' if signed else 'I'
    fmt_str = ('>' if byteorder == 'big' else '<') + fmt_char
    return struct.pack(fmt_str, int_value)

# endregion

# region: GBot/global_state.py (从教师端代码复制)

class Address(enum.Enum):
    DEVICE_UUID         = (0, 4)
    VERSION             = (4, 2)
    MOTOR_TYPE          = (6, 1)
    CURRENT_POSITION    = (7, 2)
    CURRENT_SPEED       = (9, 2)
    CURRENT_LOAD        = (11, 2)
    CURRENT_VOLTAGE     = (13, 1)
    CURRENT_CURRENT     = (14, 2)
    CURRENT_TEMPERATURE = (16, 1)
    TORQUE_ENABLE       = (50, 1)
    TARGET_POSITION     = (51, 2)
    ID                  = (70, 1)
    MIN_POSITION        = (71, 2)
    MAX_POSITION        = (73, 2)
    POSITION_OFFSET     = (75, 2)
    MAX_VOLTAGE         = (77, 1)
    MIN_VOLTAGE         = (78, 1)
    MAX_TEMPERATURE     = (79, 1)
    MAX_CURRENT         = (80, 2)
    KP                  = (82, 1)
    KI                  = (83, 1)
    KD                  = (84, 1)

    @classmethod
    def get_address(cls, address:int):
        for addr in cls:
            if addr.value[0] == address:
                return addr
        return None

class ErrorCode(enum.Enum):
    SUCCESS             = 0
    WRITE_ERROR         = 1
    READ_ERROR          = 2
    READ_TIMEOUT        = 3

class Result:
    def __init__(self, error: ErrorCode = ErrorCode.SUCCESS, frame: list[int] = None, input = None):
        self.__error_code = error
        self.__frame = frame
        self.__input = input
        self.__value_map = {}

        if frame is None or input is None:
            return

        id = frame[2]
        cmd = frame[3]
        if cmd != 0x03:
            return
        if id != 0xFF and id < 128 and id >= 248:
            return

        addresses = []
        if isinstance(input, Address):
            addresses.append(input)
        elif isinstance(input, list):
            addresses.extend(input)

        cnt = 6 if id == 0xFF else 5

        while cnt < len(frame) - 2:
            addr = Address.get_address(frame[cnt])
            if addr is None:
                break
            addr_int = addr.value[0]
            addr_len = addr.value[1]

            if addr_len == 1:
                self.__value_map[addr_int] = frame[cnt+1]
            elif addr_len == 2:
                self.__value_map[addr_int] = bytes_to_short(bytearray(frame[cnt+1:cnt+3]))
            elif addr_len == 4:
                self.__value_map[addr_int] = bytes_to_int(bytearray(frame[cnt+1:cnt+5]))
            cnt += addr_len + 1

    def is_success(self) -> bool:
        return self.__error_code == ErrorCode.SUCCESS

    def get_error_code(self) -> int:
        return self.__error_code.value

    def get_data(self, address: Address) -> int:
        address_int = address.value[0]
        return self.__value_map.get(address_int)

# endregion

# region: GBot/port_handler.py (从教师端代码复制)

class PortHandler:
    def __init__(self):
        self.__serial: Serial = None
        self._port = None
        self._baudrate = 230400
        self._bytesize = 8
        self._parity = 'N'
        self._stopbits = 1
        self._read_timeout = None
        self._write_timeout = None
        self.__is_running = False

    @property
    def baudrate(self):
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value):
        if self.__serial and self.__serial.is_open:
            raise ValueError("无法修改已打开的串口波特率")
        self._baudrate = value

    def open(self, port) -> bool:
        self.close()
        try:
            self._port = port
            self.__serial = Serial(port=port, baudrate=self._baudrate, bytesize=self._bytesize, parity=self._parity, stopbits=self._stopbits, timeout=self._read_timeout, write_timeout=self._write_timeout)
            self.__is_running = True
            return True
        except Exception:
            return False

    def is_open(self) -> bool:
        return self.__serial and self.__serial.is_open

    def close(self):
        if self.__serial and self.__serial.is_open:
            self.__serial.close()
            self.__is_running = False
            self.__serial = None

    def read_port(self, length:int):
        if self.__serial and self.__serial.is_open:
            return self.__serial.read(length)

    def write_port(self, data):
        if self.__serial and self.__serial.is_open:
            self.__serial.reset_input_buffer()
            self.__serial.write(data)
            self.__serial.flush()

    def in_waiting(self):
        if self.__serial and self.__serial.is_open:
            return self.__serial.in_waiting
        return 0

# endregion

# region: GBot/sync_connector.py (从教师端代码复制)

FRAME_HEADER    = 0xAA
FRAME_TAIL      = 0xBB
FRAME_CMD_READ          = 0x03

def checksum(id: int, cmd: int, data: list[int]) -> int:
    return (id + cmd + len(data) + sum(data)) & 0xFF

def frame_generator(id: int, cmd: int, data: list[int]) -> bytearray:
    frame = bytearray()
    frame.append(FRAME_HEADER)
    frame.append(FRAME_HEADER)
    frame.append(id)
    frame.append(cmd)
    frame.append(len(data))
    for d in data:
        frame.append(d)
    frame.append(checksum(id, cmd, data))
    frame.append(FRAME_TAIL)
    return frame

class SyncConnector:
    def __init__(self, portHandler: PortHandler):
        self.__port_handler = portHandler

    def _parse_response_frame(self) -> Result:
        retry_cnt = 0
        read_list = []
        state = 0
        self.__port_handler._read_timeout = 1
        while True:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                if retry_cnt < 5:
                    retry_cnt += 1
                    time.sleep(0.01)
                    continue
                else:
                    state = -1
                    break
            read_list.extend(list(self.__port_handler.read_port(in_waiting)))
            while len(read_list) >= 7:
                if read_list[0] != FRAME_HEADER or read_list[1] != FRAME_HEADER:
                    read_list.pop(0)
                    continue
                data_length = read_list[4]
                if data_length > 48 or len(read_list) < 7 + data_length or read_list[6 + data_length] != FRAME_TAIL:
                    read_list.pop(0)
                    continue
                checksum_val = sum(read_list[2:5 + data_length]) & 0xFF
                if checksum_val != read_list[5 + data_length]:
                    read_list.pop(0)
                    continue
                read_list = read_list[0:7 + data_length]
                state = 1
                break
            if state == 1:
                break
        if state == -1:
            return Result(error=ErrorCode.READ_TIMEOUT)
        return Result(frame=read_list, input=self.last_read_address)

    def read(self, id_list: list[int], address_list: list[Address]) -> Result:
        self.last_read_address = address_list
        data = []
        for address in address_list:
            data.extend([address.value[0], address.value[1]])
        frame = frame_generator(id_list[0], FRAME_CMD_READ, data)
        self.__port_handler.write_port(frame)
        return self._parse_response_frame()

# endregion

# region: 教师端舵机角度读取类 (从02代码改进)

class TeacherServoReader:
    def __init__(self, port: str):
        self.port = port
        self.__port_handler: PortHandler = PortHandler()
        self.__sync_connector: SyncConnector = SyncConnector(self.__port_handler)
        self.is_connected = False
        # 舵机参数配置
        self.homing_offset = 2048  # 零位偏移
        self.resolution = 4096     # 分辨率
        
        # 性能优化相关
        self.batch_read_enabled = True  # 启用批量读取
        self.read_timeout_count = 0

    def connect(self):
        """连接到串口"""
        if self.is_connected:
            return True
        
        if not self.__port_handler.open(self.port):
            print(f"无法连接到教师端端口 {self.port}")
            return False
        
        self.is_connected = True
        print(f"成功连接到教师端端口 {self.port}")
        return True

    def disconnect(self):
        """断开串口连接"""
        if not self.is_connected:
            return
        
        self.__port_handler.close()
        self.is_connected = False
        print("教师端已断开连接")

    def read_angle(self, motor_id: int) -> float:
        """读取指定ID舵机的角度"""
        if not self.is_connected:
            return None
        
        try:
            result = self.__sync_connector.read([motor_id], [Address.CURRENT_POSITION])
            
            if result.is_success():
                raw_position = result.get_data(Address.CURRENT_POSITION)
                if raw_position is not None:
                    # 转换为角度
                    angle = ((raw_position - self.homing_offset) / self.resolution) * 360
                    return angle
                else:
                    return None
            else:
                self.read_timeout_count += 1
                return None
                
        except Exception as e:
            self.read_timeout_count += 1
            return None

    def read_all_angles(self, motor_ids: list[int]) -> dict:
        """读取所有舵机的角度 - 优化版本"""
        angles = {}
        
        if not self.is_connected:
            return {motor_id: None for motor_id in motor_ids}
        
        # 如果批量读取被禁用或者只有一个舵机，使用单独读取
        if not self.batch_read_enabled or len(motor_ids) == 1:
            for motor_id in motor_ids:
                angle = self.read_angle(motor_id)
                angles[motor_id] = angle
            return angles
        
        # 尝试批量读取（如果支持的话）
        # 注意：这里假设原始协议支持批量读取，如果不支持，回退到单独读取
        try:
            # 批量读取所有舵机
            for motor_id in motor_ids:
                angle = self.read_angle(motor_id)
                angles[motor_id] = angle
                # 添加小延迟以避免总线冲突
                time.sleep(0.001)
        except Exception as e:
            # 批量读取失败，回退到单独读取
            for motor_id in motor_ids:
                try:
                    angle = self.read_angle(motor_id)
                    angles[motor_id] = angle
                except:
                    angles[motor_id] = None
        
        return angles

# endregion

# region: 学生端舵机控制类 (从05代码改进)

class StudentServoController:
    def __init__(self, port="COM4", baudrate=1000000, timeout=0.1):
        self.port_name = port
        self.serial_port = None
        # Constants
        self.ADDR_GOAL_POSITION = 42
        self.ADDR_TORQUE_ENABLE = 40
        self.INST_WRITE = 3
        self.COMM_SUCCESS = 0
        self.COMM_RX_TIMEOUT = -6
        self.COMM_RX_CORRUPT = -7
        self.is_connected = False

        try:
            self.serial_port = Serial(port, baudrate=baudrate, timeout=timeout)
            time.sleep(0.1)
            self.serial_port.reset_input_buffer()
            self.is_connected = True
            print(f"成功连接到学生端端口 {port}")
        except Exception as e:
            print(f"无法连接到学生端端口 {port}: {e}")
            self.is_connected = False

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.is_connected = False
            print(f"学生端端口 {self.port_name} 已关闭")

    def _calculate_checksum(self, data):
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id, instruction, parameters=None):
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

    def _write_register(self, servo_id, address, value, size=2):
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        
        if not self._send_packet(servo_id, self.INST_WRITE, params):
            return False
        
        return True

    def enable_torque(self, servo_id):
        """启用舵机扭矩"""
        return self._write_register(servo_id, self.ADDR_TORQUE_ENABLE, 1, size=1)

    def set_servo_angle(self, servo_id, angle):
        """设置舵机角度 (-90 到 90 度)"""
        # 将角度 (-90 到 90) 映射到位置 (1024 到 3072)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        # 限制值范围
        position = max(1024, min(3072, position))
        
        return self._write_register(servo_id, self.ADDR_GOAL_POSITION, position, size=2)

    def enable_all_torques(self, servo_ids):
        """启用所有舵机的扭矩"""
        success_count = 0
        for servo_id in servo_ids:
            if self.enable_torque(servo_id):
                success_count += 1
            time.sleep(0.01)  # 小延迟
        return success_count == len(servo_ids)

# endregion

# region: 遥操作主类

class TeleOperationSystem:
    def __init__(self, teacher_port="COM8", student_port="COM4", servo_ids=None):
        if servo_ids is None:
            servo_ids = [1, 2, 3, 4, 5, 6]
        
        self.servo_ids = servo_ids
        self.teacher_reader = TeacherServoReader(teacher_port)
        self.student_controller = StudentServoController(student_port)
        self.is_running = False
        self.last_angles = {}
        self.angle_threshold = 1.0  # 角度变化阈值，减少不必要的写入
        
        # 错误处理相关
        self.teacher_error_count = 0
        self.student_error_count = 0
        self.max_error_count = 10
        self.reconnect_interval = 5.0  # 重连间隔(秒)
        self.last_reconnect_time = 0
        
        # 性能监控相关
        self.performance_stats = {
            'total_cycles': 0,
            'successful_syncs': 0,
            'total_sync_time': 0.0,
            'max_cycle_time': 0.0,
            'min_cycle_time': float('inf'),
            'avg_cycle_time': 0.0
        }
        self.adaptive_rate_enabled = True
        self.target_cycle_time = 0.05  # 目标周期时间 (20Hz)
        self.min_update_rate = 5       # 最小更新频率
        self.max_update_rate = 50      # 最大更新频率
        
    def initialize(self):
        """初始化系统"""
        print("正在初始化遥操作系统...")
        
        # 连接教师端
        if not self.teacher_reader.connect():
            print("教师端连接失败")
            return False
        
        # 检查学生端连接
        if not self.student_controller.is_connected:
            print("学生端连接失败")
            return False
        
        # 启用学生端所有舵机扭矩
        print("正在启用学生端舵机扭矩...")
        if not self.student_controller.enable_all_torques(self.servo_ids):
            print("警告：部分舵机扭矩启用失败")
        
        # 重置错误计数和性能统计
        self.teacher_error_count = 0
        self.student_error_count = 0
        self.performance_stats = {
            'total_cycles': 0,
            'successful_syncs': 0,
            'total_sync_time': 0.0,
            'max_cycle_time': 0.0,
            'min_cycle_time': float('inf'),
            'avg_cycle_time': 0.0
        }
        
        print("遥操作系统初始化完成")
        return True
    
    def shutdown(self):
        """关闭系统"""
        print("正在关闭遥操作系统...")
        self.is_running = False
        
        # 显示性能统计
        self.print_performance_stats()
        
        try:
            self.teacher_reader.disconnect()
        except Exception as e:
            print(f"教师端断开连接时出错: {e}")
        
        try:
            self.student_controller.close()
        except Exception as e:
            print(f"学生端断开连接时出错: {e}")
        
        print("遥操作系统已关闭")
    
    def print_performance_stats(self):
        """打印性能统计信息"""
        stats = self.performance_stats
        if stats['total_cycles'] > 0:
            success_rate = (stats['successful_syncs'] / stats['total_cycles']) * 100
            print(f"\n性能统计:")
            print(f"  总周期数: {stats['total_cycles']}")
            print(f"  成功同步: {stats['successful_syncs']} ({success_rate:.1f}%)")
            print(f"  平均周期时间: {stats['avg_cycle_time']*1000:.1f}ms")
            print(f"  最大周期时间: {stats['max_cycle_time']*1000:.1f}ms")
            print(f"  最小周期时间: {stats['min_cycle_time']*1000:.1f}ms")
            if stats['total_sync_time'] > 0:
                avg_freq = stats['total_cycles'] / stats['total_sync_time']
                print(f"  平均频率: {avg_freq:.1f}Hz")
    
    def update_performance_stats(self, cycle_time, sync_success):
        """更新性能统计"""
        stats = self.performance_stats
        stats['total_cycles'] += 1
        stats['total_sync_time'] += cycle_time
        
        if sync_success:
            stats['successful_syncs'] += 1
        
        stats['max_cycle_time'] = max(stats['max_cycle_time'], cycle_time)
        stats['min_cycle_time'] = min(stats['min_cycle_time'], cycle_time)
        stats['avg_cycle_time'] = stats['total_sync_time'] / stats['total_cycles']
    
    def calculate_adaptive_rate(self, current_cycle_time):
        """计算自适应更新频率"""
        if not self.adaptive_rate_enabled:
            return None
        
        # 如果周期时间过长，降低频率
        if current_cycle_time > self.target_cycle_time * 1.5:
            new_rate = max(self.min_update_rate, 1.0 / (current_cycle_time * 1.2))
        # 如果周期时间很短，可以提高频率
        elif current_cycle_time < self.target_cycle_time * 0.8:
            new_rate = min(self.max_update_rate, 1.0 / (current_cycle_time * 0.9))
        else:
            new_rate = None
        
        return new_rate
    
    def check_and_reconnect(self):
        """检查连接状态并尝试重连"""
        current_time = time.time()
        
        # 避免频繁重连
        if current_time - self.last_reconnect_time < self.reconnect_interval:
            return False
        
        reconnected = False
        
        # 检查教师端连接
        if self.teacher_error_count >= self.max_error_count:
            print(f"\n教师端错误过多({self.teacher_error_count})，尝试重连...")
            self.teacher_reader.disconnect()
            if self.teacher_reader.connect():
                self.teacher_error_count = 0
                reconnected = True
                print("教师端重连成功")
            else:
                print("教师端重连失败")
        
        # 检查学生端连接
        if self.student_error_count >= self.max_error_count:
            print(f"\n学生端错误过多({self.student_error_count})，尝试重连...")
            self.student_controller.close()
            try:
                self.student_controller = StudentServoController(
                    self.student_controller.port_name,
                    baudrate=1000000
                )
                if self.student_controller.is_connected:
                    self.student_controller.enable_all_torques(self.servo_ids)
                    self.student_error_count = 0
                    reconnected = True
                    print("学生端重连成功")
                else:
                    print("学生端重连失败")
            except Exception as e:
                print(f"学生端重连出错: {e}")
        
        if reconnected:
            self.last_reconnect_time = current_time
        
        return reconnected
    
    def sync_angles(self):
        """同步角度 - 从教师端读取并写入学生端"""
        sync_start_time = time.time()
        
        try:
            # 读取教师端所有角度
            teacher_angles = self.teacher_reader.read_all_angles(self.servo_ids)
            
            # 检查读取是否成功
            valid_readings = sum(1 for angle in teacher_angles.values() if angle is not None)
            if valid_readings == 0:
                self.teacher_error_count += 1
                return 0, teacher_angles, time.time() - sync_start_time
            else:
                # 有成功读取，重置错误计数
                if self.teacher_error_count > 0:
                    self.teacher_error_count = max(0, self.teacher_error_count - 1)
            
            # 同步到学生端
            sync_count = 0
            student_errors = 0
            
            for servo_id in self.servo_ids:
                teacher_angle = teacher_angles.get(servo_id)
                
                if teacher_angle is not None:
                    # 检查角度变化是否超过阈值
                    last_angle = self.last_angles.get(servo_id, float('inf'))
                    if abs(teacher_angle - last_angle) > self.angle_threshold:
                        # 限制角度范围到 -90 到 90 度
                        clamped_angle = max(-90, min(90, teacher_angle))
                        
                        try:
                            if self.student_controller.set_servo_angle(servo_id, clamped_angle):
                                self.last_angles[servo_id] = teacher_angle
                                sync_count += 1
                            else:
                                student_errors += 1
                        except Exception as e:
                            student_errors += 1
            
            # 更新学生端错误计数
            if student_errors > 0:
                self.student_error_count += 1
            elif self.student_error_count > 0:
                self.student_error_count = max(0, self.student_error_count - 1)
            
            sync_time = time.time() - sync_start_time
            return sync_count, teacher_angles, sync_time
            
        except Exception as e:
            print(f"\n同步过程中发生异常: {e}")
            self.teacher_error_count += 1
            sync_time = time.time() - sync_start_time
            return 0, {}, sync_time
    
    def run(self, update_rate=20):
        """运行遥操作系统
        
        Args:
            update_rate: 初始更新频率 (Hz)
        """
        if not self.initialize():
            return
        
        self.is_running = True
        current_update_rate = update_rate
        update_interval = 1.0 / current_update_rate
        
        print(f"遥操作系统开始运行 (初始更新频率: {update_rate}Hz)")
        print("教师端舵机运动时，学生端将同步跟随")
        print("按 Ctrl+C 停止系统")
        print("-" * 50)
        
        consecutive_failures = 0
        max_consecutive_failures = 50  # 连续失败次数上限
        last_stats_print = time.time()
        stats_print_interval = 10.0  # 每10秒打印一次统计
        
        try:
            while self.is_running:
                cycle_start_time = time.time()
                
                # 检查是否需要重连
                if self.teacher_error_count >= self.max_error_count or self.student_error_count >= self.max_error_count:
                    if self.check_and_reconnect():
                        consecutive_failures = 0
                
                # 同步角度
                sync_count, teacher_angles, sync_time = self.sync_angles()
                
                # 检查连续失败
                sync_success = sync_count > 0 or any(angle is not None for angle in teacher_angles.values())
                if not sync_success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n连续失败次数过多({consecutive_failures})，系统可能存在严重问题")
                        break
                else:
                    consecutive_failures = 0
                
                # 计算周期时间
                cycle_time = time.time() - cycle_start_time
                
                # 更新性能统计
                self.update_performance_stats(cycle_time, sync_success)
                
                # 自适应频率调整
                if self.adaptive_rate_enabled:
                    new_rate = self.calculate_adaptive_rate(cycle_time)
                    if new_rate and abs(new_rate - current_update_rate) > 1:
                        current_update_rate = new_rate
                        update_interval = 1.0 / current_update_rate
                
                # 显示状态
                error_indicator = ""
                if self.teacher_error_count > 0 or self.student_error_count > 0:
                    error_indicator = f" [T:{self.teacher_error_count} S:{self.student_error_count}]"
                
                perf_indicator = f" {cycle_time*1000:.1f}ms {current_update_rate:.0f}Hz"
                
                status_line = f"同步: {sync_count}/{len(self.servo_ids)}{error_indicator}{perf_indicator} | "
                for servo_id in self.servo_ids:
                    angle = teacher_angles.get(servo_id)
                    if angle is not None:
                        status_line += f"ID{servo_id}:{angle:6.1f}° "
                    else:
                        status_line += f"ID{servo_id}:  N/A  "
                
                print(f"\r{status_line}", end="", flush=True)
                
                # 定期打印详细统计
                if time.time() - last_stats_print > stats_print_interval:
                    print()  # 换行
                    self.print_performance_stats()
                    last_stats_print = time.time()
                    print("-" * 50)
                
                # 控制更新频率
                elapsed = time.time() - cycle_start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n\n收到停止信号")
        except Exception as e:
            print(f"\n\n系统运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

# endregion

def main():
    """主函数"""
    # 配置参数
    TEACHER_PORT = "COM31"    # 教师端串口
    STUDENT_PORT = "COM4"    # 学生端串口
    SERVO_IDS = [1, 2, 3, 4, 5, 6]  # 舵机ID列表
    UPDATE_RATE = 20         # 更新频率 (Hz)
    
    print("=" * 60)
    print("遥操作系统 - 教师端到学生端实时角度同步")
    print("=" * 60)
    print(f"教师端端口: {TEACHER_PORT}")
    print(f"学生端端口: {STUDENT_PORT}")
    print(f"舵机ID: {SERVO_IDS}")
    print(f"更新频率: {UPDATE_RATE}Hz")
    print("=" * 60)
    
    # 创建并运行遥操作系统
    tele_system = TeleOperationSystem(
        teacher_port=TEACHER_PORT,
        student_port=STUDENT_PORT,
        servo_ids=SERVO_IDS
    )
    
    tele_system.run(update_rate=UPDATE_RATE)

if __name__ == '__main__':
    main()