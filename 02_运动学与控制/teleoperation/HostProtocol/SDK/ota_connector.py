from .utils import bytes_to_int, int_to_bytes
from .port_handler import PortHandler

import time
import crcmod
from threading import Thread
from typing import List

import os


FRAME_HEADER    = 0xAA
FRAME_TAIL      = 0xBB

FRAME_ID_TX     = 0x7D
FRAME_ID_RX     = 0xFD

FRAME_CMD_TAG       = 0x10
FRAME_CMD_INFO      = 0x11
FRAME_CMD_DATA      = 0x12
FRAME_CMD_FINISH    = 0x13

FRAME_CMD_OTA_CK    = 0x20
FRAME_CMD_OTA_BK    = 0x21


def checksum(id: int, cmd: int, data: List[int]) -> int:
    """
    计算帧校验和
    
    Args:
        id: 设备ID
        cmd: 命令
        data: 数据列表
        
    Returns:
        int: 计算后的校验和(0-255)
    """
    return (id + cmd + len(data) + sum(data)) & 0xFF


def frame_generator(id: int, cmd: int, data: List[int]) -> bytearray:
    """
    帧生成器
    Args:
        id (int): 设备ID
        cmd (int): 命令
        data (List[int]): 数据
    """
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


class OTAConnector:
    def __init__(self, portHandler: PortHandler):
        self.__port_handler = portHandler
        self.__err_callback = None
        self.__start_callback = None
        self.__progress_callback = None
        self.__finish_callback = None
        
        self.__is_running = False
        self.__mac = 0
        self.__mojor = 0
        self.__minor = 0
        self.__crc = 0
        self.__file_path = ''
        self.__file_size = 0
        self.__offset = 0

        self.__recv_thread = None
        
    def __crc32(self, file_path: str):
        crc_func = crcmod.mkCrcFun(0x104C11DB7, initCrc=0xFFFFFFFF, xorOut=0xFFFFFFFF, rev=False)
        with open(file_path, 'rb') as f:
            data = f.read()
            crc_value = crc_func(data)
        return crc_value
    
    def add_err_callback(self, callback):
        if self.__err_callback is None:
            self.__err_callback = []
        if callback not in self.__err_callback:
            self.__err_callback.append(callback)

    def remove_err_callback(self, callback):
        if self.__err_callback is not None:
            if callback in self.__err_callback:
                self.__err_callback.remove(callback)

    def __notify_err(self, msg: str):
        if self.__err_callback is not None:
            for callback in self.__err_callback:
                callback(msg)

    def add_start_callback(self, callback):
        if self.__start_callback is None:
            self.__start_callback = []
        if callback not in self.__start_callback:
            self.__start_callback.append(callback)

    def remove_start_callback(self, callback):
        if self.__start_callback is not None:
            if callback in self.__start_callback:
                self.__start_callback.remove(callback)

    def __notify_start(self):
        if self.__start_callback is not None:
            for callback in self.__start_callback:
                callback()

    def add_progress_callback(self, callback):
        if self.__progress_callback is None:
            self.__progress_callback = []
        if callback not in self.__progress_callback:
            self.__progress_callback.append(callback)
            
    def remove_progress_callback(self, callback):
        if self.__progress_callback is not None:
            if callback in self.__progress_callback:
                self.__progress_callback.remove(callback)

    def __notify_progress(self, progress: int):
        if self.__progress_callback is not None:
            for callback in self.__progress_callback:
                callback(progress)

    def add_finish_callback(self, callback):
        if self.__finish_callback is None:
            self.__finish_callback = []
        if callback not in self.__finish_callback:
            self.__finish_callback.append(callback)

    def remove_finish_callback(self, callback):
        if self.__finish_callback is not None:
            if callback in self.__finish_callback:
                self.__finish_callback.remove(callback)            
    
    def __notify_finish(self):
        if self.__finish_callback is not None:
            for callback in self.__finish_callback:
                callback()
    
    def start_ota(self, mac: int, mojor: int, minor: int, file_path: str):
        if self.__is_running:
            self.__notify_err('OTA is running')
            return
        if not os.path.exists(file_path):
            self.__notify_err('File not found')
            return
        if os.path.getsize(file_path) == 0:
            self.__notify_err('File size is 0')
            return
        if os.path.getsize(file_path) >= 64 * 1024: 
            self.__notify_err('File size is too large')
            return
        self.__mac = mac
        self.__mojor = mojor
        self.__minor = minor
        self.__file_path = file_path
        self.__crc = self.__crc32(file_path)
        self.__file_size = os.path.getsize(file_path)
        
        self.__is_running = True
        self.__notify_start()
        
        self.__recv_thread = Thread(target=self.__recv_task)
        self.__recv_thread.start()
        
        self.__send_tag()
    
    def __send_tag(self):
        data = []
        data.extend(int_to_bytes(self.__mac))
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_TAG, data)
        self.__port_handler.write_port(frame)
        
    def __send_info(self):
        data = []
        data.extend(int_to_bytes(self.__mac))
        data.extend(int_to_bytes(self.__crc))
        data.extend(int_to_bytes(self.__file_size))
        data.append(self.__mojor)
        data.append(self.__minor)
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_INFO, data)
        self.__port_handler.write_port(frame)

    def __send_data(self):
        with open(self.__file_path, 'rb') as f:
            f.seek(self.__offset) 
            content = f.read(16)
            content = list(content)
        
        data = []
        data.extend(int_to_bytes(self.__mac))
        data.extend(int_to_bytes(self.__offset))
        data.append(len(content))
        data.extend(content)
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_DATA, data)
        self.__port_handler.write_port(frame)
        
        self.__offset += len(content)
        
        progress = self.__offset / self.__file_size * 100
        self.__notify_progress(progress)
        
    def __send_finish(self):
        data = []
        data.extend(int_to_bytes(self.__mac))
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_FINISH, data)
        # 缺少发送操作
        self.__port_handler.write_port(frame)  # 需要添加这行

    def __do_logic(self, buff: List[int]):
        id = buff[2]
        if id != FRAME_ID_RX:
            return
        
        cmd = buff[3]
        mac = bytes_to_int(bytearray(buff[5:9]))
        if mac != self.__mac:
            return
        
        state = buff[9]

        if state == 0:
            # success
            if cmd == FRAME_CMD_TAG:
                # 发送摘要
                self.__send_info()
            elif cmd == FRAME_CMD_INFO:
                # 发送数据
                self.__send_data()
            elif cmd == FRAME_CMD_DATA:
                # 继续发送数据或发送完成
                if self.__offset >= self.__file_size:
                    # 发送完成
                    self.__send_finish()
                else:
                    # 继续发送数据
                    self.__send_data()
            elif cmd == FRAME_CMD_FINISH:
                # 结束
                self.__notify_finish()
            elif cmd == FRAME_CMD_OTA_CK:
                # 校验成功
                self.__notify_err('Checksum success')
            elif cmd == FRAME_CMD_OTA_BK:
                # 备份成功
                self.__notify_err('Backup success')
                self.__is_running = False
        else:
            # failed
            self.__is_running = False
            if cmd == FRAME_CMD_TAG:
                # 发送摘要失败
                self.__notify_err('Send tag failed')
            elif cmd == FRAME_CMD_INFO:
                # 发送数据失败
                self.__notify_err('Send info failed')
            elif cmd == FRAME_CMD_DATA:
                # 继续发送数据或发送完成失败
                self.__notify_err('Send data failed')
            elif cmd == FRAME_CMD_FINISH:
                # 结束失败
                self.__notify_err('Send finish failed')
            elif cmd == FRAME_CMD_OTA_CK:
                # 校验失败
                self.__notify_err('Checksum failed')
            elif cmd == FRAME_CMD_OTA_BK:
                # 备份失败
                self.__notify_err('Backup failed')
    
    def __recv_task(self):
        result_list = []
        read_list = []
        state = 0
        self.__port_handler.read_timeout = 1
        while self.__port_handler.is_open() and self.__is_running:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                time.sleep(0.01)
                continue
                
            read_list.extend(list(self.__port_handler.read_port(in_waiting)))
                
            while len(read_list) >= 7:
                if read_list[0] != FRAME_HEADER or read_list[1] != FRAME_HEADER:
                    read_list.pop(0)
                    continue
                data_length = read_list[4]
                if data_length > 48:
                    read_list.pop(0)
                    continue
                if len(read_list) < 7 + data_length:
                    break
                if read_list[6 + data_length]!= FRAME_TAIL:
                    read_list.pop(0)
                    continue
                checksum = 0
                for i in range(2, 5 + data_length):
                    checksum += read_list[i]
                checksum = checksum & 0xFF
                if checksum != read_list[5 + data_length]:
                    read_list.pop(0)
                    continue
                # 将数据添加到结果列表，移除已解析的数据
                result_list.extend(read_list[0:7 + data_length])
                read_list = read_list[7 + data_length:]
                
                # 处理逻辑
                self.__do_logic(result_list)
                result_list = []

        self.__is_running = False
            