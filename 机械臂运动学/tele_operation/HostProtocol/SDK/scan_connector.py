
from math import e
import stat
import time
from unittest import result
from .utils import bytes_to_int, int_to_bytes
from .port_handler import PortHandler
from threading import Thread
from typing import List


FRAME_HEADER    = 0xAA
FRAME_TAIL      = 0xBB

FRAME_ID_TX     = 0x7E
FRAME_ID_RX     = 0xFE

FRAME_CMD_SCAN          = 0x00
FRAME_CMD_CONFIG_ID     = 0x01
FRAME_CMD_CONFIG_LED    = 0x02


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


class ScanConnector:
    
    def __init__(self, port_handler: PortHandler):
        self.__port_handler = port_handler
        
        self.__is_running = False
        self.__result_callback = None
        
    def add_result_callback(self, callback):
        if self.__result_callback is None:
            self.__result_callback = []
        
        if callback not in self.__result_callback:
            self.__result_callback.append(callback)

    def remove_result_callback(self, callback):
        if self.__result_callback is None:
            return

        if callback in self.__result_callback:
            self.__result_callback.remove(callback)
    
    def __notify_result_callback(self, result):
        if self.__result_callback is None:
            return

        for callback in self.__result_callback:
            callback(result)    
    
    def start_scan(self):
        if self.__is_running:
            return
        self.__is_running = True
        
        self.__scan_thread = Thread(target=self.__scan_task)
        self.__scan_thread.start()
        
    def config_id(self, mac: int, id: int):
        if self.__is_running:
            return
        self.__is_running = True
        
        self.__config_id_thread = Thread(target=self.__config_id_task, args=(mac, id))
        self.__config_id_thread.start()      
    
    def config_led(self, mac: int, state: int):
        if self.__is_running:
            return
        self.__is_running = True
        
        self.__config_led_thread = Thread(target=self.__config_led_task, args=(mac, state))
        self.__config_led_thread.start()        
    
    def __scan_mac(self, h_mac, m_mac = None, l_mac = None):
        if h_mac is None:
            return (2, None)
        
        if m_mac is None and l_mac is not None:
            return (2, None)
        
        mac_type = 0
        if m_mac is None and l_mac is None:
            mac_type = 0
        elif m_mac is not None and l_mac is None:
            mac_type = 1
        elif m_mac is not None and l_mac is not None:
            mac_type = 2
        
        data = []
        data.append(mac_type) # type
        data.append(h_mac) # h_mac
        data.append(m_mac if m_mac is not None else 0x00) # m_mac
        data.append(m_mac if l_mac is not None else 0x00) # l_mac
        
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_SCAN, data)
        
        self.__port_handler.write_port(frame)
        
        read_list = []
        self.__port_handler.read_timeout = 1
        
        # 读取响应
        retry_cnt = 0
        is_success = False
        has_data = False
        while True:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                if retry_cnt < 5:
                    retry_cnt += 1
                    time.sleep(0.01)
                    continue
                else:
                    break
            has_data = True
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
                read_list = read_list[0:7 + data_length]
                is_success = True
                break
                
            if is_success:
                break
             
        if is_success:
            # 解析数据
            return (0, bytes_to_int(bytearray(read_list[5:9])), read_list[9])
        else:
            if has_data:
                # 有数据，但是校验失败
                return (1, None, None)
            else:
                # 没有数据，可能是设备没有响应
                return (2, None, None)


    def __scan_task(self):
        if not self.__port_handler.is_open():
            return
        if not self.__is_running:
            return
        
        success_mac_list = []
        
        ##### 1. 逐个mac地址进行扫描
        failed_h_mac_list = []
        for h_mac in range(0, 256):
            result = self.__scan_mac(h_mac)
            if result[0] == 0:
                # 成功
                success_mac_list.append((result[1], result[2]))
            elif result[0] == 1:
                # 校验失败
                failed_h_mac_list.append(h_mac)
        
        if len(failed_h_mac_list) > 0:
            # 2. 逐个h_mac, m_mac进行扫描
            failed_h_m_mac_list = []
            for h_mac in failed_h_mac_list:
                for m_mac in range(0, 256):
                    result = self.__scan_mac(h_mac, m_mac)
                    if result[0] == 0:
                        # 成功
                        success_mac_list.append((result[1], result[2]))
                    elif result[0] == 1:
                        # 校验失败
                        failed_h_m_mac_list.append((h_mac, m_mac))

            if len(failed_h_m_mac_list) > 0:
                # 3. 逐个h_mac, m_mac, l_mac进行扫描
                failed_h_m_l_mac_list = []
                for h_mac, m_mac in failed_h_m_mac_list:
                    for l_mac in range(0, 256):
                        result = self.__scan_mac(h_mac, m_mac, l_mac)
                        if result[0] == 0:
                            # 成功
                            success_mac_list.append((result[1], result[2]))
                        elif result[0] == 1:
                            # 校验失败
                            failed_h_m_l_mac_list.append((h_mac, m_mac, l_mac))
                
                if len(failed_h_m_l_mac_list) > 0:
                    # 存在相同的mac，不可能发生
                    print("存在相同的mac，不可能发生")
                    
        print("success_mac_list: ", success_mac_list) 
        self.__notify_result_callback(success_mac_list)

        self.__is_running = False
        self.__scan_thread = None
        
    def __config_id_task(self, mac: int, id: int):
        if not self.__port_handler.is_open():
            return
        if not self.__is_running:
            return

        data = []
        data.extend(list(int_to_bytes(mac))) # mac
        data.append(id) # id
        
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_CONFIG_ID, data)

        self.__port_handler.write_port(frame)
        
        read_list = []
        self.__port_handler.read_timeout = 1
        
        # 读取响应
        retry_cnt = 0
        is_success = False
        has_data = False
        while True:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                if retry_cnt < 5:
                    retry_cnt += 1
                    time.sleep(0.01)
                    continue
                else:
                    break
            has_data = True
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
                read_list = read_list[0:7 + data_length]
                is_success = True
                break
                
            if is_success:
                break
             
        if is_success:
            # 解析数据
            if read_list[9] == 0x00:
                print("配置成功")
            else:
                print("配置失败")
        else:
            print("配置失败")
        self.__is_running = False
        self.__config_thread = None

    def __config_led_task(self, mac: int, state: int):
        if not self.__port_handler.is_open():
            return
        if not self.__is_running:
            return

        data = []
        data.extend(list(int_to_bytes(mac))) # mac
        data.append(state) # state
        
        frame = frame_generator(FRAME_ID_TX, FRAME_CMD_CONFIG_LED, data)

        self.__port_handler.write_port(frame)
        
        read_list = []
        self.__port_handler.read_timeout = 1
        
        # 读取响应
        retry_cnt = 0
        is_success = False
        has_data = False
        while True:
            in_waiting = self.__port_handler.in_waiting()
            if in_waiting == 0:
                if retry_cnt < 5:
                    retry_cnt += 1
                    time.sleep(0.01)
                    continue
                else:
                    break
            has_data = True
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
                read_list = read_list[0:7 + data_length]
                is_success = True
                break
                
            if is_success:
                break
             
        if is_success:
            # 解析数据
            if read_list[9] == 0x00:
                print("配置成功")
            else:
                print("配置失败")
        else:
            print("配置失败")
        self.__is_running = False
        self.__config_thread = None