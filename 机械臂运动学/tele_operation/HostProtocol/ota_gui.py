from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QComboBox, QPushButton, QTextEdit,
                            QLineEdit, QGroupBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal, Qt
from SDK.port_handler import PortHandler
from SDK.sync_connector import SyncConnector
from SDK.ota_connector import OTAConnector
from SDK.global_state import Address, Result

import serial.tools.list_ports

import sys


class OTAGUI(QMainWindow):
    logSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OTA升级工具")
        self.setGeometry(100, 100, 1000, 600)
        self.port_handler = PortHandler()
        
        self.port_handler.add_write_callback(self.__write_callback)
        self.port_handler.add_read_callback(self.__read_callback)
        
        self.sync_connector = SyncConnector(self.port_handler)
        
        self.ota_connector = OTAConnector(self.port_handler)
        self.ota_connector.add_err_callback(self.__update_log)
        self.ota_connector.add_start_callback(lambda: self.__update_log("OTA开始"))
        self.ota_connector.add_progress_callback(lambda p: self.__update_log(f"进度: {p}%"))
        self.ota_connector.add_finish_callback(lambda: self.__update_log("OTA完成"))

        self.init_ui()
        self.scan_ports()
        
        self.logSignal.connect(self.__update_log_text)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧串口配置区域
        port_group = QGroupBox("串口配置")
        port_group.setMinimumWidth(320)
        port_group.setMaximumWidth(320)
        port_layout = QVBoxLayout()

        self.port_combo = QComboBox()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.scan_ports)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400"])
        self.baudrate_combo.setCurrentIndex(5)
        
        port_control_layout = QHBoxLayout()
        port_control_layout.addWidget(QLabel("端口:"))
        port_control_layout.addWidget(self.port_combo, stretch=1)
        port_control_layout.addWidget(self.refresh_btn)
        
        baudrate_layout = QHBoxLayout()
        baudrate_layout.addWidget(QLabel("波特率:"))
        baudrate_layout.addWidget(self.baudrate_combo, stretch=1)
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)

        port_layout.addLayout(port_control_layout)
        port_layout.addLayout(baudrate_layout)
        port_layout.addWidget(self.connect_btn)
        port_layout.addStretch()
        port_group.setLayout(port_layout)

        # 中间日志区域
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # 添加清理按钮
        clear_btn = QPushButton("清理日志")
        clear_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_btn)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # 右侧区域
        right_group = QGroupBox()
        right_group.setMinimumWidth(360)
        right_group.setMaximumWidth(360)
        right_layout = QVBoxLayout()
        
        # 设备查询区域
        device_group = QGroupBox("设备查询")
        device_layout = QVBoxLayout()
        
        self.query_btn = QPushButton("查询设备")
        self.query_btn.clicked.connect(self.query_device)
        
        self.device_info = QTextEdit()
        self.device_info.setReadOnly(True)
        
        device_layout.addWidget(self.query_btn)
        device_layout.addWidget(self.device_info)
        device_group.setLayout(device_layout)
        
        # OTA配置区域
        ota_group = QGroupBox("OTA配置")
        ota_layout = QVBoxLayout()
        
        # MAC地址输入
        mac_layout = QHBoxLayout()
        mac_layout.addWidget(QLabel("MAC地址:"))
        self.mac_input = QLineEdit()
        mac_layout.addWidget(self.mac_input)
        
        # 版本号输入
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("主版本:"))
        self.major_input = QLineEdit()
        version_layout.addWidget(self.major_input)
        
        version_layout.addWidget(QLabel("次版本:"))
        self.minor_input = QLineEdit()
        version_layout.addWidget(self.minor_input)
        
        # 固件选择
        firmware_layout = QHBoxLayout()
        firmware_layout.addWidget(QLabel("固件文件:"))
        self.firmware_path = QLineEdit()
        self.firmware_path.setReadOnly(True)
        firmware_layout.addWidget(self.firmware_path)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_firmware)
        firmware_layout.addWidget(self.browse_btn)
        
        # 开始OTA按钮
        self.start_ota_btn = QPushButton("开始OTA")
        self.start_ota_btn.clicked.connect(self.start_ota)
        
        ota_layout.addLayout(mac_layout)
        ota_layout.addLayout(version_layout)
        ota_layout.addLayout(firmware_layout)
        ota_layout.addWidget(self.start_ota_btn)
        ota_layout.addStretch()
        ota_group.setLayout(ota_layout)
        
        right_layout.addWidget(device_group)
        right_layout.addWidget(ota_group)
        right_group.setLayout(right_layout)

        main_layout.addWidget(port_group)
        main_layout.addWidget(log_group)
        main_layout.addWidget(right_group)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def __update_log(self, msg):
        self.logSignal.emit(msg)
    
    def __update_log_text(self, msg):
        self.log_text.append(msg)
        
    def clear_log(self):
        self.log_text.clear()
    
    def scan_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
    
    def toggle_connection(self):
        if self.port_handler.is_open():
            self.port_handler.close()
            self.connect_btn.setText("连接")
            self.__update_log("串口已关闭")
        else:
            port = self.port_combo.currentText()
            baudrate = int(self.baudrate_combo.currentText())
            self.port_handler.baudrate = baudrate
            if self.port_handler.open(port):
                self.connect_btn.setText("断开")
                self.__update_log(f"串口已连接: {port} @ {baudrate}")
            else:
                self.__update_log("串口连接失败")
    
    def browse_firmware(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择固件文件", "", "固件文件 (*.bin *.hex);;所有文件 (*)")
        if file_path:
            self.firmware_path.setText(file_path)
    
    def query_device(self):
        if not self.port_handler.is_open():
            self.__update_log("错误: 请先连接串口")
            return
            
        try:
            self.__update_log("正在查询设备...")
            # 这里添加实际的设备查询逻辑
            
            for i in range(6):
                result: Result = self.sync_connector.read(i, Address.DEVICE_UUID)
                if result.is_success():
                    self.__update_log("设备{}: {}".format(i, result.get_data(Address.DEVICE_UUID)))
                    self.device_info.setText("{}".format(result.get_data(Address.DEVICE_UUID)))
                else:
                    self.__update_log(f"设备{i}: 未找到")
            
            self.__update_log("设备查询完成")
        except Exception as e:
            self.__update_log(f"设备查询错误: {str(e)}")
    
    def start_ota(self):
        if not self.port_handler.is_open():
            self.__update_log("错误: 请先连接串口")
            return
            
        try:
            mac = int(self.mac_input.text())
            major = int(self.major_input.text())
            minor = int(self.minor_input.text())
            file_path = self.firmware_path.text()
            
            if not file_path:
                self.__update_log("错误: 请选择固件文件")
                return
                
            self.ota_connector.start_ota(mac, major, minor, file_path)
        except ValueError as e:
            self.__update_log(f"错误: 输入格式不正确 - {str(e)}")
    
    def __write_callback(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = "<font color='blue'>发送: {}</font>".format(data)
        self.__update_log(data)

    def __read_callback(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = "<font color='red'>接收: {}</font>".format(data)
        self.__update_log(data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OTAGUI()
    window.show()
    sys.exit(app.exec_())
    
    