from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QComboBox, QPushButton, QTextEdit,
                            QLineEdit, QGroupBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal, Qt
from SDK.port_handler import PortHandler
from SDK.scan_connector import ScanConnector
import serial.tools.list_ports
import sys


class ScanGUI(QMainWindow):
    logSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("扫描工具")
        self.setGeometry(100, 100, 1000, 600)
        self.port_handler = PortHandler()
        
        self.port_handler.add_write_callback(self.__write_callback)
        self.port_handler.add_read_callback(self.__read_callback)
        
        self.scan_connector = ScanConnector(self.port_handler)
        self.scan_connector.add_result_callback(self.__result_callback)

        self.__is_running = False
        
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
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "1000000"])
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
        
        clear_btn = QPushButton("清理日志")
        clear_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_btn)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        # 右侧扫描配置区域
        scan_group = QGroupBox("扫描配置")
        scan_group.setMinimumWidth(360)
        scan_group.setMaximumWidth(360)
        scan_layout = QVBoxLayout()
        
        # 扫描按钮区域
        scan_btn_group = QGroupBox()
        scan_btn_layout = QVBoxLayout()
        
        self.scan_btn = QPushButton("开始扫描")
        self.scan_btn.clicked.connect(self.start_scan)
        
        scan_btn_layout.addWidget(self.scan_btn)
        scan_btn_group.setLayout(scan_btn_layout)
        
        # ID配置区域
        config_group = QGroupBox("ID配置")
        config_layout = QVBoxLayout()
        
        # MAC地址输入
        mac_layout = QHBoxLayout()
        mac_layout.addWidget(QLabel("MAC地址:"))
        self.mac_input = QLineEdit()
        mac_layout.addWidget(self.mac_input)
        
        # ID输入
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("设备ID:"))
        self.device_id = QLineEdit()
        id_layout.addWidget(self.device_id)
        
        # 配置按钮
        self.config_btn = QPushButton("配置")
        self.config_btn.clicked.connect(self.config_device)
        
        config_layout.addLayout(mac_layout)
        config_layout.addLayout(id_layout)
        config_layout.addWidget(self.config_btn)
        config_group.setLayout(config_layout)
        
        # 扫描结果显示
        self.scan_result = QTextEdit()
        self.scan_result.setReadOnly(True)
        
        # LED配置区域
        led_group = QGroupBox("LED配置")
        led_layout = QVBoxLayout()
        
        # MAC地址输入
        led_mac_layout = QHBoxLayout()
        led_mac_layout.addWidget(QLabel("MAC地址:"))
        self.led_mac_input = QLineEdit()
        led_mac_layout.addWidget(self.led_mac_input)
        
        # LED状态输入
        led_state_layout = QHBoxLayout()
        led_state_layout.addWidget(QLabel("LED状态:"))
        self.led_state_input = QLineEdit()
        led_state_layout.addWidget(self.led_state_input)
        
        # 配置按钮
        self.led_config_btn = QPushButton("设置LED")
        self.led_config_btn.clicked.connect(self.config_led)
        
        led_layout.addLayout(led_mac_layout)
        led_layout.addLayout(led_state_layout)
        led_layout.addWidget(self.led_config_btn)
        led_group.setLayout(led_layout)
        
        scan_layout.addWidget(scan_btn_group)
        scan_layout.addWidget(config_group)
        scan_layout.addWidget(led_group)
        scan_layout.addWidget(self.scan_result)
        scan_group.setLayout(scan_layout)

        main_layout.addWidget(port_group)
        main_layout.addWidget(log_group)
        main_layout.addWidget(scan_group)
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
    
    def start_scan(self):
        if not self.port_handler.is_open():
            self.__update_log("错误: 请先连接串口")
            return
            
        self.scan_connector.start_scan()
    
    def __write_callback(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = "<font color='blue'>发送: {}</font>".format(data)
        self.__update_log(data)

    def __read_callback(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = "<font color='red'>接收: {}</font>".format(data)
        self.__update_log(data)
        
    def __result_callback(self, result):
        result = ''.join(["<font color='green'>({} 0x{:08X}) {}</font><br>".format(mac, mac, id) for mac, id in result])
        self.__update_log(result)
        
    def config_device(self):
        if not self.port_handler.is_open():
            self.__update_log("错误: 请先连接串口")
            return
            
        mac = self.mac_input.text().strip()
        device_id = self.device_id.text().strip()
        
        if not mac or not device_id:
            self.__update_log("错误: MAC地址和设备ID不能为空")
            return
            
        try:
            mac_int = int(mac)
            id_int = int(device_id)
            
            # 这里添加实际的配置逻辑，通过port_handler发送配置指令
            # 示例: self.port_handler.write(bytes([0x01, 0x02, ...]))
            self.scan_connector.config_id(mac_int, id_int)
            self.__update_log(f"配置设备: MAC={mac}, ID={device_id}")
        except ValueError:
            self.__update_log("错误: MAC地址和设备ID必须是有效的十六进制数字")
            
    def config_led(self):
        if not self.port_handler.is_open():
            self.__update_log("错误: 请先连接串口")
            return
            
        mac = self.led_mac_input.text().strip()
        led_state = self.led_state_input.text().strip()
        
        if not mac or not led_state:
            self.__update_log("错误: MAC地址和LED状态不能为空")
            return
            
        try:
            mac_int = int(mac)
            led_state_int = int(led_state)
            
            # 这里添加实际的LED配置逻辑
            # 示例: self.port_handler.write(bytes([0x03, 0x04, ...]))
            self.scan_connector.config_led(mac_int, led_state_int)
            self.__update_log(f"配置LED: MAC={mac}, 状态={led_state}")
        except ValueError:
            self.__update_log("错误: MAC地址和LED状态必须是有效的数字")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScanGUI()
    window.show()
    sys.exit(app.exec_())