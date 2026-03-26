from calendar import c
import sys
from this import d
import serial.tools.list_ports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QComboBox, QPushButton, QTextEdit,
                            QLineEdit, QGroupBox, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from SDK.sync_connector import SyncConnector
from SDK.port_handler import PortHandler
from SDK.global_state import Address


class SerialThread(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, connector):
        super().__init__()
        self.connector = connector
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            result = self.connector._parse_response_frame()
            if result.is_success() and result.frame:
                self.data_received.emit(f"Received: {result.frame}")

    def stop(self):
        self.running = False


class SerialGUI(QMainWindow):
    logSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("串口通信工具")
        self.setGeometry(100, 100, 800, 600)
        self.port_handler = PortHandler()
        self.port_handler.add_write_callback(self.__update_write_text)
        self.port_handler.add_read_callback(self.__update_read_text)
        self.connector = None
        self.serial_thread = None

        self.init_ui()
        self.scan_ports()
        
        self.logSignal.connect(self.__update_log_text)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧串口配置区域
        config_group = QGroupBox("串口配置")
        config_group.setMaximumWidth(320)
        config_layout = QVBoxLayout()

        self.port_combo = QComboBox()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.scan_ports)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400"])
        self.baudrate_combo.setCurrentIndex(5)
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口:"))
        port_layout.addWidget(self.port_combo, stretch=1)
        port_layout.addWidget(self.refresh_btn)
        
        baudrate_layout = QHBoxLayout()
        baudrate_layout.addWidget(QLabel("波特率:"))
        baudrate_layout.addWidget(self.baudrate_combo, stretch=1)
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)

        config_layout.addLayout(port_layout, stretch=1)
        config_layout.addLayout(baudrate_layout, stretch=1)
        config_layout.addWidget(self.connect_btn, stretch=1)
        config_layout.addWidget(QLabel(""), stretch=20)
        config_group.setLayout(config_layout)

        # 右侧数据收发区域
        data_group = QGroupBox("数据通信")
        data_layout = QVBoxLayout()

        self.receive_text = QTextEdit()
        self.receive_text.setReadOnly(True)

        self.send_text = QLineEdit()
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_data)

        send_layout = QHBoxLayout()
        send_layout.addWidget(self.send_text)
        send_layout.addWidget(self.send_btn)

        data_layout.addWidget(QLabel("接收数据:"))
        data_layout.addWidget(self.receive_text)
        data_layout.addWidget(QLabel("发送数据:"))
        data_layout.addLayout(send_layout)
        data_group.setLayout(data_layout)

        # 测试功能区
        test_group = QGroupBox("测试功能")
        test_layout = QVBoxLayout()
        
        # 临时写测试配置面板
        temp_write_config = QGroupBox("临时写配置")
        temp_write_config_layout = QVBoxLayout()
        
        # ID输入
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("设备ID:"))
        self.temp_write_id = QLineEdit("1")
        id_layout.addWidget(self.temp_write_id)
        temp_write_config_layout.addLayout(id_layout)
        
        # 寄存器组输入
        for i in range(1, 5):
            group_layout = QHBoxLayout()
            
            # 添加复选框
            setattr(self, f"temp_write_check_{i}", QCheckBox(f"寄存器{i}地址:"))
            group_layout.addWidget(getattr(self, f"temp_write_check_{i}"))
            
            addr_layout = QHBoxLayout()
            setattr(self, f"temp_write_addr_{i}", QLineEdit())
            addr_layout.addWidget(getattr(self, f"temp_write_addr_{i}"))
            
            value_layout = QHBoxLayout()
            value_layout.addWidget(QLabel(f"值{i}:"))
            setattr(self, f"temp_write_value_{i}", QLineEdit("1"))
            value_layout.addWidget(getattr(self, f"temp_write_value_{i}"))
            
            group_layout.addLayout(addr_layout)
            group_layout.addLayout(value_layout)
            temp_write_config_layout.addLayout(group_layout)
        
        self.temp_write_btn = QPushButton("临时写测试")
        self.temp_write_btn.clicked.connect(self.temp_write_test)
        temp_write_config_layout.addWidget(self.temp_write_btn)
    
        temp_write_config.setLayout(temp_write_config_layout)
        # 持久写测试配置面板
        persist_write_config = QGroupBox("持久写配置")
        persist_write_config_layout = QVBoxLayout()
        
        # ID输入
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("设备ID:"))
        self.persist_write_id = QLineEdit("1")
        id_layout.addWidget(self.persist_write_id)
        persist_write_config_layout.addLayout(id_layout)
        
        # 寄存器组输入
        for i in range(1, 5):
            group_layout = QHBoxLayout()
            
            # 添加复选框
            setattr(self, f"persist_write_check_{i}", QCheckBox(f"寄存器{i}地址:"))
            group_layout.addWidget(getattr(self, f"persist_write_check_{i}"))
            
            addr_layout = QHBoxLayout()
            setattr(self, f"persist_write_addr_{i}", QLineEdit(f""))
            addr_layout.addWidget(getattr(self, f"persist_write_addr_{i}"))
            
            value_layout = QHBoxLayout()
            value_layout.addWidget(QLabel(f"值{i}:"))
            setattr(self, f"persist_write_value_{i}", QLineEdit("1"))
            value_layout.addWidget(getattr(self, f"persist_write_value_{i}"))
            
            group_layout.addLayout(addr_layout)
            group_layout.addLayout(value_layout)
            persist_write_config_layout.addLayout(group_layout)
        
        self.persist_write_btn = QPushButton("持久写测试")
        self.persist_write_btn.clicked.connect(self.persist_write_test)
        persist_write_config_layout.addWidget(self.persist_write_btn)
    
        persist_write_config.setLayout(persist_write_config_layout)
        
        # 读测试配置面板
        read_config = QGroupBox("读配置")
        read_config_layout = QVBoxLayout()
        
        # ID输入
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("设备ID(逗号分隔):"))
        self.read_ids = QLineEdit("1")
        id_layout.addWidget(self.read_ids)
        read_config_layout.addLayout(id_layout)
        
        # 寄存器组输入
        for i in range(1, 5):
            group_layout = QHBoxLayout()
            
            # 添加复选框
            setattr(self, f"read_check_{i}", QCheckBox(f"寄存器{i}地址:"))
            group_layout.addWidget(getattr(self, f"read_check_{i}"))
            
            addr_layout = QHBoxLayout()
            setattr(self, f"read_addr_{i}", QLineEdit(f"0x0{i}"))
            addr_layout.addWidget(getattr(self, f"read_addr_{i}"))
            
            group_layout.addLayout(addr_layout)
            read_config_layout.addLayout(group_layout)
        
        self.read_btn = QPushButton("读测试")
        self.read_btn.clicked.connect(self.read_test)
        read_config_layout.addWidget(self.read_btn)
        read_config.setLayout(read_config_layout)
        
        test_layout.addWidget(temp_write_config, stretch=1)
        test_layout.addWidget(persist_write_config, stretch=1)
        test_layout.addWidget(read_config, stretch=1)
        test_group.setLayout(test_layout)

        main_layout.addWidget(config_group)
        main_layout.addWidget(data_group)
        main_layout.addWidget(test_group)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def toggle_connection(self):
        if not self.port_handler.is_open():
            port = self.port_combo.currentText()
            baudrate = int(self.baudrate_combo.currentText())
            try:
                self.port_handler.port = port
                self.port_handler.baudrate = baudrate
                self.port_handler.open(port) 
                self.receive_text.append(f"已连接到 {port} {baudrate}")

                self.connect_btn.setText("断开连接")
            except Exception as e:
                self.receive_text.append(f"连接失败: {str(e)}")
        else:

            self.port_handler.close()
            self.receive_text.append("已断开连接")
            
            self.connect_btn.setText("连接")

    def send_data(self):
        if self.connector:
            data = self.send_text.text()
            if data:
                try:
                    # 这里需要根据实际协议实现数据发送
                    # 示例: 发送到ID为1的设备，地址0x01，值为data
                    result = self.connector.write(1, Address(0x01, 1), int(data))
                    if result.is_success():
                        self.receive_text.append(f"已发送: {data}")
                    else:
                        self.receive_text.append(f"发送失败: {result.error}")
                except Exception as e:
                    self.receive_text.append(f"发送错误: {str(e)}")
            else:
                self.receive_text.append("请输入要发送的数据")
        else:
            self.receive_text.append("请先连接串口")

    def scan_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
        if ports:
            self.receive_text.append(f"找到 {len(ports)} 个可用串口")
        else:
            self.receive_text.append("未找到可用串口")

    def update_receive_text(self, data):
        self.receive_text.append(data)
        
    def temp_write_test(self):
        if self.port_handler and self.port_handler.is_open():
            try:
                device_id = int(self.temp_write_id.text())
            except Exception as e:
                self.receive_text.append(f"无效的设备ID: {str(e)}")
                return
            
            addrs = []
            values = []
            for i in range(1, 5):
                if getattr(self, f"temp_write_check_{i}").isChecked():
                    addr = getattr(self, f"temp_write_addr_{i}").text()
                    value = getattr(self, f"temp_write_value_{i}").text()
                    if addr and value:
                        try: 
                            addr_int = int(addr)
                            value_int = int(value)
                            
                            addrs.append(Address.get_address(addr_int))
                            values.append(value_int)
                        except Exception as e:
                            self.receive_text.append(f"无效的地址或值: {str(e)}")
                            return

            connector = SyncConnector(self.port_handler)
            result = connector.write(device_id, addrs, values)
            if result.is_success():
                self.receive_text.append("临时写测试成功")
            else:
                self.receive_text.append(f"临时写测试失败: {result.get_error_code()}")
        else:
            self.receive_text.append("请先连接串口")
            
    def persist_write_test(self):
        if self.port_handler and self.port_handler.is_open():
            try:
                device_id = int(self.persist_write_id.text())
            except Exception as e:
                self.receive_text.append(f"无效的设备ID: {str(e)}")
                return
            
            addrs = []
            values = []
            for i in range(1, 5):
                if getattr(self, f"persist_write_check_{i}").isChecked():
                    addr = getattr(self, f"persist_write_addr_{i}").text()
                    value = getattr(self, f"persist_write_value_{i}").text()
                    if addr and value:
                        try: 
                            addr_int = int(addr)
                            value_int = int(value)
                            
                            addrs.append(Address.get_address(addr_int))
                            values.append(value_int)
                        except Exception as e:
                            self.receive_text.append(f"无效的地址或值: {str(e)}")
                            return

            connector = SyncConnector(self.port_handler)
            result = connector.store(device_id, addrs, values)
            if result.is_success():
                self.receive_text.append("持久写测试成功")
            else:
                self.receive_text.append(f"持久写测试失败: {result.get_error_code()}")
        else:
            self.receive_text.append("请先连接串口")
            
    def read_test(self):
        if self.port_handler and self.port_handler.is_open():
            try:
                # 解析设备ID
                ids = [int(id_str.strip()) for id_str in self.read_ids.text().split(",") if id_str.strip()]
                if not ids:
                    self.receive_text.append("请输入至少一个设备ID")
                    return
                
                addrs = []
                for i in range(1, 5):
                    if getattr(self, f"read_check_{i}").isChecked():
                        addr = getattr(self, f"read_addr_{i}").text()
                        if addr:
                            addr_int = int(addr)
                            
                            addrs.append(Address.get_address(addr_int))
                            
                connector = SyncConnector(self.port_handler)
                result = connector.read(ids, addrs)
                if result.is_success():
                    self.receive_text.append("读测试成功")
                else:
                    self.receive_text.append(f"读测试失败: {result.get_error_code()}")
            except Exception as e:
                self.receive_text.append(f"读测试错误: {str(e)}")
        else:
            self.receive_text.append("请先连接串口")
            
    def __update_log_text(self, message):
        self.receive_text.append(message)       
    
    def __update_write_text(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = f"发送: {data}"
        
        self.logSignal.emit(data)

    def __update_read_text(self, data):
        data = ''.join([f"{byte:02X} " for byte in data])
        data = f"接收: {data}"

        self.logSignal.emit(data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SerialGUI()
    gui.show()
    sys.exit(app.exec_())
    