import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QComboBox, QPushButton, QFrame, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from SDK.sync_connector import SyncConnector
from SDK.global_state import Address
from SDK.port_handler import PortHandler
import serial.tools.list_ports


class OffsetWriter:
    @staticmethod
    def write_offset(sync_connector, servo_id, offset):
        if not (sync_connector and sync_connector.port_handler.is_open()):
            return False
        offset_value = int(round(offset))
        if not (0 <= offset_value < 4096):
            print(f"错误: 无效的偏移量 {offset_value}。值必须在 [0, 4095] 范围内。")
            return False
        result = sync_connector.write(servo_id, Address.POSITION_OFFSET, offset_value)
        if result and result.is_success():
            print(f"成功写入偏移量 {offset_value} 到舵机 {servo_id}")
            return True
        else:
            print(f"写入偏移量到舵机 {servo_id} 失败")
            return False


class OffsetReader:
    @staticmethod
    def read_offset(sync_connector, servo_id):
        if sync_connector and sync_connector.port_handler.is_open():
            result = sync_connector.read(servo_id, Address.POSITION_OFFSET)
            if result and result.is_success():
                return result.get_data(Address.POSITION_OFFSET)
        return "读取失败"


class AllPositionsReader(QObject):
    positions_updated = pyqtSignal(dict)
    
    def __init__(self, sync_connector):
        super().__init__()
        self._sync_connector = sync_connector
        self._servo_ids = list(range(1, 7))
    
    def read_all_positions(self):
        positions = {}
        if self._sync_connector and self._sync_connector.port_handler.is_open():
            for servo_id in self._servo_ids:
                result = self._sync_connector.read(servo_id, Address.CURRENT_POSITION)
                if result and result.is_success():
                    pos = result.get_data(Address.CURRENT_POSITION)
                    positions[servo_id] = pos
            if positions:
                self.positions_updated.emit(positions)


class CalibrationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GenkiArm一键标定工具')
        self.setGeometry(100, 100, 900, 700)
        
        # 初始化连接组件
        self.port_handler = PortHandler()
        self.sync_connector = SyncConnector(self.port_handler)
        self.is_connected = False
        
        # 位置读取器
        self.reader_thread = QThread()
        self.positions_reader = AllPositionsReader(self.sync_connector)
        self.positions_reader.moveToThread(self.reader_thread)
        self.positions_reader.positions_updated.connect(self.update_position_labels)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.positions_reader.read_all_positions)
        self.reader_thread.start()
        
        # 存储当前位置和偏移量
        self.current_positions = {}
        self.current_offsets = {}
        
        self.setup_ui()
        self.scan_ports()
    
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 标题
        title_label = QLabel('GenkiArm 一键标定工具')
        title_label.setFont(QFont('Arial', 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 连接区域
        connection_frame = QFrame()
        connection_frame.setFrameStyle(QFrame.Box)
        connection_layout = QHBoxLayout(connection_frame)
        
        port_label = QLabel('串口:')
        self.port_combo = QComboBox()
        baud_label = QLabel('波特率:')
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "1000000"])
        self.baud_combo.setCurrentText('230400')
        
        self.connect_button = QPushButton('连接')
        self.refresh_button = QPushButton('刷新')
        
        self.connect_button.clicked.connect(self.toggle_connection)
        self.refresh_button.clicked.connect(self.scan_ports)
        
        connection_layout.addWidget(port_label)
        connection_layout.addWidget(self.port_combo, 1)
        connection_layout.addSpacing(20)
        connection_layout.addWidget(baud_label)
        connection_layout.addWidget(self.baud_combo)
        connection_layout.addSpacing(20)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.refresh_button)
        connection_layout.addStretch(1)
        
        main_layout.addWidget(connection_frame)
        
        # 说明区域
        instruction_frame = QFrame()
        instruction_frame.setFrameStyle(QFrame.Box)
        instruction_layout = QVBoxLayout(instruction_frame)
        
        instruction_title = QLabel('使用说明:')
        instruction_title.setFont(QFont('Arial', 14, QFont.Bold))
        
        instructions = [
            '1. 连接串口后，手动调整所有舵机到合适的中间位置',
            '2. 点击"一键重置所有偏移量"清除之前的标定数据',
            '3. 点击"一键设置当前位置为中心点"完成标定',
            '4. 查看日志确认标定结果'
        ]
        
        instruction_layout.addWidget(instruction_title)
        for instruction in instructions:
            label = QLabel(instruction)
            label.setFont(QFont('Arial', 12))
            instruction_layout.addWidget(label)
        
        main_layout.addWidget(instruction_frame)
        
        # 位置显示区域
        position_frame = QFrame()
        position_frame.setFrameStyle(QFrame.Box)
        position_layout = QVBoxLayout(position_frame)
        
        position_title = QLabel('舵机当前位置:')
        position_title.setFont(QFont('Arial', 14, QFont.Bold))
        position_layout.addWidget(position_title)
        
        # 位置网格
        position_grid = QGridLayout()
        self.position_labels = {}
        self.offset_labels = {}
        
        for i in range(1, 7):
            # 舵机ID标签
            id_label = QLabel(f'{i}号舵机:')
            id_label.setFont(QFont('Arial', 12, QFont.Bold))
            
            # 当前位置标签
            pos_label = QLabel('N/A')
            pos_label.setFont(QFont('Arial', 12))
            pos_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
            
            # 当前偏移量标签
            offset_label = QLabel('偏移: N/A')
            offset_label.setFont(QFont('Arial', 10))
            offset_label.setStyleSheet("color: #666;")
            
            self.position_labels[i] = pos_label
            self.offset_labels[i] = offset_label
            
            row = (i - 1) // 3
            col = (i - 1) % 3
            
            # 为每行预留3行空间：ID标签、位置标签、偏移量标签
            position_grid.addWidget(id_label, row * 3, col)
            position_grid.addWidget(pos_label, row * 3 + 1, col)
            position_grid.addWidget(offset_label, row * 3 + 2, col)
        
        position_layout.addLayout(position_grid)
        main_layout.addWidget(position_frame)
        
        # 操作按钮区域
        button_frame = QFrame()
        button_frame.setFrameStyle(QFrame.Box)
        button_layout = QHBoxLayout(button_frame)
        
        self.reset_all_button = QPushButton('一键重置所有偏移量')
        self.reset_all_button.setMinimumHeight(50)
        self.reset_all_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.reset_all_button.setStyleSheet("background-color: #ff6b6b; color: white;")
        self.reset_all_button.clicked.connect(self.reset_all_offsets)
        self.reset_all_button.setEnabled(False)
        
        self.calibrate_all_button = QPushButton('一键设置当前位置为中心点')
        self.calibrate_all_button.setMinimumHeight(50)
        self.calibrate_all_button.setFont(QFont('Arial', 12, QFont.Bold))
        self.calibrate_all_button.setStyleSheet("background-color: #4ecdc4; color: white;")
        self.calibrate_all_button.clicked.connect(self.calibrate_all_servos)
        self.calibrate_all_button.setEnabled(False)
        
        button_layout.addWidget(self.reset_all_button)
        button_layout.addWidget(self.calibrate_all_button)
        
        main_layout.addWidget(button_frame)
        
        # 日志区域
        log_frame = QFrame()
        log_frame.setFrameStyle(QFrame.Box)
        log_layout = QVBoxLayout(log_frame)
        
        log_title = QLabel('操作日志:')
        log_title.setFont(QFont('Arial', 12, QFont.Bold))
        log_layout.addWidget(log_title)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont('Consolas', 10))
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_frame)
    
    def scan_ports(self):
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        self.log(f"扫描到 {len(ports)} 个串口")
    
    def toggle_connection(self):
        if self.port_handler.is_open():
            self.disconnect_serial()
        else:
            self.connect_serial()
    
    def connect_serial(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baud_combo.currentText())
        
        if not port:
            self.log("错误: 请选择串口")
            return
        
        self.port_handler.baudrate = baudrate
        if self.port_handler.open(port):
            self.connect_button.setText("断开")
            self.is_connected = True
            self.reset_all_button.setEnabled(True)
            self.calibrate_all_button.setEnabled(True)
            self.timer.start(200)  # 每200ms读取一次位置
            self.log(f"串口连接成功: {port} @ {baudrate}")
            self.update_offsets()
        else:
            self.log(f"串口连接失败: {port}")
    
    def disconnect_serial(self):
        self.port_handler.close()
        self.connect_button.setText("连接")
        self.is_connected = False
        self.reset_all_button.setEnabled(False)
        self.calibrate_all_button.setEnabled(False)
        self.timer.stop()
        self.log("串口已断开")
        
        # 清空显示
        for i in range(1, 7):
            self.position_labels[i].setText('N/A')
            self.offset_labels[i].setText('偏移: N/A')
    
    def update_position_labels(self, positions):
        self.current_positions = positions
        for servo_id, position in positions.items():
            if servo_id in self.position_labels:
                self.position_labels[servo_id].setText(str(position))
    
    def update_offsets(self):
        """更新所有舵机的偏移量显示"""
        if not self.is_connected:
            return
        
        for servo_id in range(1, 7):
            offset = OffsetReader.read_offset(self.sync_connector, servo_id)
            self.current_offsets[servo_id] = offset
            if servo_id in self.offset_labels:
                self.offset_labels[servo_id].setText(f'偏移: {offset}')
    
    def reset_all_offsets(self):
        """重置所有舵机的偏移量为0"""
        if not self.is_connected:
            self.log("错误: 请先连接串口")
            return
        
        reply = QMessageBox.question(self, '确认重置', 
                                   '确定要重置所有舵机的偏移量为0吗？\n这将清除之前的标定数据。',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        success_count = 0
        self.log("开始重置所有舵机偏移量...")
        
        for servo_id in range(1, 7):
            if OffsetWriter.write_offset(self.sync_connector, servo_id, 0):
                success_count += 1
                self.log(f"舵机 {servo_id} 偏移量重置成功")
            else:
                self.log(f"舵机 {servo_id} 偏移量重置失败")
        
        self.log(f"重置完成: {success_count}/6 个舵机成功")
        self.update_offsets()
    
    def calibrate_all_servos(self):
        """一键标定所有舵机"""
        if not self.is_connected:
            self.log("错误: 请先连接串口")
            return
        
        if not self.current_positions:
            self.log("错误: 无法读取舵机位置，请检查连接")
            return
        
        reply = QMessageBox.question(self, '确认标定', 
                                   '确定要将当前位置设置为所有舵机的中心点吗？\n请确保所有舵机都已调整到合适的中间位置。',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        success_count = 0
        self.log("开始标定所有舵机...")
        
        for servo_id in range(1, 7):
            if servo_id in self.current_positions:
                current_pos = self.current_positions[servo_id]
                # 计算偏移量: (当前位置 - 2048 + 4096) % 4096
                offset = (current_pos - 2048 + 4096) % 4096
                
                if OffsetWriter.write_offset(self.sync_connector, servo_id, offset):
                    success_count += 1
                    self.log(f"舵机 {servo_id}: 位置 {current_pos} -> 偏移量 {offset} (成功)")
                else:
                    self.log(f"舵机 {servo_id}: 位置 {current_pos} -> 偏移量 {offset} (失败)")
            else:
                self.log(f"舵机 {servo_id}: 无法读取当前位置")
        
        self.log(f"标定完成: {success_count}/6 个舵机成功")
        self.update_offsets()
        
        if success_count == 6:
            QMessageBox.information(self, '标定完成', '所有舵机标定成功！')
        else:
            QMessageBox.warning(self, '标定完成', f'标定完成，但有 {6-success_count} 个舵机失败，请检查连接。')
    
    def log(self, message):
        """添加日志信息"""
        self.log_text.append(f"[{self.get_timestamp()}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_connected:
            self.disconnect_serial()
        
        self.reader_thread.quit()
        self.reader_thread.wait()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalibrationTool()
    window.show()
    sys.exit(app.exec_())