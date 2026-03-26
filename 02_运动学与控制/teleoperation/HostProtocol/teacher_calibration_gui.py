import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QComboBox, QPushButton, QListWidget, QListWidgetItem, QFrame, QStackedWidget, QLineEdit
)
from PyQt5.QtGui import QFont, QMovie, QPixmap
from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from SDK.sync_connector import SyncConnector
from SDK.global_state import Address
from SDK.port_handler import PortHandler
import serial.tools.list_ports


class InstructionPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        font = QFont('Arial', 14)
        
        label1 = QLabel('1. 选择正确的串口, 点击连接')
        label1.setFont(font)
        label2 = QLabel('2. 连接成功后, 点击下一步')
        label2.setFont(font)
        label3 = QLabel('3. 如果连接错误, 程序会阻塞, 重启即可')
        label3.setFont(font)
        
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)

class PositionReader(QObject):
    position_updated = pyqtSignal(int)

    def __init__(self, sync_connector, servo_id):
        super().__init__()
        self._sync_connector = sync_connector
        self._servo_id = servo_id
    
    def read_position(self):
        """This method will run in a separate thread."""
        if self._sync_connector and self._sync_connector.port_handler.is_open():
            result = self._sync_connector.read(self._servo_id, Address.CURRENT_POSITION)
            if result and result.is_success():
                position = result.get_data(Address.CURRENT_POSITION)
                if position is not None:
                    self.position_updated.emit(position)

class OffsetWriter:
    @staticmethod
    def write_offset(sync_connector, servo_id, offset):
        if sync_connector and sync_connector.port_handler.is_open():
            # Ensure offset is an integer before writing
            offset_value = int(offset)
            result = sync_connector.write(servo_id, Address.POSITION_OFFSET, offset_value)
            if result and result.is_success():
                print(f"Successfully wrote offset {offset_value} to servo {servo_id}")
                return True
            else:
                print(f"Failed to write offset to servo {servo_id}")
                return False
        return False

class OffsetReader:
    @staticmethod
    def read_offset(sync_connector, servo_id):
        if sync_connector and sync_connector.port_handler.is_open():
            result = sync_connector.read(servo_id, Address.POSITION_OFFSET)
            if result and result.is_success():
                return result.get_data(Address.POSITION_OFFSET)
        return None

class ServoCalibrationPage(QWidget):
    calibration_confirmed = pyqtSignal(int) # Signal with servo_id
    calibration_reset = pyqtSignal(int)   # Signal when reset is clicked
    def __init__(self, servo_id, sync_connector: SyncConnector):
        super().__init__()
        self.servo_id = servo_id
        self.limit1 = None
        self.limit2 = None
        self.mid_pos = None

        # Setup worker thread for reading position
        self.reader_thread = QThread()
        self.position_reader = PositionReader(sync_connector, self.servo_id)
        self.position_reader.moveToThread(self.reader_thread)
        self.position_reader.position_updated.connect(self.update_position_label)
        self.sync_connector = sync_connector

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.position_reader.read_position)
        
        self.reader_thread.start()

        main_layout = QHBoxLayout(self)

        # Left side for GIF
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.gif_label = QLabel()
        self.movie = QMovie("assets/joint{}.gif".format(self.servo_id))
        self.movie.frameChanged.connect(self.update_gif_frame)
        self.gif_label.setMovie(self.movie)
        self.movie.start()
        left_layout.addWidget(self.gif_label)
        left_layout.setAlignment(Qt.AlignCenter)

        # Right side for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Top part: instruction text
        instruction_label = QLabel(f'请将{self.servo_id}号舵机旋转至两个极限位置')
        instruction_label.setFont(QFont('Arial', 12))
        instruction_label.setWordWrap(True)

        # Middle part: input boxes and buttons
        controls_layout = QVBoxLayout()
        
        self.lb_pos = QLabel('2048')

        self.pos1_button = QPushButton('设置极限值1')
        self.pos1_button.clicked.connect(self.set_limit1)
        self.pos2_button = QPushButton('设置极限值2')
        self.pos2_button.clicked.connect(self.set_limit2)
        self.set_mid_button = QPushButton('设置中值')
        self.set_mid_button.clicked.connect(self.set_mid_point)

        self.confirm_button = QPushButton('确认')
        self.confirm_button.clicked.connect(self.on_confirm)
        self.reset_button = QPushButton('重置')
        self.reset_button.clicked.connect(self.reset_calibration)

        self.current_offset_label = QLabel('当前偏移量: N/A')

        self.limit1_label = QLabel('极限值1: N/A')
        self.limit2_label = QLabel('极限值2: N/A')
        self.offset_label = QLabel('偏移量: 未计算')

        controls_layout.addWidget(self.lb_pos)

        if self.servo_id in [5, 6]:
            instruction_label.setText(f'请将{self.servo_id}号舵机旋转至中位')
            self.pos1_button.hide()
            self.pos2_button.hide()
            self.limit1_label.hide()
            self.limit2_label.hide()
            controls_layout.addWidget(self.set_mid_button)
        else:
            controls_layout.addWidget(self.pos1_button)
            controls_layout.addWidget(self.limit1_label)
            controls_layout.addWidget(self.pos2_button)
            controls_layout.addWidget(self.limit2_label)
            self.set_mid_button.hide()
        controls_layout.addWidget(self.offset_label)
        controls_layout.addWidget(self.confirm_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.current_offset_label)

        right_layout.addWidget(instruction_label)
        right_layout.addLayout(controls_layout)
        right_layout.addStretch(1)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

    def update_gif_frame(self):
        pixmap = self.movie.currentPixmap()
        scaled_pixmap = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
        self.gif_label.setPixmap(scaled_pixmap)

    def update_position_label(self, pos):
        self.lb_pos.setText(str(pos))

    def update_offset_label(self):
        offset = OffsetReader.read_offset(self.sync_connector, self.servo_id)
        if offset is not None:
            self.current_offset_label.setText(f'当前偏移量: {offset}')
        else:
            self.current_offset_label.setText('当前偏移量: 读取失败')

    def set_limit1(self):
        try:
            pos = int(self.lb_pos.text())
            self.limit1 = pos
            self.limit1_label.setText(f'极限值1: {self.limit1}')
        except (ValueError, TypeError):
            self.limit1_label.setText('极限值1: 无效值')

    def set_limit2(self):
        try:
            pos = int(self.lb_pos.text())
            self.limit2 = pos
            self.limit2_label.setText(f'极限值2: {self.limit2}')
        except (ValueError, TypeError):
            self.limit2_label.setText('极限值2: 无效值')

    def set_mid_point(self):
        try:
            pos = int(self.lb_pos.text())
            self.mid_pos = pos
            self.offset_label.setText(f'偏移量: {self.mid_pos}')
        except (ValueError, TypeError):
            self.offset_label.setText('偏移量: 无效值')

    def on_confirm(self):
        if self.servo_id in [5, 6]:
            offset = 0
            if self.mid_pos is not None:
                if self.mid_pos > 2048:
                    offset = self.mid_pos - 2048
                else:
                    offset = 2048 + self.mid_pos
            else:
                # Handle case where mid_pos is not set
                return
        elif self.limit1 is not None and self.limit2 is not None:
            offset = 0 # Default offset
            if self.servo_id == 1:
                max_range = 2040
                if abs(self.limit1 - self.limit2) > max_range:
                    offset =(int((4096 + self.limit1 + self.limit2)/2) + 2048) % 4096
                else:
                    current = int((self.limit1 + self.limit2)/2)
                    offset = current
                    if current > 2048:
                        offset = current - 2048
                    else:
                        offset = 2048 - current + 2048
            elif self.servo_id == 2:
                max_range = 2100
                if abs(self.limit1 - self.limit2) < max_range:
                    offset =(int((4096 + self.limit1 + self.limit2)/2) + 2048) % 4096
                else:
                    current = int((self.limit1 + self.limit2)/2)
                    offset = current
                    if current > 2048:
                        offset = current - 2048
                    else:
                        offset = 2048 - current + 2048
            elif self.servo_id == 3:
                max_range = 2100
                if abs(self.limit1 - self.limit2) < max_range:
                    offset =(int((4096 + self.limit1 + self.limit2)/2) + 2048) % 4096
                else:
                    current = int((self.limit1 + self.limit2)/2)
                    offset = current
                    if current > 2048:
                        offset = current - 2048
                    else:
                        offset = 2048 - current + 2048
            elif self.servo_id == 4:
                max_range = 2100
                if abs(self.limit1 - self.limit2) < max_range:
                    offset =(int((4096 + self.limit1 + self.limit2)/2) + 2048) % 4096
                else:
                    current = int((self.limit1 + self.limit2)/2)
                    offset = current
                    if current > 2048:
                        offset = current - 2048
                    else:
                        offset = 2048 - current + 2048
            else:
                pass
        else:
            # Handle case where limits are not set for servos 1-4
            return
                
        self.offset_label.setText(f'偏移量: {offset:.0f}')
        # Write the offset to the servo
        write_success = OffsetWriter.write_offset(self.sync_connector, self.servo_id, offset)
        if write_success:
            self.offset_label.setText(f'偏移量: {offset:.0f} (已写入)')
            # Re-read the offset to confirm
            self.update_offset_label()
        else:
            self.offset_label.setText(f'偏移量: {offset:.0f} (写入失败)')

        self.calibration_confirmed.emit(self.servo_id)
        if self.servo_id not in [5, 6]:
            self.pos1_button.setEnabled(False)
            self.pos2_button.setEnabled(False)
        else:
            self.set_mid_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        self.confirm_button.setText('已确认')

    def reset_calibration(self):
        self.limit1 = None
        self.limit2 = None
        self.mid_pos = None
        self.limit1_label.setText('极限值1: N/A')
        self.limit2_label.setText('极限值2: N/A')
        self.offset_label.setText('偏移量: 未计算')
        if self.servo_id not in [5, 6]:
            self.pos1_button.setEnabled(True)
            self.pos2_button.setEnabled(True)
        else:
            self.set_mid_button.setEnabled(True)
        self.confirm_button.setEnabled(True)
        self.confirm_button.setText('确认')

        # Reset offset to 0
        write_success = OffsetWriter.write_offset(self.sync_connector, self.servo_id, 0)
        if write_success:
            print(f"Offset for servo {self.servo_id} has been reset to 0.")
        else:
            print(f"Failed to reset offset for servo {self.servo_id}.")

        self.update_offset_label() # Re-read offset on reset
        self.calibration_reset.emit(self.servo_id)

    def start_polling(self):
        self.timer.start(100)  # Poll every 100ms
        self.update_offset_label() # Initial read of offset

    def stop_polling(self):
        self.timer.stop()

    def cleanup(self):
        self.reader_thread.quit()
        self.reader_thread.wait()

class AllPositionsReader(QObject):
    positions_updated = pyqtSignal(dict)

    def __init__(self, sync_connector):
        super().__init__()
        self._sync_connector = sync_connector
        self._servo_ids = list(range(1, 7))

    def read_all_positions(self):
        positions = {}
        if self._sync_connector and self._sync_connector.port_handler.is_open():
            # results = self._sync_connector.read(self._servo_ids, Address.CURRENT_POSITION)
            for servo_id in self._servo_ids:
                result = self._sync_connector.read(servo_id, Address.CURRENT_POSITION)
                if result and result.is_success():
                    pos = result.get_data(Address.CURRENT_POSITION)
                    if pos is not None:
                        positions[servo_id] = pos
            self.positions_updated.emit(positions)


class SimulationPage(QWidget):
    def __init__(self, sync_connector: SyncConnector):
        super().__init__()
        self.sync_connector = sync_connector
        self.position_labels = {}

        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        for i in range(1, 7):
            id_label = QLabel(f'{i}号舵机位置:')
            id_label.setFont(QFont('Arial', 14))
            pos_label = QLabel('N/A')
            pos_label.setFont(QFont('Arial', 14))
            self.position_labels[i] = pos_label
            layout.addWidget(id_label, i - 1, 0)
            layout.addWidget(pos_label, i - 1, 1)

        # Setup worker thread for reading all positions
        self.reader_thread = QThread()
        self.positions_reader = AllPositionsReader(self.sync_connector)
        self.positions_reader.moveToThread(self.reader_thread)
        self.positions_reader.positions_updated.connect(self.update_position_labels)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.positions_reader.read_all_positions)
        self.reader_thread.start()

    def update_position_labels(self, positions):
        for servo_id, position in positions.items():
            if servo_id in self.position_labels:
                self.position_labels[servo_id].setText(str(position))

    def start_polling(self):
        self.timer.start(200)  # Poll every 200ms

    def stop_polling(self):
        self.timer.stop()

    def cleanup(self):
        self.reader_thread.quit()
        self.reader_thread.wait()


class CalibrationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GenkiArm标定工具')
        self.setGeometry(100, 100, 800, 600)

        self.port_handler = PortHandler()
        self.sync_connector = SyncConnector(self.port_handler)
        self.is_connected = False
        self.confirmed_servos = set()

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top section for connection
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        port_label = QLabel('串口:')
        self.port_combo = QComboBox()
        
        baud_label = QLabel('波特率:')
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "1000000"])
        self.baud_combo.setCurrentText('230400')
        
        self.connect_button = QPushButton('连接')
        self.connect_button.clicked.connect(self.toggle_connection)
        self.refresh_button = QPushButton('刷新')
        self.refresh_button.clicked.connect(self.scan_ports)
        
        top_layout.addWidget(port_label)
        top_layout.addWidget(self.port_combo, 1)
        top_layout.addSpacing(20)
        top_layout.addWidget(baud_label)
        top_layout.addWidget(self.baud_combo)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.connect_button)
        top_layout.addWidget(self.refresh_button)
        top_layout.addStretch(1)

        # Main content area
        content_layout = QHBoxLayout()

        # Left navigation panel
        self.left_panel = QListWidget()
        self.left_panel.setFixedWidth(200)
        self.left_panel.setFont(QFont('Arial', 12))
        self.left_panel.setStyleSheet("""
            QListWidget::item:!enabled { 
                background-color: #d3d3d3; 
                color: #808080; 
            }
            QListWidget::item:enabled { 
                background-color: #ffffff; 
                color: #000000; 
            }
            QListWidget::item:selected { 
                background-color: #a6d8f4; 
                color: #000000; 
            }
        """)
        
        # Add items to the list
        QListWidgetItem('标定说明', self.left_panel)
        QListWidgetItem('1号舵机标定', self.left_panel)
        QListWidgetItem('2号舵机标定', self.left_panel)
        QListWidgetItem('3号舵机标定', self.left_panel)
        QListWidgetItem('4号舵机标定', self.left_panel)
        QListWidgetItem('5号舵机标定', self.left_panel)
        QListWidgetItem('6号舵机标定', self.left_panel)
        QListWidgetItem('仿真确认', self.left_panel)
        self.left_panel.setCurrentRow(0)

        # Center content area - using QStackedWidget
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(InstructionPage())
        for i in range(1, 7):
            page = ServoCalibrationPage(i, self.sync_connector)
            page.calibration_confirmed.connect(self.on_calibration_confirmed)
            page.calibration_reset.connect(self.on_calibration_reset)
            self.stacked_widget.addWidget(page)
        self.stacked_widget.addWidget(SimulationPage(self.sync_connector))

        # Connect list widget to stacked widget
        self.left_panel.currentRowChanged.connect(self.on_page_changed)

        # 初始时禁用左侧列表和右侧内容
        self.left_panel.setEnabled(False)
        self.stacked_widget.setEnabled(False)

        # Add panels to content layout
        content_layout.addWidget(self.left_panel)
        content_layout.addWidget(self.stacked_widget, 1)

        # Bottom section
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.next_button = QPushButton('下一步')
        self.next_button.setEnabled(False)
        self.next_button.setFixedWidth(150)
        self.next_button.clicked.connect(self.go_to_next_step)
        bottom_layout.addWidget(self.next_button)

        # Add all sections to main layout
        main_layout.addWidget(top_widget)
        main_layout.addLayout(content_layout, 1)
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        main_layout.addLayout(bottom_layout)

        self.scan_ports()

    def scan_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)

    def update_ui_state(self, is_connected):
        self.is_connected = is_connected
        self.stacked_widget.setEnabled(is_connected)
        self.next_button.setEnabled(is_connected)

        if is_connected:
            self.left_panel.setEnabled(True)
            for i in range(self.left_panel.count()):
                item = self.left_panel.item(i)
                item.setFlags(item.flags() | Qt.ItemIsEnabled if i == 0 else item.flags() & ~Qt.ItemIsEnabled)
            self.left_panel.setCurrentRow(0)
        else:
            self.left_panel.setEnabled(False)

    def go_to_next_step(self):
        current_row = self.left_panel.currentRow()
        if current_row < self.left_panel.count() - 1:
            next_item = self.left_panel.item(current_row + 1)
            next_item.setFlags(next_item.flags() | Qt.ItemIsEnabled)
            self.left_panel.setCurrentRow(current_row + 1)
            self.next_button.setEnabled(False) # Disable after moving to next

    def on_calibration_confirmed(self, servo_id):
        self.confirmed_servos.add(servo_id)
        # Re-evaluate button state if the current page is the one that was just confirmed
        current_widget = self.stacked_widget.currentWidget()
        if isinstance(current_widget, ServoCalibrationPage) and current_widget.servo_id == servo_id:
            self.next_button.setEnabled(True)

    def on_calibration_reset(self, servo_id):
        if servo_id in self.confirmed_servos:
            self.confirmed_servos.remove(servo_id)
        # Re-evaluate button state if the current page is the one that was just reset
        current_widget = self.stacked_widget.currentWidget()
        if isinstance(current_widget, ServoCalibrationPage) and current_widget.servo_id == servo_id:
            self.next_button.setEnabled(False)

    def on_page_changed(self, index):
        self.stacked_widget.setCurrentIndex(index)
        widget = self.stacked_widget.widget(index)

        # Allow proceeding from instruction and simulation pages directly
        if isinstance(widget, (InstructionPage, SimulationPage)):
            self.next_button.setEnabled(True)
        elif isinstance(widget, ServoCalibrationPage):
            # For servo pages, enable only if already confirmed
            self.next_button.setEnabled(widget.servo_id in self.confirmed_servos)
        else:
            self.next_button.setEnabled(False)

        # Stop all timers
        for i in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(i)
            if isinstance(widget, (ServoCalibrationPage, SimulationPage)):
                widget.stop_polling()

        # Start the timer for the newly selected page if the index is valid
        if index != -1:
            widget = self.stacked_widget.widget(index)
            if isinstance(widget, (ServoCalibrationPage, SimulationPage)):
                widget.start_polling()

    def closeEvent(self, event):
        # Clean up all threads when closing the window
        for i in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(i)
            if hasattr(widget, 'cleanup'):
                widget.cleanup()
        super().closeEvent(event)

    def toggle_connection(self):
        if self.port_handler.is_open():
            self.port_handler.close()
            self.connect_button.setText("连接")
            print("串口已关闭")
            self.update_ui_state(False)
        else:
            port = self.port_combo.currentText()
            baudrate = int(self.baud_combo.currentText())
            self.port_handler.baudrate = baudrate
            if self.port_handler.open(port):
                self.connect_button.setText("断开")
                print(f"串口已连接: {port} @ {baudrate}")
                self.update_ui_state(True)
            else:
                print("串口连接失败")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CalibrationTool()
    window.show()
    sys.exit(app.exec_())