# real_time_graph.py
import numpy as np
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton,
                             QHBoxLayout, QCheckBox, QMessageBox)
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from app_styles import AppStyles
from arduino_serial_handler import ArduinoSerialHandler


class RealTimeGraph(QWidget):
    data_updated = pyqtSignal(np.ndarray)
    data_source_changed = pyqtSignal(bool)  # True for real hardware, False for simulated

    # Data source options
    USE_RANDOM_DATA = 0
    USE_ARDUINO_DATA = 1

    def __init__(self, data_source=USE_ARDUINO_DATA):
        super().__init__()

        # Set data source
        self.data_source = data_source

        layout = QVBoxLayout(self)

        # Title
        self.title_text = "Real time IMU data"
        if self.data_source == self.USE_ARDUINO_DATA:
            self.title_text += " from Arduino"
        else:
            self.title_text += " (Simulated Data)"

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet(AppStyles.HEADING_STYLE)
        layout.addWidget(self.title)

        # Data source selector
        data_source_layout = QHBoxLayout()
        self.data_source_checkbox = QCheckBox("Use Arduino Data")
        self.data_source_checkbox.setChecked(self.data_source == self.USE_ARDUINO_DATA)
        self.data_source_checkbox.toggled.connect(self.toggle_data_source)
        data_source_layout.addWidget(self.data_source_checkbox)
        data_source_layout.addStretch()
        layout.addLayout(data_source_layout)

        # Add connection button (only visible for Arduino data)
        button_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect to Arduino")
        self.connect_button.setFixedWidth(200)
        self.connect_button.clicked.connect(self.toggle_connection)
        button_layout.addWidget(self.connect_button)
        button_layout.addStretch()  # Add stretch to push button to the left
        self.connect_button.setVisible(self.data_source == self.USE_ARDUINO_DATA)
        layout.addLayout(button_layout)

        # Initialize plot
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Set up the plot with dark grey background
        self.plot_widget.setBackground('#2D2D2D')  # Dark grey background
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Acceleration')
        self.plot_widget.setLabel('bottom', 'Time', units='s')

        self.plot_widget.enableAutoRange(axis='y')  # Auto-scale Y axis
        self.plot_widget.getViewBox().setAutoVisible(y=True)

        # Create three curve objects, one for each axis
        self.x_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2), name="X-Axis")
        self.y_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=2), name="Y-Axis")
        self.z_curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2), name="Z-Axis")

        # Add legend
        self.plot_widget.addLegend(labelTextColor='w')  # White text for legend

        # Style axis labels with white text
        self.plot_widget.getAxis('left').setTextPen('w')
        self.plot_widget.getAxis('bottom').setTextPen('w')

        label_style = {'color': AppStyles.GRAPH_LABEL_COLOR,
                       'font-size': f'{AppStyles.GRAPH_LABEL_FONT_SIZE}pt'}
        self.plot_widget.getAxis('left').setLabel('Acceleration', **label_style)
        self.plot_widget.getAxis('bottom').setLabel('Time', units='s', **label_style)

        # Set larger font size for tick values
        font = QFont()
        font.setPointSize(AppStyles.TEXT_FONT_SIZE)
        self.plot_widget.getAxis('bottom').setStyle(tickFont=font)
        self.plot_widget.getAxis('left').setStyle(tickFont=font)


        # Data buffer
        self.window_size = 8  # 8 seconds of data at 50Hz = 400 points
        self.sample_rate = 50  # Hz
        self.time_data = np.linspace(0, self.window_size, self.window_size * self.sample_rate)
        self.x_data = np.zeros(self.window_size * self.sample_rate)
        self.y_data = np.zeros(self.window_size * self.sample_rate)
        self.z_data = np.zeros(self.window_size * self.sample_rate)

        # Setup serial handler (for Arduino data)
        self.serial_handler = ArduinoSerialHandler()
        self.serial_handler.data_received.connect(self.on_new_data)
        self.serial_handler.connection_status.connect(self.update_connection_status)

        # Timer for updating the plot
        self.timer = QTimer()
        self.timer.setInterval(20)  # 50Hz = 20ms interval

        if self.data_source == self.USE_ARDUINO_DATA:
            self.timer.timeout.connect(self.update_plot)
        else:
            self.timer.timeout.connect(self.generate_random_data)

        self.timer.start()

    def toggle_data_source(self, checked):
        """Toggle between Arduino and random data sources"""
        # Disconnect Arduino if it was connected
        if self.data_source == self.USE_ARDUINO_DATA and self.serial_handler.running:
            self.serial_handler.disconnect()

        # Update data source
        self.data_source = self.USE_ARDUINO_DATA if checked else self.USE_RANDOM_DATA

        # Emit signal about data source change
        self.data_source_changed.emit(self.data_source == self.USE_ARDUINO_DATA)

        # Update UI elements
        if self.data_source == self.USE_ARDUINO_DATA:
            self.title.setText("Real time IMU data from Arduino Board")
            self.connect_button.setVisible(True)
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.update_plot)
        else:
            self.title.setText("Real time IMU data (Simulated Data)")
            self.connect_button.setVisible(False)
            self.timer.timeout.disconnect()
            self.timer.timeout.connect(self.generate_random_data)

    def toggle_connection(self):
        """Connect or disconnect from the Arduino device"""
        if self.serial_handler.running:
            # Disconnect
            self.serial_handler.disconnect()
            self.connect_button.setText("Connect to Arduino")
        else:
            # Connect and start thread
            try:
                self.serial_handler.start()
                # Button text will be updated via the connection_status signal
            except Exception as e:
                # Show error popup if connection fails
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setText("Connection Error")
                error_dialog.setInformativeText(f"Failed to connect to Arduino device: {str(e)}")
                error_dialog.setWindowTitle("Connection Error")
                error_dialog.setStyleSheet("QLabel{min-width: 400px;} QMessageBox{min-width: 500px;}")
                error_dialog.exec_()

    def update_connection_status(self, connected, message):
        """Update connection status and button state"""
        if connected:
            self.connect_button.setText("Disconnect")
        else:
            self.connect_button.setText("Connect to Arduino")
            # Show error popup if this is a disconnection event (not initial state)
            if message and message != "Disconnected":
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Warning)
                error_dialog.setText("Connection Error")
                error_dialog.setInformativeText(message)
                error_dialog.setWindowTitle("Connection Error")
                error_dialog.exec_()

    def generate_random_data(self):
        """Generate random IMU data"""
        # Generate random IMU data
        new_x = np.random.normal(0, 10)
        new_y = np.random.normal(0, 10)
        new_z = np.random.normal(-10, 15)

        # Roll the arrays to make room for new data
        self.x_data = np.roll(self.x_data, -1)
        self.y_data = np.roll(self.y_data, -1)
        self.z_data = np.roll(self.z_data, -1)

        # Add new data
        self.x_data[-1] = new_x
        self.y_data[-1] = new_y
        self.z_data[-1] = new_z

        # Update plot
        self.x_curve.setData(self.time_data, self.x_data)
        self.y_curve.setData(self.time_data, self.y_data)
        self.z_curve.setData(self.time_data, self.z_data)

        # Emit signal with current data
        current_data = np.column_stack((self.x_data, self.y_data, self.z_data))
        self.data_updated.emit(current_data)

    def on_new_data(self, x, y, z):
        """Handle new data from the serial handler"""
        # Roll the arrays to make room for new data
        self.x_data = np.roll(self.x_data, -1)
        self.y_data = np.roll(self.y_data, -1)
        self.z_data = np.roll(self.z_data, -1)

        # Add new data
        self.x_data[-1] = x
        self.y_data[-1] = y
        self.z_data[-1] = z

    def update_plot(self):
        """Update the plot with current data from Arduino"""
        # Update plot
        self.x_curve.setData(self.time_data, self.x_data)
        self.y_curve.setData(self.time_data, self.y_data)
        self.z_curve.setData(self.time_data, self.z_data)

        # Emit signal with current data
        current_data = np.column_stack((self.x_data, self.y_data, self.z_data))
        self.data_updated.emit(current_data)

    def get_current_data(self):
        """Return the current data buffer"""
        return np.column_stack((self.time_data, self.x_data, self.y_data, self.z_data))

    def closeEvent(self, event):
        """Clean up when the widget is closed"""
        if self.data_source == self.USE_ARDUINO_DATA and self.serial_handler.running:
            self.serial_handler.disconnect()
            self.serial_handler.wait()
        super().closeEvent(event)