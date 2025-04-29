# record_data_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import pyqtSignal, QTimer
import numpy as np
import os
from app_styles import AppStyles

class RecordDataWidget(QGroupBox):
    sample_recorded = pyqtSignal(np.ndarray, str)

    def __init__(self, real_time_graph):
        super().__init__("Record Data")
        # Apply heading style
        self.setStyleSheet(AppStyles.get_group_box_style())

        self.real_time_graph = real_time_graph
        self.is_recording = False
        self.recorded_data = None
        self.manual_stop = False  # Flag to track if recording was manually stopped

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Sample length input
        sample_length_layout = QHBoxLayout()
        # sample_length_layout.setSpacing(AppStyles.WIDGET_SPACING)
        sample_length_label = QLabel("Sample length (sec)")
        sample_length_label.setStyleSheet(AppStyles.get_label_style())
        self.sample_length_input = QLineEdit()
        self.sample_length_input.setStyleSheet(AppStyles.get_line_edit_style())
        self.sample_length_input.setText("1")
        sample_length_layout.addWidget(sample_length_label)
        sample_length_layout.addWidget(self.sample_length_input)
        layout.addLayout(sample_length_layout)

        # Note
        note_label = QLabel("Use the same sample length both here and in Inference.ino")
        note_label.setStyleSheet(AppStyles.get_note_label_style())
        layout.addWidget(note_label)

        # Class label input with intuitive default
        class_label_layout = QHBoxLayout()
        class_label_layout.setSpacing(AppStyles.WIDGET_SPACING)
        class_label_label = QLabel("Class Label")
        class_label_label.setStyleSheet(AppStyles.get_label_style())
        self.class_label_input = QLineEdit()
        self.class_label_input.setStyleSheet(AppStyles.get_line_edit_style())
        self.class_label_input.setText("")  # Intuitive default label
        class_label_layout.addWidget(class_label_label)
        class_label_layout.addWidget(self.class_label_input)
        layout.addLayout(class_label_layout)

        # Note
        note_label = QLabel("Do not use hyphens '-' in the class names")
        note_label.setStyleSheet(AppStyles.get_note_label_style())
        layout.addWidget(note_label)

        # Record button
        self.record_button = QPushButton("Record")
        self.record_button.setStyleSheet(AppStyles.get_button_style())
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)

        # Note
        note_label = QLabel("Note: Upload Record.ino to the Arduino\nboard before recording data")
        note_label.setStyleSheet(AppStyles.get_note_label_style())
        layout.addWidget(note_label)

        # Initialize recording variables
        self.recording_buffer = []
        self.counter = {}  # To keep track of repetition count for each label

        # Timer for recording progress
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_progress)
        self.recording_time = 0
        self.target_recording_time = 0
        self.progress_update_interval = 100  # milliseconds

    def toggle_recording(self):
        if not self.is_recording:
            # Check if the real-time graph is running
            if not self.check_real_time_graph_running():
                return

            # Check if class label is provided
            if not self.validate_class_label():
                return

            # Get sample length
            try:
                sample_length = float(self.sample_length_input.text())
                if sample_length <= 0:
                    self.show_error_message("Invalid Input", "Sample length must be positive")
                    return
            except ValueError:
                self.show_error_message("Invalid Input", "Invalid sample length")
                return

            # Start recording
            self.is_recording = True
            self.manual_stop = False  # Reset manual stop flag
            self.recording_buffer = []

            # Set up recording timer
            self.recording_time = 0
            self.target_recording_time = sample_length
            self.recording_timer.start(self.progress_update_interval)

            # Update button text
            self.record_button.setText(f"Recording: 0.0/{sample_length:.1f}s")

            # Connect to data updates from the real-time graph
            self.real_time_graph.data_updated.connect(self.capture_data)
        else:
            # Manual stop - set flag
            self.manual_stop = True
            # Stop recording
            self.stop_recording()

    def validate_class_label(self):
        """Validate that a class label has been provided"""
        if not self.class_label_input.text().strip():
            self.show_error_message("Missing Class Label", "Please enter a class label before recording")
            return False
        return True

    def check_real_time_graph_running(self):
        """Check if the real-time graph is running"""
        # For Arduino data, check if serial connection is active
        if self.real_time_graph.data_source == self.real_time_graph.USE_ARDUINO_DATA:
            if not self.real_time_graph.serial_handler.running:
                self.show_error_message("Not Connected",
                                        "Please connect an IMU device before recording data")
                return False

        # For both data sources, check if timer is active
        if not self.real_time_graph.timer.isActive():
            self.show_error_message("Graph Not Running",
                                    "The real-time graph is not active. Please ensure data is flowing")
            return False

        return True

    def show_error_message(self, title, message):
        """Display an error message box"""
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Warning)
        error_dialog.setText(title)
        error_dialog.setInformativeText(message)
        error_dialog.setWindowTitle(title)
        error_dialog.exec_()

    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False
        self.record_button.setText("Record")
        self.recording_timer.stop()

        # Disconnect from data updates
        self.real_time_graph.data_updated.disconnect(self.capture_data)

        # Process and emit the recorded data only if not manually stopped
        if self.recording_buffer and not self.manual_stop:
            self.process_recorded_data()
        else:
            self.recording_buffer = []  # Clear buffer if manually stopped

    def capture_data(self, data):
        """Capture current frame of data"""
        self.recording_buffer.append(data[-1].copy())  # Only capture the latest point

    def update_recording_progress(self):
        """Update the recording progress and button text"""
        self.recording_time += self.progress_update_interval / 1000.0  # Convert ms to seconds

        # Update button text
        self.record_button.setText(f"Recording: {self.recording_time:.1f}/{self.target_recording_time:.1f}s")

        # Check if recording should stop
        if self.recording_time >= self.target_recording_time:
            self.stop_recording()

    def process_recorded_data(self):
        """Process the recorded data and emit a signal"""
        if not self.recording_buffer:
            return

        # Get the class label
        class_label = self.class_label_input.text().strip()
        if not class_label:
            class_label = "EmptyLabel"  # Fallback to default if empty

        # Increment the repetition counter for this label
        if class_label not in self.counter:
            self.counter[class_label] = 1
        else:
            self.counter[class_label] += 1

        # Create the full label with repetition count
        full_label = f"{class_label}-{self.counter[class_label]}"

        # Convert the buffer to a numpy array
        recorded_data = np.array(self.recording_buffer)

        # Emit the signal with the recorded data and label
        self.sample_recorded.emit(recorded_data, full_label)

        # Clear the buffer
        self.recording_buffer = []