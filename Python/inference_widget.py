# inference_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QSpacerItem, QSizePolicy
from app_styles import AppStyles
import os

class InferenceWidget(QGroupBox):
    def __init__(self):
        super().__init__("Inference")
        # Apply heading style
        self.setStyleSheet(AppStyles.get_group_box_style())

        # This widget does not create its own serial handler
        # It will use the one from real_time_graph
        self.arduino_handler = None

        # Keep track of whether real hardware is being used
        self.using_real_hardware = True

        layout = QVBoxLayout(self)
        layout.setSpacing(AppStyles.WIDGET_SPACING)

        # Start live inference button
        self.start_inference_button = QPushButton("Start Live Inference")
        self.start_inference_button.setStyleSheet(AppStyles.get_button_style())
        self.start_inference_button.clicked.connect(self.toggle_inference)
        self.start_inference_button.setEnabled(False)  # Initially disabled until connected
        layout.addWidget(self.start_inference_button)

        # Current prediction layout
        prediction_layout = QHBoxLayout()
        prediction_layout.setSpacing(AppStyles.WIDGET_SPACING)
        prediction_label = QLabel("Current Prediction")
        prediction_label.setStyleSheet(AppStyles.get_label_style())
        self.prediction_display = QLineEdit()
        self.prediction_display.setStyleSheet(AppStyles.get_line_edit_style())
        self.prediction_display.setReadOnly(True)
        prediction_layout.addWidget(prediction_label)
        prediction_layout.addWidget(self.prediction_display)
        layout.addLayout(prediction_layout)

        # Note
        note_label = QLabel("Note: Upload Inference.ino to the Arduino board before\nusing this")
        note_label.setStyleSheet(AppStyles.get_note_label_style())
        layout.addWidget(note_label)

        # Load the class names
        self.class_names = self.load_class_names()

        self.inference_running = False

    def set_arduino_handler(self, handler):
        """Set the Arduino serial handler from main window"""
        # Disconnect from old handler if exists
        if self.arduino_handler:
            try:
                self.arduino_handler.probability_received.disconnect(self.process_probability_data)
                self.arduino_handler.connection_status.disconnect(self.handle_connection_status)
            except:
                pass  # In case it wasn't connected

        # Set new handler
        self.arduino_handler = handler

        # Connect to the signals if handler exists
        if self.arduino_handler:
            self.arduino_handler.probability_received.connect(self.process_probability_data)
            self.arduino_handler.connection_status.connect(self.handle_connection_status)

            # Update button state based on current connection status
            self.start_inference_button.setEnabled(self.arduino_handler.running and self.using_real_hardware)

    def set_hardware_mode(self, using_real_hardware):
        """Set whether the application is using real hardware or simulated data"""
        self.using_real_hardware = using_real_hardware

        # If switching to simulated data, stop inference and disable button
        if not using_real_hardware:
            if self.inference_running:
                self.stop_inference()
            self.start_inference_button.setEnabled(False)
        else:
            # If using real hardware, enable button only if Arduino is connected
            if self.arduino_handler:
                self.start_inference_button.setEnabled(self.arduino_handler.running)
            else:
                self.start_inference_button.setEnabled(False)

    def handle_connection_status(self, connected, message):
        """Handle Arduino connection status changes"""
        # Update button state based on connection status
        if self.using_real_hardware:
            self.start_inference_button.setEnabled(connected)

        # If disconnected and inference is running, stop it
        if not connected and self.inference_running:
            self.stop_inference()
            print(f"Live inference stopped: {message}")

    def load_class_names(self):
        """Load class names from the data folder"""
        class_names = []

        try:
            # Try to detect class names from data folder structure
            data_folder = 'data'
            name_set = set()

            if os.path.exists(data_folder):
                for filename in os.listdir(data_folder):
                    if os.path.isfile(os.path.join(data_folder, filename)):
                        # Extract gesture name from filename (format: gesture-repetition.extension)
                        parts = filename.split('-')
                        if len(parts) > 1:
                            gesture_name = parts[0]
                            name_set.add(gesture_name)

            class_names = sorted(list(name_set))
            print(f"Loaded class names: {class_names}")

            # If no class names found, try to check model_info.txt
            if not class_names and os.path.exists('model_info.txt'):
                with open('model_info.txt', 'r') as f:
                    for line in f:
                        if line.startswith('Classes:'):
                            classes_str = line.strip().split(':', 1)[1].strip()
                            class_names = [cls.strip() for cls in classes_str.split(',')]
                            break
        except Exception as e:
            print(f"Error loading class names: {e}")

        # If still no class names, set a default
        if not class_names:
            class_names = ["Unknown"]

        return class_names

    def toggle_inference(self):
        """Toggle inference on/off"""
        if not self.inference_running:
            # Check if we have a valid Arduino handler
            if not self.arduino_handler or not self.arduino_handler.running:
                print("Error: Arduino not connected")
                return

            # Check if using real hardware
            if not self.using_real_hardware:
                print("Error: Simulated data cannot be used for inference")
                return

            self.start_inference()
        else:
            self.stop_inference()

    def start_inference(self):
        """Start the inference process"""
        self.inference_running = True
        self.start_inference_button.setText("Stop Live Inference")
        print("Starting live inference")

    def stop_inference(self):
        """Stop the inference process"""
        self.inference_running = False
        self.start_inference_button.setText("Start Live Inference")

        # Clear the prediction display
        self.prediction_display.clear()
        print("Stopping live inference")

    def process_probability_data(self, probabilities):
        """Process probability data from Arduino and update prediction display"""
        if not self.inference_running:
            return

        # Make sure we have valid probabilities and class names
        if not probabilities or not self.class_names:
            return

        # If there are more probabilities than classes, truncate
        if len(probabilities) > len(self.class_names):
            probabilities = probabilities[:len(self.class_names)]

        # If there are fewer probabilities than classes, pad with zeros
        elif len(probabilities) < len(self.class_names):
            probabilities.extend([0.0] * (len(self.class_names) - len(probabilities)))

        # Find the class with the highest probability
        max_index = probabilities.index(max(probabilities))
        max_prob = probabilities[max_index]

        # Get the class name and format with probability
        if max_index < len(self.class_names):
            prediction = self.class_names[max_index]
            confidence = max_prob * 100  # Convert to percentage

            # Format the prediction with confidence
            result = f"{prediction} ({confidence:.1f}%)"
            self.prediction_display.setText(result)
            print(f"Prediction: {result}")
        else:
            print(f"Error: max_index {max_index} exceeds class_names length {len(self.class_names)}")