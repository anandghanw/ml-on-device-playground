# main_window.py
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
from real_time_graph import RealTimeGraph
from record_data_widget import RecordDataWidget
from sample_data_widget import SampleDataWidget
from collected_data_widget import CollectedDataWidget
from training_widget import TrainingWidget
from inference_widget import InferenceWidget
from app_styles import AppStyles

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("On-Device Machine Learning Playground")
        self.resize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(AppStyles.WIDGET_SPACING)

        # Create widgets first
        self.real_time_graph = RealTimeGraph()
        self.sample_data_widget = SampleDataWidget()
        self.record_data_widget = RecordDataWidget(self.real_time_graph)
        self.collected_data_widget = CollectedDataWidget(self.sample_data_widget)
        self.training_widget = TrainingWidget(self.collected_data_widget)
        self.inference_widget = InferenceWidget()

        # Set up the connections between widgets that need to share serial data
        self.setup_connections()

        # Connect record widget to collected data widget
        self.record_data_widget.sample_recorded.connect(
            lambda data, label: self.collected_data_widget.add_sample(data, label)
        )

        # Left column: Record Data and Collected Data (full window height)
        left_column = QVBoxLayout()
        left_column.setSpacing(AppStyles.WIDGET_SPACING)
        left_column.addWidget(self.record_data_widget, 1)  # 1 stretch factor (1/3 height)
        left_column.addWidget(self.collected_data_widget, 3)  # 2 stretch factor (2/3 height)
        main_layout.addLayout(left_column, 1)  # 1 stretch factor of window width

        # Right content area
        right_content = QVBoxLayout()
        right_content.setSpacing(AppStyles.WIDGET_SPACING)

        # Real-time IMU data graph
        right_content.addWidget(self.real_time_graph, 1)  # 1 stretch factor

        # Lower right section with sample, training, and inference
        lower_right = QHBoxLayout()
        lower_right.setSpacing(AppStyles.WIDGET_SPACING)

        # Sample Data widget
        lower_right.addWidget(self.sample_data_widget, 2)  # 2 stretch factor

        # Training and Inference column
        training_inference_column = QVBoxLayout()
        training_inference_column.setSpacing(AppStyles.WIDGET_SPACING)
        training_inference_column.addWidget(self.training_widget)
        training_inference_column.addWidget(self.inference_widget)
        lower_right.addLayout(training_inference_column, 1)  # 1 stretch factor

        # Add lower right section to right content
        right_content.addLayout(lower_right, 1)  # 1 stretch factor

        # Add right content to main layout
        main_layout.addLayout(right_content, 3)  # 3 stretch factor

    def setup_connections(self):
        """Set up connections between widgets that need to share serial data"""
        # Share the Arduino serial handler from real_time_graph with inference_widget
        if hasattr(self.real_time_graph, 'serial_handler'):
            # Set the Arduino handler in the inference widget
            self.inference_widget.set_arduino_handler(self.real_time_graph.serial_handler)

            # Connect data source change signal from real_time_graph to inference_widget
            self.real_time_graph.data_source_changed.connect(self.inference_widget.set_hardware_mode)

            # Set initial hardware mode
            self.inference_widget.set_hardware_mode(
                self.real_time_graph.data_source == self.real_time_graph.USE_ARDUINO_DATA
            )