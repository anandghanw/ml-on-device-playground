# sample_data_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np
from app_styles import AppStyles


class SampleDataWidget(QGroupBox):
    def __init__(self):
        super().__init__("Selected Sample")  # More descriptive title
        # Apply heading style
        self.setStyleSheet(AppStyles.get_group_box_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(AppStyles.WIDGET_SPACING)

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

        # Current sample data
        self.current_sample = None
        self.current_label = None

    def display_sample(self, data, label):
        """Display a sample in the plot"""
        self.current_sample = data
        self.current_label = label

        # Create time array based on data length
        time_data = np.linspace(0, len(data)/50, len(data))  # Assuming 50Hz

        # Update plot
        self.x_curve.setData(time_data, data[:, 0])
        self.y_curve.setData(time_data, data[:, 1])
        self.z_curve.setData(time_data, data[:, 2])

        # Update plot title with white text
        title_html = f"<span style='{AppStyles.GRAPH_TITLE_STYLE}'>Sample: {label}</span>"
        self.plot_widget.setTitle(title_html)