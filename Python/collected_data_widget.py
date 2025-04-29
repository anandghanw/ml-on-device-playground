# collected_data_widget.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QGroupBox, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal
import numpy as np
import os
import glob
from app_styles import AppStyles

class CollectedDataWidget(QGroupBox):
    sample_selected = pyqtSignal(object, str)
    samples_cleared = pyqtSignal()

    def __init__(self, sample_data_widget):
        super().__init__("Collected Data")
        # Apply heading style
        self.setStyleSheet(AppStyles.get_group_box_style())

        self.sample_data_widget = sample_data_widget

        layout = QVBoxLayout(self)
        layout.setSpacing(AppStyles.WIDGET_SPACING)

        # List of collected samples
        self.sample_list = QListWidget()
        self.sample_list.setStyleSheet(AppStyles.get_list_widget_style())
        self.sample_list.currentRowChanged.connect(self.on_sample_selected)
        layout.addWidget(self.sample_list)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(AppStyles.WIDGET_SPACING)

        # Delete sample button
        self.delete_sample_button = QPushButton("Delete Sample")
        self.delete_sample_button.setStyleSheet(AppStyles.get_button_style())
        self.delete_sample_button.clicked.connect(self.delete_sample)
        buttons_layout.addWidget(self.delete_sample_button)

        # Delete all button
        self.delete_all_button = QPushButton("Delete All")
        self.delete_all_button.setStyleSheet(AppStyles.get_button_style())
        self.delete_all_button.clicked.connect(self.delete_all_samples)
        buttons_layout.addWidget(self.delete_all_button)

        layout.addLayout(buttons_layout)

        # Dictionary to store the samples
        self.samples = {}

        # Ensure data directory exists and load saved samples
        self.data_dir = "data"
        self.ensure_data_directory()
        self.load_saved_samples()

    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def load_saved_samples(self):
        """Load all saved samples from the data directory"""
        # Get all .txt files in the data directory
        sample_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        for file_path in sample_files:
            try:
                # Get label from filename (remove path and extension)
                label = os.path.splitext(os.path.basename(file_path))[0]

                # Load data from file
                data = np.loadtxt(file_path)

                # Skip files with invalid data
                if data.size == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue

                # Make sure data is 2D (reshape single samples)
                if len(data.shape) == 1:
                    # If only one row, reshape to 2D with X, Y, Z columns
                    if data.size % 3 == 0:
                        data = data.reshape(-1, 3)
                    else:
                        print(f"Skipping file with invalid dimensions: {file_path}")
                        continue

                # Add to samples
                self.samples[label] = data
                self.sample_list.addItem(label)
                print(f"Loaded sample: {label}")

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

        # Select first item if available
        if self.sample_list.count() > 0:
            self.sample_list.setCurrentRow(0)

    def add_sample(self, data, label):
        """Add a new sample to the collection and save to disk"""
        self.samples[label] = data
        self.sample_list.addItem(label)

        # Save to disk
        self.save_sample(data, label)

        # Select the newly added sample
        self.sample_list.setCurrentRow(self.sample_list.count() - 1)

    def save_sample(self, data, label):
        """Save sample data to a text file"""
        # Ensure data directory exists
        self.ensure_data_directory()

        # Create file path
        file_path = os.path.join(self.data_dir, f"{label}.txt")

        # Save data to file
        try:
            np.savetxt(file_path, data)
            print(f"Saved sample to {file_path}")
        except Exception as e:
            print(f"Error saving sample {label}: {e}")

    def on_sample_selected(self, row):
        """Handle sample selection"""
        if row >= 0:
            label = self.sample_list.item(row).text()
            data = self.samples[label]
            self.sample_data_widget.display_sample(data, label)
            self.sample_selected.emit(data, label)

    def delete_sample(self):
        """Delete the selected sample from collection and disk"""
        current_row = self.sample_list.currentRow()
        if current_row >= 0:
            label = self.sample_list.item(current_row).text()
            self.sample_list.takeItem(current_row)

            # Remove from dictionary
            del self.samples[label]

            # Delete file from disk
            file_path = os.path.join(self.data_dir, f"{label}.txt")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

            # Update sample view if needed
            if self.sample_list.count() > 0:
                self.sample_list.setCurrentRow(max(0, current_row - 1))

    def delete_all_samples(self):
        """Delete all samples from collection and disk"""
        # Get all labels before clearing
        labels = [self.sample_list.item(i).text() for i in range(self.sample_list.count())]

        # Clear UI list
        self.sample_list.clear()

        # Clear samples dictionary
        self.samples.clear()

        # Delete files from disk
        for label in labels:
            file_path = os.path.join(self.data_dir, f"{label}.txt")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        self.samples_cleared.emit()

    def get_all_samples(self):
        """Return all samples"""
        return self.samples