# training_widget.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGroupBox, QSpacerItem,
                             QSizePolicy, QCheckBox, QMessageBox, QDialog, QLabel, QGridLayout,
                             QApplication)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from app_styles import AppStyles
import seaborn as sns
from io import BytesIO
import datetime


class ConfusionMatrixDialog(QDialog):
    def __init__(self, cm, class_names, accuracy, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Results")
        self.setMinimumSize(500, 500)

        # Match the dark gray background from the application
        self.setStyleSheet("background-color: #2E2E2E;")

        layout = QVBoxLayout(self)

        # Add accuracy label with green accent color matching the app
        accuracy_label = QLabel(f"Model Accuracy: {accuracy:.2f}%")
        accuracy_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(accuracy_label)

        # Create confusion matrix plot with matching theme
        figure = Figure(figsize=(8, 6))
        figure.patch.set_facecolor('#2E2E2E')  # Dark gray background for the figure
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        ax = figure.add_subplot(111)
        ax.set_facecolor('#2E2E2E')  # Dark gray background for the plot area

        # Use a colormap similar to the application's green-blue-red scheme
        heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                              xticklabels=class_names, yticklabels=class_names,
                              ax=ax, cbar_kws={"shrink": 0.8})

        # Set the annotation text color to white for better visibility
        for text in ax.texts:
            text.set_color('white')

        # Style the colorbar to match theme
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Set text colors to white for better visibility on dark background
        ax.set_xlabel('Predicted labels', color='white')
        ax.set_ylabel('True labels', color='white')
        ax.set_title('Confusion Matrix', color='white', fontsize=14, fontweight='bold')

        # Set tick colors to white
        ax.tick_params(colors='white')

        # Set axes and tick label colors to white
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')

        # Add gridlines in gray to match the application style
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        # Add a tight layout to ensure everything fits
        figure.tight_layout()

        # Add close button with style matching the application
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        # Gray button with white text to match application style
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """)
        layout.addWidget(close_button)


class TrainingWorker(QObject):
    """Worker class that performs the model training in a separate thread"""

    # Define signals
    update_status = pyqtSignal(str)
    training_complete = pyqtSignal(object, object, object, float, list)  # model, cm, class_names, accuracy, test_data
    training_error = pyqtSignal(str)
    epoch_progress = pyqtSignal(int, int)  # current_epoch, total_epochs

    def __init__(self, X, y, class_names, model_type, input_shape, num_classes):
        super().__init__()
        self.X = X
        self.y = y
        self.class_names = class_names
        self.model_type = model_type  # "nn" or "cnn"
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self):
        """Create a model based on the selected architecture"""
        if self.model_type == "nn":
            # Simple Neural Network
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_shape,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])

        else:  # CNN model
            # 1D CNN
            model = tf.keras.Sequential([
                tf.keras.layers.Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def balance_classes(self, X, y):
        """Balance classes by upsampling minority classes"""
        # Count samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)

        # Create lists for balanced dataset
        balanced_X = []
        balanced_y = []

        # Upsample each class to match the largest class
        for cls in unique_classes:
            cls_X = X[y == cls]
            cls_y = y[y == cls]

            if len(cls_X) < max_count:
                # Random upsampling
                cls_X_resampled, cls_y_resampled = resample(
                    cls_X, cls_y,
                    n_samples=max_count,
                    random_state=42,
                    replace=True
                )
                balanced_X.append(cls_X_resampled)
                balanced_y.append(cls_y_resampled)
            else:
                balanced_X.append(cls_X)
                balanced_y.append(cls_y)

        # Concatenate all balanced classes
        balanced_X = np.vstack(balanced_X)
        balanced_y = np.concatenate(balanced_y)

        return balanced_X, balanced_y

    def run(self):
        """Main worker function that runs the training process"""
        try:
            # Update status
            self.update_status.emit("Status: Balancing classes...")

            # Balance classes
            X_balanced, y_balanced = self.balance_classes(self.X, self.y)
            print(f"Balanced dataset: {len(X_balanced)} samples")

            # Check if we have enough samples for stratified splitting
            min_samples_per_class = np.min([np.sum(y_balanced == i) for i in range(len(self.class_names))])
            print(f"Minimum samples per class: {min_samples_per_class}")

            # Update status
            self.update_status.emit("Status: Splitting data into train/test sets...")

            # For very small datasets (like 3 samples per class), we need a special approach
            if min_samples_per_class <= 3:
                print("Very small dataset detected. Using manual splitting...")

                # Create indices for each class
                indices_by_class = {}
                for cls in range(len(self.class_names)):
                    indices_by_class[cls] = np.where(y_balanced == cls)[0]

                # Take 1 sample from each class for testing
                test_indices = []
                for cls, indices in indices_by_class.items():
                    # Take the last sample of each class for testing
                    test_indices.append(indices[-1])

                # All other samples for training
                train_indices = np.setdiff1d(np.arange(len(y_balanced)), test_indices)

                # Create train/test sets
                X_train = X_balanced[train_indices]
                y_train = y_balanced[train_indices]
                X_test = X_balanced[test_indices]
                y_test = y_balanced[test_indices]

                print(f"Manual split complete: train={len(X_train)}, test={len(X_test)}")

            # For datasets with more samples, use standard approach
            elif min_samples_per_class >= 5:
                test_size = min(0.2, 1.0 - (3.0 / min_samples_per_class))
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=test_size, random_state=42, stratify=y_balanced
                )
                print(f"Split data with test_size={test_size}: train={len(X_train)}, test={len(X_test)}")
            else:
                # For intermediate cases (4 samples per class)
                test_size = 0.25  # 1 out of 4
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=test_size, random_state=42, stratify=y_balanced
                )
                print(f"Split data with fixed test_size={test_size}: train={len(X_train)}, test={len(X_test)}")

            # Update status
            self.update_status.emit("Status: Creating model architecture...")

            # Create model
            model = self.create_model()

            # Adjust batch size based on data size - for tiny datasets, use batch size of 1
            batch_size = max(1, min(32, len(X_train) // 2))

            # Determine epochs - more epochs for smaller datasets
            epochs = min(50, max(30, 200 // len(X_train)))

            # For very small datasets, don't use validation split
            validation_split = 0.0 if len(X_train) < 10 else 0.2

            print(f"Training with batch_size={batch_size}, epochs={epochs}, validation_split={validation_split}")

            # Custom callback to update UI with progress
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, worker):
                    super().__init__()
                    self.worker = worker

                def on_epoch_begin(self, epoch, logs=None):
                    self.worker.epoch_progress.emit(epoch + 1, epochs)

            # Train the model
            # Neural network training
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                callbacks=[ProgressCallback(self)]
            )

            # Update status
            self.update_status.emit("Status: Evaluating model...")

            # Evaluate the model
            # Neural network prediction
            y_pred = np.argmax(model.predict(X_test), axis=1)

            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred) * 100

            print(f"Model evaluation complete. Accuracy: {accuracy:.2f}%")

            # Signal completion with results
            self.training_complete.emit(model, cm, self.class_names, accuracy, [X_test, y_test])

        except Exception as e:
            self.training_error.emit(str(e))
            import traceback
            traceback.print_exc()

class TrainingWidget(QGroupBox):
    def __init__(self, collected_data_widget):
        super().__init__("Training")
        # Apply heading style
        self.setStyleSheet(AppStyles.get_group_box_style())

        self.collected_data_widget = collected_data_widget
        self.model = None
        self.thread = None
        self.worker = None

        layout = QVBoxLayout(self)
        layout.setSpacing(AppStyles.WIDGET_SPACING)

        # Status label with lighter color
        self.status_label = QLabel("Status: Ready to train")
        self.status_label.setStyleSheet("font-weight: bold; color: #666; background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.status_label)

        # Model architecture selection
        self.model_nn_checkbox = QCheckBox("Model NN (Simple Neural Network)")
        self.model_1dcnn_checkbox = QCheckBox("Model 1D CNN (1D Convolutional Neural Network)")

        # Set the first option as default
        self.model_nn_checkbox.setChecked(True)

        # Make checkboxes exclusive (only one can be selected)
        self.model_nn_checkbox.clicked.connect(lambda: self.update_checkboxes(self.model_nn_checkbox))
        self.model_1dcnn_checkbox.clicked.connect(lambda: self.update_checkboxes(self.model_1dcnn_checkbox))

        # Add checkboxes to layout
        layout.addWidget(self.model_nn_checkbox)
        layout.addWidget(self.model_1dcnn_checkbox)

        # Train model button
        self.train_button = QPushButton("Train Model and Generate TFLite")
        self.train_button.setStyleSheet(AppStyles.get_button_style())
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

    def update_checkboxes(self, checkbox):
        """Ensure only one checkbox is selected at a time"""
        if checkbox.isChecked():
            if checkbox != self.model_nn_checkbox:
                self.model_nn_checkbox.setChecked(False)
            if checkbox != self.model_1dcnn_checkbox:
                self.model_1dcnn_checkbox.setChecked(False)
            checkbox.setChecked(True)
        else:
            # Don't allow unchecking the last checked box
            checkbox.setChecked(True)

    def load_and_preprocess_data(self):
        """Load data from the data folder and preprocess it"""
        data_folder = 'data'
        X = []
        y = []
        class_names = set()
        max_sample_length = 0

        # First pass: determine the maximum length of samples and collect class names
        for filename in os.listdir(data_folder):
            if os.path.isfile(os.path.join(data_folder, filename)):
                # Extract gesture name from filename (format: gesture-repetition.extension)
                parts = filename.split('-')
                if len(parts) > 1:
                    gesture_name = parts[0]
                    class_names.add(gesture_name)

                    # Check sample length
                    file_path = os.path.join(data_folder, filename)
                    try:
                        sample_data = np.loadtxt(file_path, dtype=np.float32)
                        if isinstance(sample_data, np.ndarray):
                            if sample_data.ndim == 1:
                                # Single feature vector
                                max_sample_length = max(max_sample_length, len(sample_data))
                            elif sample_data.ndim == 2:
                                # Multiple feature vectors
                                max_sample_length = max(max_sample_length, sample_data.size)
                    except Exception as e:
                        print(f"Error checking file {filename}: {e}")

        if max_sample_length == 0:
            raise ValueError("No valid data files found or all files are empty")

        print(f"Maximum sample length: {max_sample_length}")

        # Second pass: load and pad all samples to the same length
        for filename in os.listdir(data_folder):
            if os.path.isfile(os.path.join(data_folder, filename)):
                parts = filename.split('-')
                if len(parts) > 1:
                    gesture_name = parts[0]
                    file_path = os.path.join(data_folder, filename)

                    try:
                        sample_data = np.loadtxt(file_path, dtype=np.float32)

                        # Handle different dimensionalities
                        if sample_data.ndim == 1:
                            # Pad or truncate to max_sample_length
                            padded_data = np.zeros(max_sample_length, dtype=np.float32)
                            padded_data[:min(len(sample_data), max_sample_length)] = sample_data[:min(len(sample_data), max_sample_length)]
                        else:
                            # Flatten and pad/truncate
                            flattened = sample_data.flatten()
                            padded_data = np.zeros(max_sample_length, dtype=np.float32)
                            padded_data[:min(len(flattened), max_sample_length)] = flattened[:min(len(flattened), max_sample_length)]

                        X.append(padded_data)
                        y.append(gesture_name)
                    except Exception as e:
                        print(f"Error loading file {filename}: {e}")

        if not X:
            raise ValueError("No valid data could be loaded")

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)

        # Convert class names to numerical indices
        class_names = sorted(list(class_names))
        class_indices = {name: i for i, name in enumerate(class_names)}
        y_numeric = np.array([class_indices[name] for name in y])

        print(f"Loaded data shape: {X.shape}, Labels shape: {y_numeric.shape}")
        print(f"Classes: {class_names}")

        return X, y_numeric, class_names

    def train_model(self):
        """Train the model using collected data and automatically generate TFLite model"""
        self.train_button.setEnabled(False)
        try:
            # Update status
            self.status_label.setText("Status: Loading and preprocessing data...")
            QApplication.processEvents()  # Force UI update

            # Load and preprocess data
            X, y, class_names = self.load_and_preprocess_data()
            if len(X) == 0:
                self.status_label.setText("Status: Error: No samples found")
                QMessageBox.warning(self, "Warning", "No samples found in the data folder!")
                self.train_button.setEnabled(True)
                return

            print(f"Loaded {len(X)} samples across {len(class_names)} classes")

            # Determine which model type is selected
            model_type = "nn"
            if self.model_1dcnn_checkbox.isChecked():
                model_type = "cnn"

            input_shape = X.shape[1]
            num_classes = len(class_names)

            # Create thread and worker
            self.thread = QThread()
            self.worker = TrainingWorker(X, y, class_names, model_type, input_shape, num_classes)
            self.worker.moveToThread(self.thread)

            # Connect signals
            self.thread.started.connect(self.worker.run)
            self.worker.update_status.connect(self.status_label.setText)
            self.worker.epoch_progress.connect(self.update_epoch_progress)
            self.worker.training_complete.connect(self.on_training_complete)
            self.worker.training_error.connect(self.on_training_error)

            # Start the thread
            self.thread.start()

        except Exception as e:
            self.status_label.setText(f"Status: Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error during training setup: {str(e)}")
            print(f"Training setup error: {e}")
            import traceback
            traceback.print_exc()
            self.train_button.setEnabled(True)

    def update_epoch_progress(self, current_epoch, total_epochs):
        """Update status with current epoch progress"""
        self.status_label.setText(f"Status: Training epoch {current_epoch}/{total_epochs}...")

    def on_training_complete(self, model, cm, class_names, accuracy, test_data):
        """Handle training completion"""
        self.model = model

        # Show confusion matrix
        dialog = ConfusionMatrixDialog(cm, class_names, accuracy, self)

        # Update status
        self.status_label.setText("Status: Generating TFLite model...")
        QApplication.processEvents()

        # Generate TFLite model
        self.generate_tflite()

        # Update status with final result
        self.status_label.setText(f"Status: Training complete! Accuracy: {accuracy:.2f}%")

        # Clean up thread
        self.thread.quit()
        self.thread.wait()

        # Show evaluation results
        dialog.exec_()

        # Re-enable the train button
        self.train_button.setEnabled(True)

    def on_training_error(self, error_message):
        """Handle training errors"""
        self.status_label.setText(f"Status: Error: {error_message}")
        QMessageBox.critical(self, "Error", f"Error during training: {error_message}")

        # Clean up thread
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # Re-enable the train button
        self.train_button.setEnabled(True)

    def generate_tflite(self):
        """Generate TFLite model compatible with TFLite Micro"""
        if self.model is None:
            self.status_label.setText("Status: Error: No trained model available")
            return False

        try:
            # Convert TensorFlow model to TFLite format
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

            # Convert the model
            print("Starting TFLite conversion...")
            tflite_model = converter.convert()
            print("TFLite conversion completed")

            # Save to file
            tflite_path = 'model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # Rest of the code remains the same...
            # Convert TFLite model to C array for Arduino
            self.status_label.setText("Status: Converting model to C array for Arduino...")
            QApplication.processEvents()

            # Create a model info file with metadata
            model_type = "Neural Network"
            if self.model_1dcnn_checkbox.isChecked():
                model_type = "1D CNN"

            # Get file size
            model_size_kb = os.path.getsize(tflite_path) / 1024

            # Get max sample length from current data
            X, _, class_names = self.load_and_preprocess_data()
            max_sample_length = X.shape[1]

            # Save class names in exactly the order used during training
            class_names_str = ",".join(class_names)

            # Create C header file
            c_header_path = 'model_data.h'
            with open(c_header_path, 'w') as f:
                # Write header guard
                f.write("#ifndef MODEL_DATA_H\n")
                f.write("#define MODEL_DATA_H\n\n")

                # Write array declaration
                f.write("// This file is auto-generated. Do not edit.\n")
                f.write(f"// Model Type: {model_type}\n")
                f.write(f"// Model Size: {model_size_kb:.2f} KB\n")
                f.write(f"// Max Sample Length: {max_sample_length}\n")
                f.write(f"// Classes: {class_names_str}\n")
                f.write(f"// Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Define max sample length constant
                f.write(f"#define MAX_SAMPLE_LENGTH {max_sample_length}\n\n")

                # Define class names as C array if wanted
                f.write(f"#define NUM_CLASSES {len(class_names)}\n")
                f.write("const char* GESTURES[] = {")
                for i, name in enumerate(class_names):
                    f.write(f'"{name}"')
                    if i < len(class_names) - 1:
                        f.write(", ")
                f.write("};\n\n")

                f.write("const unsigned char model[] = {\n")

                # Write model data as hex
                line = "  "
                for i, byte in enumerate(tflite_model):
                    if isinstance(byte, int):
                        byte_val = byte
                    else:
                        # Handle Python 3 bytes type
                        byte_val = byte
                    line += f"0x{byte_val:02x}, "
                    if (i + 1) % 12 == 0:
                        f.write(line + '\n')
                        line = "  "

                # Write any remaining bytes
                if line != "  ":
                    f.write(line + '\n')

                # Close the array
                f.write("};\n\n")

                # Write array length
                f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n\n")

                # Close header guard
                f.write("#endif // MODEL_DATA_H\n")

            # Write to separate model_info.txt file as well for easier parsing
            with open('model_info.txt', 'w') as f:
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Model Size: {model_size_kb:.2f} KB\n")
                f.write(f"Max Sample Length: {max_sample_length}\n")
                f.write(f"Classes: {class_names_str}\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Update status with success message instead of showing popup
            print(f"TFLite model successfully generated and saved to {tflite_path}")
            print(f"Model size: {model_size_kb:.2f} KB")
            print(f"Max sample length: {max_sample_length}")
            print(f"Classes: {class_names_str}")

        except Exception as e:
            self.status_label.setText(f"Status: Error generating TFLite: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error generating TFLite model: {str(e)}")
            print(f"TFLite conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False