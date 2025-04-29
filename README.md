# On-Device ML Playground

This project is a tool for beginners to explore on-device ML.

## Overview

This project provides a complete environment for recording sensor data, training machine learning models, and deploying them to microcontroller devices. It's designed to help beginners understand the full workflow of on-device machine learning without requiring extensive background knowledge.

## Features

- Simple data collection interface
- Integrated model training pipeline
- Real-time visualization and inference

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
    - [Python Environment Setup](#python-environment-setup)
    - [Arduino IDE Setup](#arduino-ide-setup)
- [Getting Started](#getting-started)
    - [Data Recording](#data-recording)
    - [Model Training](#model-training)
    - [Inference](#inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- [XIAO BLE Sense](https://www.seeedstudio.com/Seeed-XIAO-BLE-Sense-nRF52840-p-5253.html) board
- Arduino IDE (version 1.8.x or newer)
- Python 3.9 or newer
- USB cable for connecting the board to your computer

## Installation

### Clone this repository:
   ```bash
   git clone https://github.com/anandghanw/ml-on-device-playground.git
   cd on-device-ml-playground
   ```
   
### Python Environment Setup

1. Make the setup script executable:
   ```bash
   chmod +x setup_python_env.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_python_env.sh
   ```

   This will create a virtual environment and install all required Python packages.

### Arduino IDE Setup

1. Install the Arduino IDE from [arduino.cc](https://www.arduino.cc/en/software)

2. Add Seeed XIAO nRF52840 Sense board support:
    - Open Arduino IDE
    - Go to `File > Preferences`
    - Add this URL to "Additional Boards Manager URLs":
      ```
      https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
      ```
    - Go to `Tools > Board > Boards Manager...`
    - Search for "Seeed nRF52 Boards" and install

3. Install the required libraries:
    - Open the Arduino IDE
    - Go to `Sketch > Include Library > Manage Libraries...`
    - Search for and install "Seeed Arduino LSM6DS3"

4. Install the TensorFlow Lite Micro Arduino Examples library:
    - In the Arduino IDE, go to `Sketch > Include Library > Add .ZIP Library...`
    - Select the `tflite-micro-arduino-examples-main.zip` file from the root directory
    - This library has been downloaded from https://github.com/lakshanthad/tflite-micro-arduino-examples

## Getting Started

### Data Recording

1. Connect your XIAO BLE Sense board to your computer

2. Open the `Arduino/Record/Record.ino` sketch in Arduino IDE

3. Upload the sketch to your XIAO BLE Sense board

4. Open a terminal, navigate to the project root directory, and activate the Python virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. Run the playground tool:
   ```bash
   cd Python
   python main.py
   ```

6. Follow the UI to collect training data for your gestures

### Model Training

After collecting sufficient data:

1. In the playground tool, select the "Train Model" option

2. The tool will automatically train and evaluate the model using your recorded data

3. Once complete, the model will be converted to TensorFlow Lite format and prepared for deployment

### Inference

1. After training a new model, open the `Arduino/Inference/Inference.ino` sketch in Arduino IDE

2. Upload the sketch to your XIAO BLE Sense board

   **Important:** You must re-upload `Inference.ino` every time you train a new model to update the model on the device.

3. If closed, re-run the playground tool to visualize the real-time inference results:
   ```bash
   cd Python
   python main.py
   ```

4. Select the "Inference" option to see live predictions from your model

## Project Structure

```
on-device-ml-playground/
├── Arduino/
│   ├── Record/            # Data recording sketch
│   │   └── Record.ino
│   └── Inference/         # Model inference sketch
│       └── Inference.ino
├── Python/
│   ├── data/              # Recorded sensor data
│   ├── models/            # Trained models
│   └── main.py            # Main application
├── setup_python_env.sh    # Python environment setup script
└── tflite-micro-arduino-examples-main.zip  # TFLite Micro library
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.