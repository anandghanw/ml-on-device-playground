# arduino_serial_handler.py
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal

class ArduinoSerialHandler(QThread):
    data_received = pyqtSignal(float, float, float)
    probability_received = pyqtSignal(list)  # New signal for probability data
    connection_status = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.running = False

    def find_xiao_device(self):
        """Find XIAO BLE Sense device connected via USB"""
        xiao_device = None
        ports = list(serial.tools.list_ports.comports())

        for port in ports:
            # Look for Seeed XIAO devices - may need to adjust these identifiers
            if "XIAO" in port.description or "Seeed" in port.description:
                xiao_device = port.device
                break

        return xiao_device

    def connect(self):
        """Connect to the XIAO BLE Sense device"""
        try:
            device_port = self.find_xiao_device()

            if not device_port:
                self.connection_status.emit(False, "XIAO BLE Sense not found. Check connection.")
                return False

            # Open serial connection - baud rate may need adjustment based on your device setup
            self.serial_port = serial.Serial(device_port, 115200, timeout=1)
            self.running = True
            self.connection_status.emit(True, f"Connected to {device_port}")
            return True

        except Exception as e:
            self.connection_status.emit(False, f"Connection error: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from the serial port"""
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception:
                # Ignore any errors during close
                pass
        self.connection_status.emit(False, "Disconnected")

    def run(self):
        """Thread main loop to read data"""
        if not self.connect():
            return

        # Main reading loop
        while self.running:
            try:
                if self.serial_port and self.serial_port.is_open:
                    line = self.serial_port.readline().decode('utf-8').strip()

                    if not line:
                        continue

                    # Handle IMU data
                    if line.startswith("IMU:"):
                        # Remove the "IMU:" prefix and parse comma-separated x,y,z values
                        try:
                            # Strip "IMU:" prefix and any potential whitespace
                            data_part = line[4:].strip()
                            x, y, z = map(float, data_part.split(','))
                            self.data_received.emit(x, y, z)
                        except ValueError:
                            # Skip malformed lines
                            pass

                    # Handle probability data
                    elif line.startswith("PROB:"):
                        try:
                            # Strip "PROB:" prefix and any potential whitespace
                            data_part = line[5:].strip()
                            # Split by comma and convert to float, ignoring the last empty item if there's an extra comma
                            probabilities = [float(p) for p in data_part.split(',') if p.strip()]
                            self.probability_received.emit(probabilities)
                        except ValueError as e:
                            print(f"Error parsing probability data: {e}")
                            pass

            except Exception as e:
                print(f"Serial reading error: {str(e)}")
                self.running = False
                break

        # Ensure disconnection when thread stops
        self.disconnect()