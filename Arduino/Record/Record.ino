#include "LSM6DS3.h"
#include "Wire.h"

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

// For timing control
unsigned long previousMillis = 0;
float sampleRate = 100.0;
unsigned long interval;   // Will be calculated based on sample rate

void setup() {
    // put your setup code here, to run once:
    Serial.begin(115200);
    while (!Serial);
    
    // Calculate interval in ms from sample rate in Hz
    interval = (unsigned long)(1000.0 / sampleRate);
    
    //Call .begin() to configure the IMUs
    if (myIMU.begin() != 0) {
        Serial.println("Device error");
    } else {
        Serial.println("Device OK!");
        Serial.print("Sample rate set to: ");
        Serial.print(sampleRate);
        Serial.println(" Hz");
    }
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Only send data at the specified sample rate
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    
    Serial.print("IMU:");
    Serial.print(myIMU.readFloatAccelX());
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelY());
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelZ());
    Serial.print("\n");
  }
}

