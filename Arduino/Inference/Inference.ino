#include <LSM6DS3.h>
#include <Wire.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

const int numSamples = MAX_SAMPLE_LENGTH;
const unsigned long sampleInterval = 10; // 10ms for 100Hz sampling rate

unsigned long lastSampleTime = 0;
int samplesRead = 0; // Start from 0 to collect a full set of samples

LSM6DS3 myIMU(I2C_MODE, 0x6A);  

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 10 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));


void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (myIMU.begin() != 0) {
    Serial.println("Device error");
  } else {
    Serial.println("Device OK!");
  }

  Serial.println();

  // get the TFL representation of the model byte array  
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    
    // Replace infinite loop with error message and delay
    while (1) {
      Serial.println("Fatal error: Model schema mismatch. Please update model or code.");
      delay(5000); // Print every 5 seconds instead of hanging
    }
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  unsigned long currentTime = millis();
  
  // Sample IMU at 100Hz (every 10ms)
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read acceleration data only
    float aX = myIMU.readFloatAccelX();
    float aY = myIMU.readFloatAccelY();
    float aZ = myIMU.readFloatAccelZ();

    Serial.print("IMU:");
    Serial.print(myIMU.readFloatAccelX());
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelY());
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelZ());
    Serial.print("\n");
    
    // Store sensor data in input tensor
    tflInputTensor->data.f[samplesRead * 3 + 0] = aX;
    tflInputTensor->data.f[samplesRead * 3 + 1] = aY;
    tflInputTensor->data.f[samplesRead * 3 + 2] = aZ;

    samplesRead++;

    // Check if we have collected enough samples
    if (samplesRead == numSamples) {
      // Run inferencing
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!");
        
        // Replace infinite loop with error message and delay
        for (int i = 0; i < 5; i++) {  // Print error 5 times
          Serial.println("Error: Inference failed. Retrying data collection.");
          delay(1000);  // 1 second between messages
        }
        
        // Reset and try again instead of hanging
        samplesRead = 0;
        return;
      }
      
      // Print inference results
      Serial.print("PROB:");
      for (int i = 0; i < tflOutputTensor->dims->data[1]; i++) {
        Serial.print(tflOutputTensor->data.f[i], 6);
        Serial.print(",");
      }
      Serial.print("\n");
      
      // Reset sample counter to start collecting new data
      samplesRead = 0;
    }
  }
}