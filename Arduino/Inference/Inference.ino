#include <LSM6DS3.h>
#include <Wire.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

const int numSamples = MAX_SAMPLE_LENGTH;
float sampleRate = 100.0;
unsigned long sampleInterval;   // Will be calculated based on sample rate

// Sliding window parameters
int overlapPercentage = 50;     // Adjustable: percentage of overlap (0-90)
int slidingStep;               // Will be calculated based on overlap percentage

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
constexpr int tensorArenaSize = 100 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));


void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Calculate interval in ms from sample rate in Hz
  sampleInterval = (unsigned long)(1000.0 / sampleRate);
  
  // Calculate sliding step based on overlap percentage
  slidingStep = numSamples * (100 - overlapPercentage) / 100;
  if (slidingStep < 1) slidingStep = 1; // Ensure at least 1 step
  
  Serial.print("Sliding window configuration: ");
  Serial.print(numSamples);
  Serial.print(" samples with ");
  Serial.print(overlapPercentage);
  Serial.print("% overlap (");
  Serial.print(slidingStep);
  Serial.println(" samples per slide)");

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
  
  // Sample IMU at defined sample rate
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read acceleration data
    float aX = myIMU.readFloatAccelX();
    float aY = myIMU.readFloatAccelY();
    float aZ = myIMU.readFloatAccelZ();

    Serial.print("IMU:");
    Serial.print(aX);
    Serial.print(",");
    Serial.print(aY);
    Serial.print(",");
    Serial.print(aZ);
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
        
        // Error handling
        for (int i = 0; i < 3; i++) {
          Serial.println("Error: Inference failed. Retrying data collection.");
          delay(500);
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
      
      // Slide window by moving data
      if (slidingStep < numSamples) {
        // Shift data in the buffer by the sliding step
        for (int i = 0; i < (numSamples - slidingStep) * 3; i++) {
          tflInputTensor->data.f[i] = tflInputTensor->data.f[i + slidingStep * 3];
        }
        samplesRead = numSamples - slidingStep;
      } else {
        // No overlap, start fresh
        samplesRead = 0;
      }
    }
  }
}