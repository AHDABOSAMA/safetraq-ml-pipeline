# SafeTraq Model Cards

This document records the known properties and validation status of the model artifacts included in this repository. It distinguishes observations verified locally from details that still require confirmation.

## 1. Training model: `crash_detector.keras`

**Location:** `models/training/crash_detector.keras`

### Purpose

A CNN-LSTM binary classifier trained by the original training script to classify fixed-length sensor windows.

### Verified interface

- Input shape: `(batch_size, 500, 6)`
- Output shape: `(batch_size, 1)`
- Input features: accelerometer XYZ and gyroscope XYZ
- The saved Keras model loads successfully in the current local environment.

### Training and evaluation caveat

The existing training script uses the test arrays as validation data for early stopping. The test set therefore influences model selection and cannot be considered an untouched final evaluation set.

### Limitations

- Real-world performance has not been established by the checks documented here.
- The suitability of the original split for related or overlapping recordings needs review.
- Metrics must be reproduced and interpreted alongside the evaluation procedure.

## 2. Training export: `crash_detector.tflite`

**Location:** `models/training/crash_detector.tflite`

### Intended role

A TensorFlow Lite export of the training model.

### Validation status

The standard TensorFlow Lite interpreter in the current environment fails to initialize this file because the model contains a Select TensorFlow operation, `FlexTensorListReserve`, that requires additional runtime support.

This is a compatibility finding, not proof that the file is corrupted.

### Next steps

- Investigate conversion settings and model operations.
- Determine whether the target runtime supports the required operations.
- Verify inference outputs against the original Keras model before describing the export as deployment-ready.

## 3. Deployment model: acceleration and speed

**Location:** `models/deployment/accel_speed/accel_speed_crash_model.tflite`

### Verified interface

- Input shape: `(1, 60, 4)`
- Output shape: `(1, 1)`
- Input dtype and output dtype: `float32`
- The model initializes successfully with the local TensorFlow Lite interpreter.

### Feature order

The accompanying `scaler_params.json` lists:

1. `accel_x`
2. `accel_y`
3. `accel_z`
4. `speed`

### Remaining verification

The feature order, units, scaler provenance, and preprocessing behavior must be checked against the code that originally trained and deployed this model. Successful initialization does not establish predictive accuracy.

## 4. Deployment model: CNN-GRU

**Location:** `models/deployment/cnn_gru/cnn_gru_crash_model.tflite`

### Verified interface

- Input shape: `(1, 60, 7)`
- Output shape: `(1, 1)`
- Input dtype and output dtype: `float32`
- The model initializes successfully with the local TensorFlow Lite interpreter.

### Feature order

The accompanying `scaler_params.json` lists:

1. `accel_x`
2. `accel_y`
3. `accel_z`
4. `gyro_x`
5. `gyro_y`
6. `gyro_z`
7. `speed`

### Remaining verification

The scaler values, feature units, training provenance, and inference preprocessing need confirmation. The model's accuracy and real-world reliability have not been established by interface inspection.

## 5. Compatibility summary

The saved training arrays use 500 timesteps and six features. The two deployment models expect 60 timesteps with four and seven features, respectively.

These interfaces are not interchangeable. Each model requires the preprocessing pipeline and feature conventions that match its own training process.

## 6. Safety and responsible use

These artifacts are experimental. They are not certified vehicle-safety systems and should not be relied upon as the sole means of crash detection or emergency notification.

Real-world use would require representative labeled data, independent evaluation, false-alarm analysis, and validation across devices and driving conditions.

## Validation method

The local inspection script is available at `scripts/validation/check_models.py`. It checks TensorFlow Lite initialization and reports input/output tensor shapes and dtypes. It does not measure classification accuracy or establish deployment readiness.