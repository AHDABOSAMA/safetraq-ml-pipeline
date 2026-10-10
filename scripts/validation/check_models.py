from pathlib import Path

import tensorflow as tf

# Project root: two levels above this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]

files = [
    PROJECT_ROOT / "models/training/crash_detector.tflite",
    PROJECT_ROOT / "models/deployment/accel_speed/accel_speed_crash_model.tflite",
    PROJECT_ROOT / "models/deployment/cnn_gru/cnn_gru_crash_model.tflite",
]

for file in files:
    print(f"\n{'=' * 60}")
    print(f"Model: {file.relative_to(PROJECT_ROOT)}")

    try:
        interpreter = tf.lite.Interpreter(model_path=str(file))
        interpreter.allocate_tensors()

        for tensor in interpreter.get_input_details():
            print("Input shape:", tensor["shape"].tolist())
            print("Input dtype:", tensor["dtype"])

        for tensor in interpreter.get_output_details():
            print("Output shape:", tensor["shape"].tolist())
            print("Output dtype:", tensor["dtype"])

        print("Status: Loaded successfully")

    except Exception as error:
        print("Status: Failed to initialize")
        print("Error:", error)
