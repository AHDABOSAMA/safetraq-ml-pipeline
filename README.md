# SafeTraq ML Pipeline

An ML engineering project focused on detecting potential vehicle crashes using smartphone motion-sensor data.

This repository organizes and validates the machine-learning components of SafeTraq, a team graduation project involving smartphone-based crash detection. It provides a separate, ML-focused workspace for inspecting datasets, preprocessing sensor signals, evaluating models, and documenting deployment requirements.

> **Current status:** The repository contains synthetic sensor data, preprocessed arrays, original training scripts, and four model files. Initial structural checks have been completed. Full inference validation, reproducible evaluation, and training-to-deployment consistency checks remain in progress.

## Problem

Accelerometer and gyroscope readings can capture sudden changes in vehicle motion. However, hard braking, potholes, and other ordinary driving events may resemble crashes. A useful crash-detection system must evaluate both crash detection and false alarms.

## Repository contents

- **Synthetic sensor data:** 300 CSV files across six generated scenarios.
- **Processed data:** Training and test arrays, labels, and normalization parameters.
- **Original training scripts:** Data generation, preprocessing, model training, and evaluation.
- **Training artifacts:** A Keras model and a TensorFlow Lite export.
- **Deployment artifacts:** Two TensorFlow Lite models with associated scaler parameter files.
- **Validation code:** A Python data-validation module and automated tests.

The data and model artifacts are included in this repository. Their presence does not, by itself, establish real-world accuracy or deployment readiness.

## Repository structure

```text
safetraq-ml-pipeline/
├── data/
│   ├── raw/
│   │   ├── crashes/
│   │   └── normal_driving/
│   └── processed/
├── docs/
├── models/
│   ├── training/
│   └── deployment/
├── reports/
├── scripts/
│   └── original/
├── src/
│   └── safetraq/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Data and preprocessing

The included raw CSV files were generated synthetically. They cover crash scenarios such as frontal impacts, side impacts, and rollovers, alongside normal-driving scenarios such as highway driving, hard braking, and potholes.

The existing preprocessing script uses six features:

- Accelerometer X, Y, and Z.
- Gyroscope X, Y, and Z.

The saved arrays have these shapes:

| Artifact | Shape |
|---|---|
| `X_train.npy` | `(720, 500, 6)` |
| `X_test.npy` | `(60, 500, 6)` |
| `y_train.npy` | `(720,)` |
| `y_test.npy` | `(60,)` |
| `norm_mean.npy` | `(6,)` |
| `norm_std.npy` | `(6,)` |

The saved labels are balanced between the two classes in both arrays. These checks confirm the inspected array structure, not the realism of the generated signals or generalization to real driving.

## Model artifacts

### Training model

`models/training/crash_detector.keras` is a CNN-LSTM model that loads successfully in the current Python environment.

- Expected input shape: `(batch_size, 500, 6)`
- Output shape: `(batch_size, 1)`

The corresponding `crash_detector.tflite` file currently fails to initialize with the standard TensorFlow Lite interpreter because it requires support for a Select TensorFlow operation. This export needs further investigation before it can be described as ready for deployment.

### Deployment models

| Model | Input shape | Features |
|---|---|---|
| `accel_speed_crash_model.tflite` | `(1, 60, 4)` | Accelerometer XYZ and speed |
| `cnn_gru_crash_model.tflite` | `(1, 60, 7)` | Accelerometer XYZ, gyroscope XYZ, and speed |

Both deployment models successfully initialized in the current local TensorFlow Lite inspection.

Each has a corresponding `scaler_params.json` file with the expected number of feature names, means, and scale values. Their numerical correctness, units, provenance, and compatibility with the original inference preprocessing still need to be verified.

**These models use different input specifications from the six-feature, 500-timestep training arrays.** They should be treated as separate model pipelines unless their training and preprocessing histories can be established.

## Reproducing the existing checks

The project targets Python 3.10–3.12.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

Run the automated tests:

```powershell
python -m pytest -v
```

Inspect the saved TensorFlow Lite models:

```powershell
python check_models.py
```

The last command applies to the local inspection script created during development; it is not yet part of the documented repository pipeline.

## Evaluation and known limitations

The original scripts are preserved under `scripts/original/`. Their behavior should not be confused with a fully validated, leakage-free production pipeline.

In particular, the existing training script uses the test arrays as validation data for early stopping. Consequently, the test set is involved in model selection and cannot be treated as an untouched final evaluation set.

Further work should include:

1. Establishing a clean, reproducible split strategy.
2. Verifying timestamp handling, feature units, and windowing.
3. Checking for leakage between related sensor recordings.
4. Evaluating false positives and false negatives on appropriate data.
5. Verifying scaler values and inference preprocessing against the deployment models.
6. Recording reproducible evaluation results and limitations in `reports/`.

Synthetic-data results alone cannot establish real-world crash-detection reliability.

## Safety

This project is experimental and is not a certified vehicle-safety system. It must not be relied upon as the sole means of crash detection or emergency notification. Real-world deployment would require rigorous testing on representative, appropriately labeled data and across different devices and driving conditions.

## Background and attribution

SafeTraq originated as a team graduation project. This repository is a separate, ML-focused organization of selected data, scripts, and model artifacts. It does not claim sole ownership of the full Android application, web application, or all original team contributions.

## License

No license has been selected. Until a license is added, reuse permissions should not be assumed.

## Reusable Pipeline Utilities

The project includes reusable Python utilities under `src/safetraq/`:

- **`data_validation.py`** — validates sensor data and checks for common data-quality issues.
- **`windowing.py`** — converts sensor readings into fixed-length windows. The default window size is 60 samples, with a step size of 30 samples.
- **`splitting.py`** — provides group-aware train/test splitting using scikit-learn's `GroupShuffleSplit`, helping prevent windows from the same segment from appearing in both sets.

Automated tests for these utilities are located in `tests/`. The latest recorded test run passed all 18 tests.

**Current limitation:** These utilities are tested individually, but the original training scripts have not yet been confirmed to use them as one integrated pipeline. Passing unit tests does not establish real-world crash-detection accuracy or deployment readiness.