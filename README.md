# SafeTraq ML Pipeline

A machine learning engineering project focused on detecting potential vehicle crashes from smartphone motion-sensor data.

This repository develops a standalone, reproducible ML workflow inspired by the SafeTraq graduation project. It focuses on data quality, preprocessing, model evaluation, and deployment considerations rather than the full Android application.

> **Project status:** Initial repository setup. The pipeline is being built and validated incrementally. Results will be documented only after they can be reproduced and verified.

## Problem

Smartphone accelerometer and gyroscope readings can capture sudden changes in vehicle motion. However, abrupt braking, potholes, and speed bumps may resemble crash events. A useful crash-detection system must therefore evaluate both its ability to detect crashes and its ability to avoid false alarms.

## Objectives

- Establish a clear, reproducible ML project structure.
- Document sensor features, sampling assumptions, and windowing.
- Prevent data leakage between training and evaluation.
- Compare appropriate classification approaches.
- Evaluate false positives, false negatives, precision, recall, and other relevant metrics.
- Prepare for deployment constraints, including consistency between training preprocessing and inference.

## Planned workflow

1. **Data validation:** inspect labels, feature columns, missing values, sampling intervals, and class balance.
2. **Segmentation:** handle timestamp gaps and create fixed-length sensor windows.
3. **Data splitting:** separate training and evaluation data before fitting preprocessing steps.
4. **Preprocessing:** fit normalization parameters on training data only.
5. **Model training:** establish a baseline before evaluating more complex architectures.
6. **Evaluation:** report confusion matrices, precision, recall, and false-positive behavior.
7. **Export readiness:** document the preprocessing and input requirements needed for consistent inference.

These are development goals, not claims that every stage has already been implemented.

## Sensor data

The project is designed around smartphone motion-sensor time series, particularly:

- Accelerometer readings along the X, Y, and Z axes.
- Gyroscope readings along the X, Y, and Z axes.
- Additional sensor features only when their availability and use are documented.

The intended windowing and sampling configuration will be explicitly documented and validated against the source data before model training.

## Repository structure

```text
safetraq-ml-pipeline/
├── configs/          # Pipeline configuration
├── docs/             # Methodology and technical documentation
├── reports/          # Tracked evaluation summaries
├── src/
│   └── safetraq/     # Python package
├── tests/            # Automated tests
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

The project currently targets Python 3.10–3.12.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

If Python 3.11 is not installed, check your available Python versions before proceeding.

## Data and reproducibility

Datasets, generated arrays, and trained model binaries are excluded from version control by default. This keeps the repository focused on source code, configuration, tests, and documented results.

The source project includes synthetic sensor-data generation. Synthetic data can help validate pipeline mechanics, but performance measured only on synthetic data does **not** establish real-world crash-detection reliability.

Any reported metrics should identify the dataset, split strategy, preprocessing, decision threshold, and evaluation procedure used.

## Limitations and safety

This project is experimental and is not a certified vehicle-safety system. It must not be relied upon as the sole means of crash detection or emergency notification.

Real-world validation requires appropriately collected and labeled sensor data, realistic non-crash events, careful false-alarm analysis, and testing across devices and driving conditions.

## Background

SafeTraq originated as a graduation project involving smartphone-based crash detection and emergency-alert functionality. This repository is a separate, ML-focused engineering effort. It does not claim ownership of the entire team application or imply that the complete workflow described above is already implemented.

## License

A license has not yet been selected. Until one is added, reuse permissions should not be assumed.