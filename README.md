# SafeTraq ML Pipeline

[![Tests](https://github.com/AHDABOSAMA/safetraq-ml-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/AHDABOSAMA/safetraq-ml-pipeline/actions/workflows/tests.yml)

Crash detection from smartphone motion-sensor data: data preparation, model comparison, and a stress test on real driving sessions.

This repository holds the machine-learning part of **SafeTraq**, a team graduation project (Ain Shams University, 2026). The Android app and website were built by teammates and live in the team repository: [Crash_Detection_App](https://github.com/AHDABOSAMA/Crash_Detection_App).

## Key finding

Models that look fine on simulated data can fail badly on real driving.

- On the simulated test split, all eight models land within about ten points of each other. Random Forest scores highest (80.5% accuracy).
- On 22 real driving sessions (speed bumps, potholes, aggressive, standard and slow driving), **Random Forest flagged every window as a crash**, while the **CNN+GRU flagged 5.4%**.
- Potholes are the weak spot: 25.5% of pothole windows were flagged, mostly from a single session.

Details and caveats are in [Results](#results) and [Limitations](#limitations).

## The pipeline

The full pipeline is in [`notebooks/01_data_pipeline_and_models.ipynb`](notebooks/01_data_pipeline_and_models.ipynb), with saved outputs.

1. **Load and merge 7 sources.** Crash and normal-driving recordings from two machines, with mixed delimiters and one malformed single-column CSV that is detected and repaired. Each row keeps its session and source ID.
2. **Clean.** Drop duplicates and missing rows, sort by time, and set crash labels to 0 in the traffic-jam file, which is known to contain normal driving only.
3. **Clip.** Limit each sensor channel to its 0.1st to 99.9th percentile.
4. **Segment.** Start a new segment wherever the gap between samples exceeds 0.3 s (333 segments).
5. **Window.** 60 timesteps (about 3 s at an effective 20 Hz) by 7 features (accelerometer XYZ, gyroscope XYZ, speed), step 30 (50% overlap). A window is a crash if any sample in it is a crash. Result: **6,830 windows** (3,383 normal, 3,447 crash).
6. **Split without leakage.** `GroupShuffleSplit` by segment (test size 0.2, seed 68), so windows from one segment never land in both train and test. The notebook checks that the groups do not overlap.
7. **Scale.** `StandardScaler` fitted on training data only.
8. **Compare eight models** and evaluate the best deep model on real sessions.

Steps 5 and 6 are also available as tested functions in `src/safetraq/`.

## Results

### Model comparison (simulated test split)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.805 | 0.754 | 0.911 | 0.825 | 0.915 | 0.910 |
| CNN + GRU | 0.783 | 0.734 | 0.893 | 0.806 | 0.903 | 0.907 |
| CNN + LSTM | 0.782 | 0.767 | 0.812 | 0.789 | 0.897 | 0.892 |
| CNN | 0.793 | 0.818 | 0.757 | 0.786 | 0.900 | 0.893 |
| CNN + SVM | 0.782 | 0.798 | 0.758 | 0.777 | 0.892 | 0.885 |
| LSTM | 0.778 | 0.796 | 0.752 | 0.773 | 0.893 | 0.895 |
| SVM | 0.744 | 0.742 | 0.752 | 0.747 | 0.859 | 0.853 |
| KNN | 0.709 | 0.766 | 0.606 | 0.677 | 0.798 | 0.760 |

On this split the CNN+GRU does not beat Random Forest. The difference appears only on real data.

### Real driving sessions (CNN+GRU)

These sessions contain no crashes, so every flagged window is a false alarm.

| Session type | Sessions | Windows | Flagged as crash | Rate |
|---|---|---|---|---|
| Standard driving | 2 | 1,217 | 3 | 0.2% |
| Speed bumps | 9 | 1,755 | 11 | 0.6% |
| Aggressive driving | 1 | 1,249 | 12 | 1.0% |
| Slow driving | 2 | 3,373 | 56 | 1.7% |
| Potholes | 8 | 1,623 | 414 | 25.5% |
| **Total** | **22** | **9,217** | **496** | **5.4%** |

Random Forest on the same sessions flagged 100% of windows in all 22. One of 23 sessions could not be processed (its columns did not match the expected format) and is excluded.

## Repository structure

```text
├── notebooks/
│   └── 01_data_pipeline_and_models.ipynb   # the pipeline above, with outputs
├── src/safetraq/
│   ├── windowing.py        # create_windows: per-segment windows and labels
│   ├── splitting.py        # split_by_group: leakage-safe GroupShuffleSplit
│   ├── scaling.py          # fit_scaler: StandardScaler fitted on training data only
│   └── data_validation.py  # sensor data checks
├── tests/                  # 25 tests
├── models/
│   ├── deployment/         # TFLite models and scaler parameters
│   └── training/           # prototype model (see Legacy)
├── scripts/
│   ├── original/           # legacy prototype scripts
│   └── validation/         # TFLite inspection script
├── data/                   # synthetic sample data and prototype arrays
├── docs/model_cards.md
├── pyproject.toml
└── requirements.txt
```

## Getting started

Python 3.10 to 3.12.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m pytest -v
```

### Reproducing the notebook

The notebook reads its data from Kaggle paths (`/kaggle/input/datasets/youssefashraf57/...`), which are not available locally. To rerun it, open it on Kaggle with those datasets attached, or point the loading cells at local copies. Dataset names, files and provenance are listed in [`data/README.md`](data/README.md).

## Model artifacts

| File | Input | Notes |
|---|---|---|
| `models/deployment/cnn_gru/cnn_gru_crash_model.tflite` | `(1, 60, 7)` | Same input specification as the notebook's CNN+GRU. The export code is not in the notebook (the conversion cell is commented out), so provenance is not fully documented. |
| `models/deployment/accel_speed/accel_speed_crash_model.tflite` | `(1, 60, 4)` | Accelerometer XYZ and speed only. |
| `models/training/crash_detector.keras` | `(500, 6)` | CNN+LSTM from the early prototype. Not the model reported above. |

More detail is in [`docs/model_cards.md`](docs/model_cards.md).

## Limitations

- **No crash recall on real data.** The real sessions contain no crashes, so only false alarms were measured there. How well the model detects real crashes is untested.
- **Pothole false alarms.** 25.5% of pothole windows were flagged, and one session accounts for most of them.
- **Simulated training data.** Crash examples come from simulation, and simulated crashes may not match real ones.
- **Early stopping is not group-aware.** The deep models use `validation_split=0.2` on the training windows, which can leak between overlapping windows of the same segment. The test set itself was never used for early stopping.
- **Single split.** Results come from one group split with one seed, with no repeated runs or confidence intervals.
- **Reproducibility.** The notebook depends on Kaggle-hosted datasets, and the deployment models' preprocessing and export are not yet verified end to end.
- **Legacy prototype.** The scripts in `scripts/original/` use 5-second, 500-step windows and a CNN+LSTM, and the prototype's early stopping uses the test arrays. They are kept for history only.

## Safety

This is an experimental research project, not a certified vehicle-safety system. It must not be relied on as the only means of crash detection or emergency notification.

## Attribution

SafeTraq was a six-person graduation project. I owned the data preprocessing and the machine-learning work in this repository. The Android application, the website and the original project infrastructure were built by teammates, in the [team repository](https://github.com/AHDABOSAMA/Crash_Detection_App).

## License

MIT. See [LICENSE](LICENSE).
