"""Validation utilities for SafeTraq sensor datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FEATURES = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

REQUIRED_COLUMNS = REQUIRED_FEATURES + ["label"]


def validate_sensor_dataframe(
    dataframe: pd.DataFrame,
    source: str = "<dataframe>",
) -> None:
    """Raise ValueError when a sensor dataframe is invalid."""
    if dataframe.empty:
        raise ValueError(f"{source}: the dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source}: missing required columns: {missing_columns}"
        )

    try:
        features = dataframe[REQUIRED_FEATURES].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}: sensor features must be numeric."
        ) from error

    if not np.isfinite(features).all():
        raise ValueError(
            f"{source}: sensor features must be finite numbers "
            "without missing values."
        )

    labels = dataframe["label"]

    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be binary values 0 or 1.")


def validate_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a sensor CSV file."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {path}")

    dataframe = pd.read_csv(path)
    validate_sensor_dataframe(dataframe, source=str(path))

    return dataframe
