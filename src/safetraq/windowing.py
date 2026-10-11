"""Windowing utilities for the SafeTraq sensor pipeline."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


DEFAULT_FEATURE_COLUMNS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "speed",
)


def create_windows(
    data: pd.DataFrame,
    *,
    segment_column: str = "segment_key",
    label_column: str = "collision",
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    window_size: int = 60,
    step_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sensor windows independently within each segment.

    A window is labeled positive when any row in it has a positive
    collision label, matching the thesis notebook's ``window.max()``
    labeling rule.

    Returns:
        X: Float32 array with shape (n_windows, window_size, n_features).
        y: Int32 array with one label per window.
        groups: Array containing the source segment key for each window.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    required_columns = {segment_column, label_column, *feature_columns}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data[list(feature_columns) + [label_column]].isna().any().any():
        raise ValueError(
            "Feature and label columns must not contain missing values."
        )

    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[object] = []

    for segment_key, segment_data in data.groupby(
        segment_column, sort=False, dropna=False
    ):
        if pd.isna(segment_key):
            raise ValueError("Segment keys must not be missing.")

        segment_data = segment_data.reset_index(drop=True)

        if len(segment_data) < window_size:
            logger.debug(
                "Skipping segment %s: %d rows is shorter than window_size %d",
                segment_key,
                len(segment_data),
                window_size,
            )
            continue

        for start in range(
            0, len(segment_data) - window_size + 1, step_size
        ):
            window = segment_data.iloc[start : start + window_size]

            windows.append(
                window.loc[:, list(feature_columns)].to_numpy(
                    dtype=np.float32
                )
            )
            labels.append(int(window[label_column].max() > 0))
            groups.append(segment_key)

    logger.info(
        "Created %d windows (%d positive) from %d segments",
        len(windows),
        sum(labels),
        len(set(groups)),
    )

    if not windows:
        return (
            np.empty(
                (0, window_size, len(feature_columns)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=object),
        )

    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups),
    )