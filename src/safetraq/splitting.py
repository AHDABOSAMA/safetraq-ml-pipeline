"""Group-aware splitting utilities for the SafeTraq pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 68,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test indices without splitting groups across sets."""
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if X.ndim == 0 or y.ndim != 1 or groups.ndim != 1:
        raise ValueError("X must have samples; y and groups must be one-dimensional.")

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must contain the same number of samples.")

    if len(y) == 0:
        raise ValueError("Cannot split an empty dataset.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if pd.isna(groups).any():
        raise ValueError("Group values must not be missing.")

    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct groups are required.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups))

    return train_indices, test_indices
