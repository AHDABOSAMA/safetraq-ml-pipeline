"""Train-only feature scaling for windowed sensor data."""

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def _check_windows(X: np.ndarray) -> np.ndarray:
    """Validate the shape of windowed sensor data."""
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            "Expected shape (n_windows, window_size, n_features)."
        )

    return X


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training windows only.

    Statistics are computed per feature across all windows and timesteps.
    """
    X_train = _check_windows(X_train)

    if len(X_train) == 0:
        raise ValueError("Cannot fit a scaler on an empty array.")

    n_features = X_train.shape[-1]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    logger.info(
        "Fitted scaler on %d training windows with %d features",
        len(X_train),
        n_features,
    )

    return scaler


def transform_windows(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Scale windows with an already fitted scaler and keep their shape."""
    X = _check_windows(X)
    n_features = X.shape[-1]

    if n_features != scaler.n_features_in_:
        raise ValueError(
            f"Expected {scaler.n_features_in_} features, got {n_features}."
        )

    scaled = scaler.transform(X.reshape(-1, n_features))

    return scaled.reshape(X.shape).astype(np.float32)