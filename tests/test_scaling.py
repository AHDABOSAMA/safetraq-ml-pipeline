import numpy as np
import pytest

from safetraq.scaling import fit_scaler, transform_windows


N_FEATURES = 7


def make_windows(loc=0.0, scale=1.0, seed=0, n=20):
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc,
        scale,
        size=(n, 60, N_FEATURES),
    )


def test_scaler_uses_training_statistics_only():
    X_train = make_windows(loc=0.0, seed=1)
    X_test = make_windows(loc=100.0, seed=2)

    scaler = fit_scaler(X_train)

    train_flat = X_train.reshape(-1, N_FEATURES)

    np.testing.assert_allclose(
        scaler.mean_,
        train_flat.mean(axis=0),
    )
    np.testing.assert_allclose(
        scaler.scale_,
        train_flat.std(axis=0),
    )

    assert not np.allclose(
        scaler.mean_,
        X_test.reshape(-1, N_FEATURES).mean(axis=0),
    )


def test_training_data_is_standardized():
    X_train = make_windows(loc=5.0, scale=3.0, seed=3)

    scaler = fit_scaler(X_train)
    scaled = transform_windows(scaler, X_train)

    flat = scaled.reshape(-1, N_FEATURES)

    np.testing.assert_allclose(
        flat.mean(axis=0),
        0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        flat.std(axis=0),
        1,
        atol=1e-3,
    )


def test_test_data_is_not_re_centered():
    scaler = fit_scaler(make_windows(loc=0.0, seed=4))
    X_test = make_windows(loc=100.0, seed=5)

    scaled_test = transform_windows(scaler, X_test)

    # Test data far from the training distribution stays far from zero.
    assert scaled_test.mean() > 10


def test_output_keeps_shape_and_is_float32():
    X = make_windows(seed=6)

    scaled = transform_windows(fit_scaler(X), X)

    assert scaled.shape == X.shape
    assert scaled.dtype == np.float32


def test_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        fit_scaler(np.zeros((10, N_FEATURES)))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_scaler(np.zeros((0, 60, N_FEATURES)))


def test_rejects_feature_count_mismatch():
    scaler = fit_scaler(make_windows(seed=7))

    with pytest.raises(ValueError):
        transform_windows(
            scaler,
            np.zeros((5, 60, N_FEATURES - 1)),
        )