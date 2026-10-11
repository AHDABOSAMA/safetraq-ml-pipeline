import numpy as np
import pandas as pd
import pytest

from safetraq.windowing import create_windows


def make_segment(segment_key, length, collision_at=None):
    data = {
        "segment_key": [segment_key] * length,
        "collision": [0] * length,
    }

    for column in (
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z", "speed",
    ):
        data[column] = np.arange(length, dtype=float)

    if collision_at is not None:
        data["collision"][collision_at] = 1

    return pd.DataFrame(data)


def test_creates_expected_window_shapes_and_groups():
    data = pd.concat(
        [
            make_segment("segment_a", 90),
            make_segment("segment_b", 60),
        ],
        ignore_index=True,
    )

    X, y, groups = create_windows(data)

    assert X.shape == (3, 60, 7)
    assert y.shape == (3,)
    assert groups.tolist() == ["segment_a", "segment_a", "segment_b"]
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_collision_in_window_makes_label_positive():
    data = make_segment("segment_a", 60, collision_at=25)

    _, y, _ = create_windows(data)

    assert y.tolist() == [1]


def test_short_segment_produces_no_windows():
    data = make_segment("segment_a", 59)

    X, y, groups = create_windows(data)

    assert X.shape == (0, 60, 7)
    assert y.size == 0
    assert groups.size == 0


def test_windows_do_not_cross_segment_boundaries():
    data = pd.concat(
        [
            make_segment("segment_a", 40),
            make_segment("segment_b", 40),
        ],
        ignore_index=True,
    )

    X, y, groups = create_windows(data)

    assert X.shape == (0, 60, 7)
    assert y.size == 0
    assert groups.size == 0


def test_rejects_missing_required_columns():
    data = pd.DataFrame({"segment_key": ["segment_a"], "collision": [0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        create_windows(data)


def test_rejects_invalid_window_size():
    data = make_segment("segment_a", 60)

    with pytest.raises(ValueError, match="window_size must be positive"):
        create_windows(data, window_size=0)
