import pandas as pd
import pytest

from safetraq.data_validation import (
    REQUIRED_FEATURES,
    validate_sensor_dataframe,
)


@pytest.fixture
def valid_sensor_dataframe():
    """Return a small, valid sensor dataset."""
    data = {feature: [0.1, 0.2, 0.3] for feature in REQUIRED_FEATURES}
    data["label"] = [0, 1, 0]
    return pd.DataFrame(data)


def test_valid_dataframe_passes(valid_sensor_dataframe):
    validate_sensor_dataframe(valid_sensor_dataframe)


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_sensor_dataframe(pd.DataFrame())


def test_missing_feature_is_rejected(valid_sensor_dataframe):
    dataframe = valid_sensor_dataframe.drop(columns=["accel_x"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_sensor_dataframe(dataframe)


def test_missing_sensor_value_is_rejected(valid_sensor_dataframe):
    valid_sensor_dataframe.loc[0, "gyro_x"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_sensor_dataframe(valid_sensor_dataframe)


def test_infinite_sensor_value_is_rejected(valid_sensor_dataframe):
    valid_sensor_dataframe.loc[0, "accel_y"] = float("inf")

    with pytest.raises(ValueError, match="finite numbers"):
        validate_sensor_dataframe(valid_sensor_dataframe)


def test_invalid_label_is_rejected(valid_sensor_dataframe):
    valid_sensor_dataframe.loc[0, "label"] = 2

    with pytest.raises(ValueError, match="binary values"):
        validate_sensor_dataframe(valid_sensor_dataframe)
