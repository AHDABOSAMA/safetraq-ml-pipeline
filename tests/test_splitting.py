import numpy as np
import pytest

from safetraq.splitting import split_by_group


def make_data(number_of_groups=10, samples_per_group=10):
    groups = np.repeat(
        [f"segment_{i}" for i in range(number_of_groups)],
        samples_per_group,
    )
    X = np.zeros((len(groups), 60, 7), dtype=np.float32)
    y = np.zeros(len(groups), dtype=np.int32)
    return X, y, groups


def test_train_and_test_groups_do_not_overlap():
    X, y, groups = make_data()

    train_idx, test_idx = split_by_group(X, y, groups)

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(test_groups)
    assert len(train_idx) > 0
    assert len(test_idx) > 0


def test_split_is_reproducible():
    X, y, groups = make_data()

    train_a, test_a = split_by_group(X, y, groups)
    train_b, test_b = split_by_group(X, y, groups)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)


def test_rejects_mismatched_lengths():
    X, y, groups = make_data()

    with pytest.raises(ValueError, match="same number of samples"):
        split_by_group(X, y[:-1], groups)


def test_rejects_empty_dataset():
    X = np.empty((0, 60, 7))
    y = np.array([], dtype=int)
    groups = np.array([], dtype=str)

    with pytest.raises(ValueError, match="empty dataset"):
        split_by_group(X, y, groups)


def test_rejects_missing_group_values():
    X = np.zeros((4, 2))
    y = np.array([0, 0, 1, 1])
    groups = np.array(["segment_a", "segment_b", None, "segment_d"], dtype=object)

    with pytest.raises(ValueError, match="must not be missing"):
        split_by_group(X, y, groups)


def test_rejects_invalid_test_size():
    X, y, groups = make_data()

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        split_by_group(X, y, groups, test_size=1.0)
