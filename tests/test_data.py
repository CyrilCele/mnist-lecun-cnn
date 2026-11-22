
import numpy as np
import pytest

import mnist_lecun_cnn.data as data_module
from mnist_lecun_cnn.data import (
    create_train_validation_split,
    validate_mnist,
)


def test_create_split_has_expected_sizes():
    labels = np.repeat(np.arange(10), 100)

    split = create_train_validation_split(
        labels,
        validation_size=200,
        random_seed=42,
    )

    assert len(split.train_indices) == 800
    assert len(split.validation_indices) == 200


def test_create_split_is_reproducible():
    labels = np.repeat(np.arange(10), 100)

    first = create_train_validation_split(labels, validation_size=200)
    second = create_train_validation_split(labels, validation_size=200)

    np.testing.assert_array_equal(
        first.train_indices,
        second.train_indices,
    )
    np.testing.assert_array_equal(
        first.validation_indices,
        second.validation_indices,
    )


def test_create_split_has_no_overlap_and_preserves_all_samples():
    labels = np.repeat(np.arange(10), 100)

    split = create_train_validation_split(labels, validation_size=200)

    assert set(split.train_indices).isdisjoint(
        set(split.validation_indices)
    )

    combined = np.concatenate(
        [split.train_indices, split.validation_indices]
    )
    np.testing.assert_array_equal(
        np.sort(combined),
        np.arange(len(labels)),
    )


def test_create_split_preserves_class_distribution():
    labels = np.repeat(np.arange(10), 100)

    split = create_train_validation_split(labels, validation_size=200)

    validation_labels = labels[split.validation_indices]
    class_counts = np.bincount(validation_labels, minlength=10)

    np.testing.assert_array_equal(class_counts, np.full(10, 20))


def test_create_split_rejects_invalid_labels():
    with pytest.raises(ValueError, match="one-dimensional"):
        create_train_validation_split(np.zeros((10, 1), dtype=int))

    with pytest.raises(ValueError, match="integers"):
        create_train_validation_split(np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="empty"):
        create_train_validation_split(np.array([], dtype=int))

    with pytest.raises(ValueError, match="must be in"):
        create_train_validation_split(np.array([0, 1, 10]))


def test_create_split_rejects_invalid_validation_size():
    labels = np.repeat(np.arange(10), 10)

    with pytest.raises(ValueError, match="validation_size"):
        create_train_validation_split(labels, validation_size=0)

    with pytest.raises(ValueError, match="validation_size"):
        create_train_validation_split(labels, validation_size=len(labels))


def test_validate_mnist_accepts_valid_arrays():
    x_train = np.zeros((20, 28, 28), dtype=np.uint8)
    y_train = np.repeat(np.arange(10), 2).astype(np.uint8)
    x_test = np.zeros((10, 28, 28), dtype=np.uint8)
    y_test = np.arange(10, dtype=np.uint8)

    validate_mnist(
        x_train,
        y_train,
        x_test,
        y_test,
        expected_train_size=20,
        expected_test_size=10,
    )


def test_validate_mnist_rejects_wrong_image_shape():
    x_train = np.zeros((20, 28, 27), dtype=np.uint8)
    y_train = np.zeros(20, dtype=np.uint8)
    x_test = np.zeros((10, 28, 28), dtype=np.uint8)
    y_test = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="training images"):
        validate_mnist(
            x_train,
            y_train,
            x_test,
            y_test,
            expected_train_size=20,
            expected_test_size=10,
        )


def test_validate_mnist_rejects_non_uint8_images():
    x_train = np.zeros((20, 28, 28), dtype=np.float32)
    y_train = np.zeros(20, dtype=np.uint8)
    x_test = np.zeros((10, 28, 28), dtype=np.uint8)
    y_test = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="dtype uint8"):
        validate_mnist(
            x_train,
            y_train,
            x_test,
            y_test,
            expected_train_size=20,
            expected_test_size=10,
        )


def test_validate_mnist_rejects_out_of_range_labels():
    x_train = np.zeros((20, 28, 28), dtype=np.uint8)
    y_train = np.zeros(20, dtype=np.uint8)
    y_train[0] = 10

    x_test = np.zeros((10, 28, 28), dtype=np.uint8)
    y_test = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="labels must be in"):
        validate_mnist(
            x_train,
            y_train,
            x_test,
            y_test,
            expected_train_size=20,
            expected_test_size=10,
        )


def test_load_mnist_dataset_preserves_training_and_test_labels(monkeypatch):
    x_train = np.zeros((100, 28, 28), dtype=np.uint8)
    y_train = np.repeat(np.arange(10), 10).astype(np.uint8)

    x_test = np.zeros((20, 28, 28), dtype=np.uint8)
    y_test = np.repeat(np.arange(10), 2).astype(np.uint8)

    monkeypatch.setattr(data_module, "EXPECTED_TRAIN_SIZE", 100)
    monkeypatch.setattr(data_module, "EXPECTED_TEST_SIZE", 20)
    monkeypatch.setattr(data_module, "VALIDATION_SIZE", 20)

    monkeypatch.setattr(
        data_module.mnist,
        "load_data",
        lambda: ((x_train, y_train), (x_test, y_test)),
    )

    dataset = data_module.load_mnist_dataset()

    np.testing.assert_array_equal(dataset.y_train, y_train)
    np.testing.assert_array_equal(dataset.y_test, y_test)
    assert dataset.y_train.shape == (100,)
    assert dataset.y_test.shape == (20,)
