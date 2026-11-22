
import numpy as np
import pytest

from mnist_lecun_cnn.preprocessing import (
    encode_chollet_targets,
    encode_lecun_targets,
    preprocess_chollet_images,
    preprocess_lecun_images,
)


def test_preprocess_lecun_images_has_expected_shape_dtype_and_range():
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[0, 5:15, 8:18] = 255
    images[1, 10:20, 10:20] = 128

    processed = preprocess_lecun_images(images)

    assert processed.shape == (3, 28, 28, 1)
    assert processed.dtype == np.float32
    assert processed.min() >= -1.0
    assert processed.max() <= 1.0


def test_preprocess_lecun_images_centers_and_preserves_aspect_ratio():
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 4:8, 10:18] = 255

    processed = preprocess_lecun_images(images)[0, :, :, 0]

    foreground_y, foreground_x = np.where(processed > -1.0)

    assert foreground_y.min() == 10
    assert foreground_y.max() == 17
    assert foreground_x.min() == 6
    assert foreground_x.max() == 21


def test_preprocess_lecun_images_handles_blank_images():
    images = np.zeros((1, 28, 28), dtype=np.uint8)

    processed = preprocess_lecun_images(images)

    assert processed.shape == (1, 28, 28, 1)
    np.testing.assert_array_equal(processed, -np.ones_like(processed))


def test_preprocess_lecun_images_does_not_modify_input():
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    images[0, 5:15, 5:15] = 200
    original = images.copy()

    preprocess_lecun_images(images)

    np.testing.assert_array_equal(images, original)


def test_preprocess_chollet_images_normalizes_without_geometric_changes():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[0, 5:15, 8:18] = 255
    images[1, 10:20, 10:20] = 128

    processed = preprocess_chollet_images(images)

    assert processed.shape == (2, 28, 28, 1)
    assert processed.dtype == np.float32
    assert processed.min() == 0.0
    assert processed.max() == 1.0

    np.testing.assert_allclose(
        processed[1, 10, 10, 0],
        128.0 / 255.0,
    )


def test_preprocessing_rejects_invalid_image_shape():
    images = np.zeros((2, 28, 27), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        preprocess_lecun_images(images)

    with pytest.raises(ValueError, match="shape"):
        preprocess_chollet_images(images)


def test_preprocessing_rejects_non_uint8_images():
    images = np.zeros((2, 28, 28), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        preprocess_lecun_images(images)

    with pytest.raises(ValueError, match="uint8"):
        preprocess_chollet_images(images)


def test_encode_lecun_targets_creates_bipolar_targets():
    labels = np.array([0, 4, 9], dtype=np.uint8)

    targets = encode_lecun_targets(labels)

    assert targets.shape == (3, 10)
    assert targets.dtype == np.float32
    assert np.all(np.sum(targets == 1.0, axis=1) == 1)
    assert np.all(np.sum(targets == -1.0, axis=1) == 9)

    np.testing.assert_array_equal(
        np.argmax(targets, axis=1),
        labels,
    )


def test_encode_chollet_targets_creates_one_hot_targets():
    labels = np.array([0, 4, 9], dtype=np.uint8)

    targets = encode_chollet_targets(labels)

    assert targets.shape == (3, 10)
    assert targets.dtype == np.float32
    assert np.all(np.sum(targets, axis=1) == 1.0)

    np.testing.assert_array_equal(
        np.argmax(targets, axis=1),
        labels,
    )


def test_target_encoders_reject_invalid_labels():
    invalid_labels = [
        np.array([[0, 1], [2, 3]]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0, 10]),
        np.array([-1, 2]),
    ]

    for labels in invalid_labels:
        with pytest.raises(ValueError):
            encode_lecun_targets(labels)

        with pytest.raises(ValueError):
            encode_chollet_targets(labels)
