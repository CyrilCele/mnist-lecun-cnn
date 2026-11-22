"""MNIST loading, validation, and reproducible dataset splitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

ImageArray = NDArray[np.uint8]
LabelArray = NDArray[np.uint8]
IndexArray = NDArray[np.int64]

RANDOM_SEED = 42
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28)
EXPECTED_TRAIN_SIZE = 60_000
EXPECTED_TEST_SIZE = 10_000
VALIDATION_SIZE = 6_000


@dataclass(frozen=True)
class DatasetSplit:
    """Indices defining the training and validation subsets."""

    train_indices: IndexArray
    validation_indices: IndexArray


@dataclass(frozen=True)
class MNISTDataset:
    """Raw MNIST data and its reproducible training/validation split."""

    x_train: ImageArray
    y_train: LabelArray
    x_test: ImageArray
    y_test: LabelArray
    split: DatasetSplit


def validate_mnist(
    x_train: NDArray,
    y_train: NDArray,
    x_test: NDArray,
    y_test: NDArray,
    *,
    expected_train_size: int = EXPECTED_TRAIN_SIZE,
    expected_test_size: int = EXPECTED_TEST_SIZE,
) -> None:
    """Raise ValueError if the supplied MNIST arrays are invalid."""

    expected_train_shape = (expected_train_size, *IMAGE_SHAPE)
    expected_test_shape = (expected_test_size, *IMAGE_SHAPE)

    if x_train.shape != expected_train_shape:
        raise ValueError(
            f"Expected training images with shape {expected_train_shape}; "
            f"received {x_train.shape}."
        )

    if x_test.shape != expected_test_shape:
        raise ValueError(
            f"Expected test images with shape {expected_test_shape}; "
            f"received {x_test.shape}."
        )

    if y_train.shape != (expected_train_size,):
        raise ValueError(
            f"Expected training labels with shape ({expected_train_size},); "
            f"received {y_train.shape}."
        )

    if y_test.shape != (expected_test_size,):
        raise ValueError(
            f"Expected test labels with shape ({expected_test_size},); "
            f"received {y_test.shape}."
        )

    for name, images in (("training", x_train), ("test", x_test)):
        if images.dtype != np.uint8:
            raise ValueError(
                f"{name.capitalize()} images must have dtype uint8; "
                f"received {images.dtype}."
            )

        if images.size and (images.min() < 0 or images.max() > 255):
            raise ValueError(
                f"{name.capitalize()} pixels must be in [0, 255].")

    for name, labels in (("training", y_train), ("test", y_test)):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{name.capitalize()} labels must be integers.")

        if labels.size and (
            labels.min() < 0 or labels.max() >= NUM_CLASSES
        ):
            raise ValueError(
                f"{name.capitalize()} labels must be in "
                f"[0, {NUM_CLASSES - 1}]."
            )


def create_train_validation_split(
    labels: NDArray,
    *,
    validation_size: int = VALIDATION_SIZE,
    random_seed: int = RANDOM_SEED,
) -> DatasetSplit:
    """Create a deterministic, stratified split using sample indices."""

    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must contain integers.")

    if labels.size == 0:
        raise ValueError("Cannot split an empty label array.")

    if np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError(f"Labels must be in [0, {NUM_CLASSES - 1}].")

    if not 0 < validation_size < len(labels):
        raise ValueError(
            "validation_size must be greater than zero and smaller "
            "than the number of samples."
        )

    indices = np.arange(len(labels), dtype=np.int64)

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_seed,
        stratify=labels,
    )

    return DatasetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def load_mnist_dataset(
    *,
    random_seed: int = RANDOM_SEED,
) -> MNISTDataset:
    """Load MNIST, validate it, and create the training/validation split."""

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    validate_mnist(
        x_train,
        y_train,
        x_test,
        y_test,
        expected_train_size=EXPECTED_TRAIN_SIZE,
        expected_test_size=EXPECTED_TEST_SIZE,
    )

    split = create_train_validation_split(
        y_train,
        validation_size=VALIDATION_SIZE,
        random_seed=random_seed,
    )

    return MNISTDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        split=split,
    )
