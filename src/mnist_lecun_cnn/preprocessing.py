"""Preprocessing pipelines for the LeCun-inspired and Chollet CNNs."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 10
IMAGE_SIZE = 28
LECUN_CONTENT_SIZE = 16

ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


def _validate_images(images: NDArray) -> None:
    """Validate raw MNIST image arrays."""

    if images.ndim != 3 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Images must have shape (N, {IMAGE_SIZE}, {IMAGE_SIZE}); "
            f"received {images.shape}."
        )

    if images.dtype != np.uint8:
        raise ValueError(
            f"Images must have dtype uint8; recieved {images.dtype}."
        )


def _validate_labels(labels: NDArray) -> None:
    """Validate integer class labels in the range 0-9."""

    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must contain integers.")

    if np.any((labels < 0) | (labels >= NUM_CLASSES)):
        raise ValueError(f"Labels must be in [0, {NUM_CLASSES - 1}].")


def preprocess_lecun_images(images: ImageArray) -> FloatArray:
    """
    Crop, resize, center, and normalize images for the LeCun-inspired CNN.

    Foreground pixels are defined as values greater than zero.
    The bounding box is resized with bilinear interpolation to fit
    within a 16x16 region, preserving its aspect ratio. The resized
    image is centered on a 28x28 black canvas.

    Pixel values are then normalized from [0, 255] to [-1, 1].
    Black images remain blank and normalize to -1.
    """

    _validate_images(images)

    processed = np.empty(
        (len(images), IMAGE_SIZE, IMAGE_SIZE, 1),
        dtype=np.float32,
    )

    for index, image in enumerate(images):
        foreground_y, foreground_x = np.where(image > 0)

        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        if foreground_y.size:
            top = int(foreground_y.min())
            bottom = int(foreground_y.max()) + 1
            left = int(foreground_x.min())
            right = int(foreground_x.max()) + 1

            cropped = image[top:bottom, left:right]

            crop_height, crop_width = cropped.shape
            scale = min(
                LECUN_CONTENT_SIZE / crop_height,
                LECUN_CONTENT_SIZE / crop_width,
            )

            resized_height = max(
                1,
                min(
                    LECUN_CONTENT_SIZE,
                    int(np.floor(crop_height * scale + 0.5)),
                ),
            )
            resized_width = max(
                1,
                min(
                    LECUN_CONTENT_SIZE,
                    int(np.floor(crop_width * scale + 0.5)),
                ),
            )

            resized = np.asarray(
                Image.fromarray(cropped).resize(
                    (resized_width, resized_height),
                    resample=Image.Resampling.BILINEAR,
                ),
            )

            canvas_top = (IMAGE_SIZE - resized_height) // 2
            canvas_left = (IMAGE_SIZE - resized_width) // 2

            canvas[
                canvas_top: canvas_top + resized_height,
                canvas_left: canvas_left + resized_width,
            ] = resized

        processed[index, :, :, 0] = (
            canvas.astype(np.float32) / 127.5
        ) - 1.0

    return processed


def preprocess_chollet_images(images: ImageArray) -> FloatArray:
    """Normalize MNIST images for the Chollet CNN without geometric changes."""

    _validate_images(images)

    return images.astype(np.float32)[..., np.newaxis] / 255.0


def encode_lecun_targets(labels: NDArray) -> FloatArray:
    """Encode class labels as bipolar targets: +1 correct, -1 otherwise."""

    _validate_labels(labels)

    targets = np.full(
        (len(labels), NUM_CLASSES),
        -1.0,
        dtype=np.float32,
    )
    targets[np.arange(len(labels)), labels] = 1.0

    return targets


def encode_chollet_targets(labels: NDArray) -> FloatArray:
    """Encode class labels as ten-class one-hot targets."""

    _validate_labels(labels)

    return to_categorical(
        labels,
        num_classes=NUM_CLASSES,
    ).astype(np.float32)
