"""Chollet's keras CNN baseline for MNIST."""

from tensorflow import keras
from tensorflow.keras import layers


def build_chollet_model() -> keras.Model:
    """Build the standard Keras MNIST CNN baseline."""

    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="image"),

            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="valid",
                name="conv1",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool1",
            ),

            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="valid",
                name="conv2",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool2",
            ),

            layers.Flatten(name="flatten"),

            layers.Dropout(
                rate=0.5,
                name="dropout",
            ),

            layers.Dense(
                units=10,
                activation="softmax",
                name="output",
            ),
        ],
        name="chollet_mnist_cnn",
    )

    return model
