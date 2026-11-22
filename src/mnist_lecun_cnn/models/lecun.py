
"""LeCun-inspired CNN with restricted feature-map connectivity."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="mnist_lecun_cnn")
class ChannelSelection(layers.Layer):
    """Select specified input feature maps."""

    def __init__(self, indices: tuple[int, ...], **kwargs):
        super().__init__(**kwargs)
        self.indices = tuple(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


@keras.utils.register_keras_serializable(package="mnist_lecun_cnn")
class PerChannelAffine(layers.Layer):
    """Trainable per-channel scale and bias followed by tanh."""

    def build(self, input_shape):
        channels = int(input_shape[-1])

        self.scale = self.add_weight(
            name="scale",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(
            inputs * self.scale + self.bias
        )


# H3 output maps and their permitted H2 input maps.
# Indices are zero-based.
H3_CONNECTIONS = (
    (0,),       # H3.1
    (0, 1),     # H3.2
    (0, 1),     # H3.3
    (1,),       # H3.4
    (0, 1),     # H3.5
    (0, 1),     # H3.6
    (2,),       # H3.7
    (2, 3),     # H3.8
    (2, 3),     # H3.9
    (3,),       # H3.10
    (2, 3),     # H3.11
    (2, 3),     # H3.12
)


def build_lecun_model() -> keras.Model:
    """Build the LeCun-inspired CNN for 28x28 grayscale images."""

    inputs = keras.Input(shape=(28, 28, 1), name="image")

    # H1: 4 feature maps, 5x5 valid convolution.
    h1 = layers.Conv2D(
        filters=4,
        kernel_size=5,
        padding="valid",
        activation="tanh",
        name="h1_convolution",
    )(inputs)

    # H2: average subsampling and trainable per-map affine transform.
    h2 = layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        name="h2_average_pooling",
    )(h1)

    h2 = PerChannelAffine(name="h2_trainable_subsampling")(h2)

    # H3: 12 maps with restricted H2 connectivity.
    h3_maps = []

    for output_index, input_indices in enumerate(H3_CONNECTIONS):
        selected = ChannelSelection(
            indices=input_indices,
            name=f"h3_{output_index + 1}_input_selection",
        )(h2)

        feature_map = layers.Conv2D(
            filters=1,
            kernel_size=5,
            padding="valid",
            activation="tanh",
            name=f"h3_{output_index + 1}_convolution",
        )(selected)

        h3_maps.append(feature_map)

    h3 = layers.Concatenate(name="h3_feature_maps")(h3_maps)

    # H4: average subsampling and trainable per-map affine transform.
    h4 = layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        name="h4_average_pooling",
    )(h3)

    h4 = PerChannelAffine(name="h4_trainable_subsampling")(h4)

    # Output: 4x4x12 = 192 features, followed by 10 tanh outputs.
    flattened = layers.Flatten(name="flatten")(h4)

    outputs = layers.Dense(
        units=10,
        activation="tanh",
        name="output",
    )(flattened)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="lecun_inspired_cnn",
    )
