"""Reproducible model compilation and training utilities."""

from time import perf_counter
from typing import Any

import tensorflow as tf
from tensorflow import keras


def set_random_seeds(
    seed: int,
    deterministic_tensorflow: bool = True,
) -> bool:
    """
    Set Python, NumPy, and TensorFlow seeds.

    Returns whether deterministic TensorFlow operations were enabled.
    """

    keras.utils.set_random_seed(seed)

    if not deterministic_tensorflow:
        return False

    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        return False

    return True


def compile_model(
    model: keras.Model,
    model_name: str,
    config: dict[str, Any],
) -> keras.Model:
    """Compile a model using its configuration."""

    if model_name not in ("lecun", "chollet"):
        raise ValueError(f"Unknown model name {model_name}")

    model_config = config["models"][model_name]
    optimizer_config = config["optimizer"]

    expected_parameters = model_config["expected_parameters"]

    if model.count_params() != expected_parameters:
        raise ValueError(
            f"{model_name} model has {model.count_params():,} parameters; "
            f"expected {expected_parameters:,}."
        )

    optimizer = keras.optimizers.Adam(
        learning_rate=optimizer_config["learning_rate"],
        beta_1=optimizer_config["beta_1"],
        beta_2=optimizer_config["beta_2"],
        epsilon=optimizer_config["epsilon"],
    )

    loss_functions = {
        "mean_squared_error": keras.losses.MeanSquaredError,
        "categorical_crossentropy": keras.losses.CategoricalCrossentropy,
    }

    loss_name = model_config["loss"]

    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    model.compile(
        optimizer=optimizer,
        loss=loss_functions[loss_name](),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    return model


def train_model(
    model: keras.Model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    verbose: int = 2,
) -> tuple[keras.callbacks.History, float]:
    """
    Train a compiled model and measure fitting duration in seconds.

    Model construction, compilation, and preprocessing are excluded
    from the duration measurement.
    """

    start_time = perf_counter()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=verbose,
    )

    duration_seconds = perf_counter() - start_time

    return history, duration_seconds
