
"""Tests for model compilation and reproducibility utilities."""

from tensorflow import keras

from mnist_lecun_cnn.models.chollet import build_chollet_model
from mnist_lecun_cnn.models.lecun import build_lecun_model
from mnist_lecun_cnn.training import compile_model, set_random_seeds


def load_config():
    import json
    from pathlib import Path

    with Path("configs/experiment.json").open(encoding="utf-8") as file:
        return json.load(file)


def test_set_random_seeds_returns_determinism_status():
    result = set_random_seeds(42, deterministic_tensorflow=True)

    assert result is True


def test_compile_lecun_model():
    config = load_config()
    model = build_lecun_model()

    compile_model(model, "lecun", config)

    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert isinstance(model.loss, keras.losses.MeanSquaredError)
    assert model.count_params() == 2_578


def test_compile_chollet_model():
    config = load_config()
    model = build_chollet_model()

    compile_model(model, "chollet", config)

    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert isinstance(
        model.loss,
        keras.losses.CategoricalCrossentropy,
    )
    assert model.count_params() == 34_826


def test_compile_rejects_unknown_model():
    config = load_config()
    model = build_lecun_model()

    try:
        compile_model(model, "unknown", config)
    except ValueError as error:
        assert "Unknown model name" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown model.")
