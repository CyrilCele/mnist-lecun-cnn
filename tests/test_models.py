
"""Architecture and output tests for both CNN models."""

import numpy as np

from mnist_lecun_cnn.models.chollet import build_chollet_model
from mnist_lecun_cnn.models.lecun import build_lecun_model


def test_lecun_model_has_expected_architecture():
    model = build_lecun_model()

    assert model.output_shape == (None, 10)
    assert model.count_params() == 2_578
    assert sum(
        int(variable.shape.num_elements())
        for variable in model.trainable_variables
    ) == 2_578

    expected_shapes = {
        "h1_convolution": (None, 24, 24, 4),
        "h2_trainable_subsampling": (None, 12, 12, 4),
        "h3_feature_maps": (None, 8, 8, 12),
        "h4_trainable_subsampling": (None, 4, 4, 12),
    }

    for layer_name, expected_shape in expected_shapes.items():
        assert model.get_layer(layer_name).output.shape == expected_shape


def test_chollet_model_has_expected_architecture():
    model = build_chollet_model()

    assert model.output_shape == (None, 10)
    assert model.count_params() == 34_826
    assert sum(
        int(variable.shape.num_elements())
        for variable in model.trainable_variables
    ) == 34_826

    expected_parameters = {
        "conv1": 320,
        "conv2": 18_496,
        "output": 16_010,
    }

    for layer_name, expected in expected_parameters.items():
        assert model.get_layer(layer_name).count_params() == expected


def test_lecun_model_produces_valid_outputs():
    model = build_lecun_model()
    images = np.zeros((2, 28, 28, 1), dtype=np.float32)

    predictions = model(images, training=False).numpy()

    assert predictions.shape == (2, 10)
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= -1.0)
    assert np.all(predictions <= 1.0)


def test_chollet_model_produces_valid_probabilities():
    model = build_chollet_model()
    images = np.zeros((2, 28, 28, 1), dtype=np.float32)

    predictions = model(images, training=False).numpy()

    assert predictions.shape == (2, 10)
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
    np.testing.assert_allclose(
        predictions.sum(axis=1),
        np.ones(2),
        rtol=1e-5,
        atol=1e-5,
    )
