
"""Tests for experiment orchestration and result recording."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mnist_lecun_cnn.data import DatasetSplit, MNISTDataset
from mnist_lecun_cnn.experiment import run_experiments


class FakeModel:
    """Small stand-in for a compiled Keras model."""

    def count_params(self):
        return 34_826

    def evaluate(self, x, y, verbose=0, return_dict=False):
        return {"loss": 0.25, "accuracy": 1.0}

    def predict(self, x, verbose=0):
        predictions = np.zeros((len(x), 10), dtype=np.float32)
        predictions[:, 0] = 1.0
        return predictions


@pytest.fixture
def experiment_setup(tmp_path, monkeypatch):
    """Create a small dataset and a controlled one-run configuration."""

    config = {
        "experiment": {
            "name": "test_experiment",
            "repetitions": 1,
            "training_seeds": [42],
            "split_seed": 42,
            "validation_size": 1,
            "epochs": 1,
            "batch_size": 2,
            "shuffle": True,
            "deterministic_tensorflow": True,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
        },
        "models": {
            "chollet": {
                "expected_parameters": 34_826,
                "loss": "categorical_crossentropy",
            }
        },
        "outputs": {"metrics": "outputs/metrics"},
    }

    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    x_train = np.zeros((3, 28, 28), dtype=np.uint8)
    y_train = np.zeros(3, dtype=np.uint8)
    x_test = np.zeros((2, 28, 28), dtype=np.uint8)
    y_test = np.zeros(2, dtype=np.uint8)

    dataset = MNISTDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        split=DatasetSplit(
            train_indices=np.array([0, 1], dtype=np.int64),
            validation_indices=np.array([2], dtype=np.int64),
        ),
    )

    calls = {"training": 0}

    def fake_train_model(*args, **kwargs):
        calls["training"] += 1
        history = SimpleNamespace(
            history={
                "loss": [0.5],
                "accuracy": [1.0],
                "val_loss": [0.4],
                "val_accuracy": [1.0],
            },
            epoch=[0],
        )
        return history, 1.25

    monkeypatch.setattr(
        "mnist_lecun_cnn.experiment._build_model",
        lambda model_name: FakeModel(),
    )
    monkeypatch.setattr(
        "mnist_lecun_cnn.experiment.compile_model",
        lambda model, model_name, config: model,
    )
    monkeypatch.setattr(
        "mnist_lecun_cnn.experiment.set_random_seeds",
        lambda seed, deterministic_tensorflow=True: True,
    )
    monkeypatch.setattr(
        "mnist_lecun_cnn.experiment.train_model",
        fake_train_model,
    )

    return config_path, dataset, tmp_path / "metrics", calls


def test_experiment_saves_results_and_skips_completed_run(
    experiment_setup,
):
    config_path, dataset, output_dir, calls = experiment_setup

    first_results = run_experiments(
        config_path,
        output_dir=output_dir,
        model_names=("chollet",),
        verbose=0,
        dataset=dataset,
    )

    assert len(first_results) == 1
    assert calls["training"] == 1

    registry_path = output_dir / "run_records.json"
    history_path = (
        output_dir / "history" / "chollet_seed42_run01_history.json"
    )

    assert registry_path.is_file()
    assert history_path.is_file()

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert len(registry["runs"]) == 1
    assert registry["runs"][0]["run_id"] == "chollet_seed42_run01"
    assert registry["runs"][0]["test_accuracy"] == 1.0
    assert registry["runs"][0]["training_duration_seconds"] == 1.25
    assert history["loss"] == [0.5]
    assert history["val_loss"] == [0.4]

    second_results = run_experiments(
        config_path,
        output_dir=output_dir,
        model_names=("chollet",),
        verbose=0,
        dataset=dataset,
    )

    assert len(second_results) == 1
    assert calls["training"] == 1

    predictions_path = (
        output_dir / "predictions" / "chollet_seed42_run01_predictions.npz"
    )

    assert predictions_path.is_file()

    with np.load(predictions_path) as predictions:
        np.testing.assert_array_equal(
            predictions["true_labels"],
            dataset.y_test,
        )
        np.testing.assert_array_equal(
            predictions["predicted_labels"],
            np.array([0, 0]),
        )

    assert (
        registry["runs"][0]["predictions_file"]
        == "predictions/chollet_seed42_run01_predictions.npz"
    )


def test_experiment_rejects_changed_configuration(experiment_setup):
    config_path, dataset, output_dir, _ = experiment_setup

    run_experiments(
        config_path,
        output_dir=output_dir,
        model_names=("chollet",),
        verbose=0,
        dataset=dataset,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"]["epochs"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="different configuration"):
        run_experiments(
            config_path,
            output_dir=output_dir,
            model_names=("chollet",),
            verbose=0,
            dataset=dataset,
        )


def test_experiment_rejects_duplicate_training_seeds(experiment_setup):
    config_path, dataset, output_dir, _ = experiment_setup

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"]["repetitions"] = 2
    config["experiment"]["training_seeds"] = [42, 42]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Training seeds must be unique"):
        run_experiments(
            config_path,
            output_dir=output_dir,
            model_names=("chollet",),
            verbose=0,
            dataset=dataset,
        )


def test_experiment_rejects_duplicate_model_names(experiment_setup):
    config_path, dataset, output_dir, _ = experiment_setup

    with pytest.raises(ValueError, match="model_names must not contain duplicates"):
        run_experiments(
            config_path,
            output_dir=output_dir,
            model_names=("chollet", "chollet"),
            verbose=0,
            dataset=dataset,
        )
