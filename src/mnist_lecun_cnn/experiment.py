"""Orchestration and result recording for the MNIST CNN comparison."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mnist_lecun_cnn.data import MNISTDataset, load_mnist_dataset
from mnist_lecun_cnn.models.chollet import build_chollet_model
from mnist_lecun_cnn.models.lecun import build_lecun_model
from mnist_lecun_cnn.preprocessing import (
    encode_chollet_targets,
    encode_lecun_targets,
    preprocess_chollet_images,
    preprocess_lecun_images,
)
from mnist_lecun_cnn.training import (
    compile_model,
    set_random_seeds,
    train_model,
)


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load the experiment configuration from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable hash of the experiment configuration."""

    serialized = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically to avoid leaving a partial result file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)

    temporary_path.replace(path)


def _save_predictions(
    path: Path,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Save true and predicted calss labels atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        )

    temporary_path.replace(path)


def _hardware_information() -> dict[str, Any]:
    """Collect basic platform and TensorFlow device information."""

    gpu_devices = tf.config.list_physical_devices("GPU")

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "Not reported",
        "tensorflow_devices": [
            device.name
            for device in tf.config.list_physical_devices()
        ],
        "gpu_devices": [device.name for device in gpu_devices],
        "gpu_available": bool(gpu_devices),
    }


def _prepare_model_data(
    dataset: MNISTDataset,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training, validation, and test arrays for one model."""

    train_indices = dataset.split.train_indices
    validation_indices = dataset.split.validation_indices

    if model_name == "lecun":
        preprocess = preprocess_lecun_images
        encode = encode_lecun_targets
    elif model_name == "chollet":
        preprocess = preprocess_chollet_images
        encode = encode_chollet_targets
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    x_train = preprocess(dataset.x_train[train_indices])
    y_train = encode(dataset.y_train[train_indices])

    x_validation = preprocess(dataset.x_train[validation_indices])
    y_validation = encode(dataset.y_train[validation_indices])

    x_test = preprocess(dataset.x_test)
    y_test = encode(dataset.y_test)

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    )


def _build_model(model_name: str) -> keras.Model:
    """Build a fresh instance of the requested model."""

    if model_name == "lecun":
        return build_lecun_model()

    if model_name == "chollet":
        return build_chollet_model()

    raise ValueError(f"Unknown model name: {model_name}")


def run_experiments(
    config_path: Path = Path("configs/experiment.json"),
    *,
    output_dir: Path | None = None,
    model_names: tuple[str, ...] = ("lecun", "chollet"),
    verbose: int = 2,
    dataset: MNISTDataset | None = None,
) -> list[dict[str, Any]]:
    """Run all configured model repetitions and save their results."""

    config = load_experiment_config(config_path)
    fingerprint = _config_fingerprint(config)

    if output_dir is None:
        output_dir = Path(config["outputs"]["metrics"])

    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    registry_path = output_dir / "run_records.json"

    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

        if registry.get("config_sha256") != fingerprint:
            raise ValueError(
                "Existing results were produced with a different "
                "configuration. Use a new output directory."
            )
    else:
        registry = {
            "experiment_name": config["experiment"]["name"],
            "config_sha256": fingerprint,
            "runs": [],
        }

    completed_ids = {run["run_id"] for run in registry["runs"]}
    seeds = config["experiment"]["training_seeds"]
    repetitions = config["experiment"]["repetitions"]

    if len(seeds) != repetitions:
        raise ValueError(
            "The number of training seeds must equal the repetitions."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique.")

    if not model_names or any(
        name not in ("lecun", "chollet") for name in model_names
    ):
        raise ValueError(
            "model_names must contain 'lecun' and/or 'chollet'."
        )

    if len(set(model_names)) != len(model_names):
        raise ValueError("model_names must not contain duplicates.")

    if dataset is None:
        dataset = load_mnist_dataset(
            random_seed=config["experiment"]["split_seed"],
        )

    all_results = list(registry["runs"])

    for model_name in model_names:
        print(f"\nPreparing data for {model_name} model...")
        model_data = _prepare_model_data(dataset, model_name)

        (
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
        ) = model_data

        for run_number, seed in enumerate(seeds, start=1):
            run_id = f"{model_name}_seed{seed}_run{run_number:02d}"

            if run_id in completed_ids:
                print(f"\nSkipping completed run: {run_id}")
                continue

            print(f"\nStarting {run_id}")
            started_at = datetime.now(UTC)

            keras.backend.clear_session()
            deterministic_enabled = set_random_seeds(
                seed,
                deterministic_tensorflow=config["experiment"][
                    "deterministic_tensorflow"
                ],
            )

            model = _build_model(model_name)
            compile_model(model, model_name, config)

            history, training_duration = train_model(
                model,
                x_train,
                y_train,
                x_validation,
                y_validation,
                epochs=config["experiment"]["epochs"],
                batch_size=config["experiment"]["batch_size"],
                shuffle=config["experiment"]["shuffle"],
                verbose=verbose,
            )

            test_metrics = model.evaluate(
                x_test,
                y_test,
                verbose=0,
                return_dict=True,
            )

            predictions = model.predict(x_test, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = dataset.y_test
            test_accuracy = float(
                np.mean(predicted_labels == true_labels)
            )

            keras_accuracy = float(test_metrics["accuracy"])

            if not np.isclose(test_accuracy, keras_accuracy, atol=1e-6):
                raise RuntimeError(
                    "Explicit test accuracy does not match Keras accuracy."
                )

            predictions_file = f"predictions/{run_id}_predictions.npz"

            _save_predictions(
                output_dir / predictions_file,
                true_labels,
                predicted_labels,
            )

            ended_at = datetime.now(UTC)

            history_data = {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            }

            _save_json(
                history_dir / f"{run_id}_history.json",
                history_data,
            )

            result = {
                "run_id": run_id,
                "model": model_name,
                "seed": int(seed),
                "run_number": run_number,
                "started_at_utc": started_at.isoformat(),
                "ended_at_utc": ended_at.isoformat(),
                "training_duration_seconds": float(training_duration),
                "epochs_completed": len(history.epoch),
                "batch_size": config["experiment"]["batch_size"],
                "trainable_parameters": int(model.count_params()),
                "final_training_loss": float(
                    history.history["loss"][-1]
                ),
                "final_validation_loss": float(
                    history.history["val_loss"][-1]
                ),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": test_accuracy,
                "keras_test_accuracy": keras_accuracy,
                "deterministic_tensorflow_enabled": deterministic_enabled,
                "split_seed": config["experiment"]["split_seed"],
                "config_sha256": fingerprint,
                "software": {
                    "python": sys.version,
                    "tensorflow": tf.__version__,
                    "numpy": np.__version__,
                    "keras": keras.__version__,
                },
                "hardware": _hardware_information(),
                "history_file": f"history/{run_id}_history.json",
                "predictions_file": predictions_file,
            }

            registry["runs"].append(result)
            _save_json(registry_path, registry)

            completed_ids.add(run_id)
            all_results.append(result)

            print(
                f"Completed {run_id}: "
                f"test accuracy={test_accuracy:.4f}, "
                f"test loss={float(test_metrics['loss']):.4f}, "
                f"training time={training_duration:.2f}s"
            )

            del model

        del model_data

    return all_results
