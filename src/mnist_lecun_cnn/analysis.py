"""Analysis and visualisation of MNIST CNN comparison experiments."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix


MODEL_NAMES = ("lecun", "chollet")
HISTORY_METRICS = ("loss", "val_loss", "accuracy", "val_accuracy")
NUM_CLASSES = 10


@dataclass(frozen=True)
class PredictionData:
    """Final test predictions for one model."""

    true_labels: NDArray[np.integer]
    predicted_labels: NDArray[np.integer]


@dataclass(frozen=True)
class AnalysisInputs:
    """Validated experiment records, histories, and diagnostic predictions."""

    runs: list[dict[str, Any]]
    histories: dict[str, dict[str, list[float]]]
    predictions: dict[str, PredictionData]


def _load_json(path: Path) -> Any:
    """Load a JSON file and report a useful error if it is missing."""

    if not path.is_file():
        raise FileNotFoundError(f"Required analysis file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_labels(
    labels: NDArray,
    *,
    name: str,
) -> None:
    """Validate a one-dimensional array of MNIST digit labels."""

    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{name} must contain integer labels.")

    if labels.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"{name} must contain labels from 0 to 9.")


def load_analysis_inputs(
    metrics_dir: str | Path,
    diagnostic_metrics_dir: str | Path,
) -> AnalysisInputs:
    """Load and validate experiment histories and diagnostic predictions.

    Args:
        metrics_dir: Directory containing run_records.json and history/.
        diagnostic_metrics_dir: Directory containing predictions/.

    Returns:
        Validated records, training histories, and test predictions.
    """

    metrics_path = Path(metrics_dir)
    diagnostic_path = Path(diagnostic_metrics_dir)

    registry = _load_json(metrics_path / "run_records.json")

    if not isinstance(registry, dict) or not isinstance(
        registry.get("runs"), list
    ):
        raise TypeError("The experiment registry must contain a runs list.")

    runs = registry["runs"]

    if not runs:
        raise ValueError("The experiment registry contains no runs.")

    histories: dict[str, dict[str, list[float]]] = {}

    for run in runs:
        run_id = run.get("run_id")
        model_name = run.get("model")
        history_file = run.get("history_file")

        if not run_id or model_name not in MODEL_NAMES or not history_file:
            raise ValueError(f"Invalid run record: {run!r}")

        if run_id in histories:
            raise ValueError(f"Duplicate run ID: {run_id}")

        history = _load_json(metrics_path / history_file)

        if not isinstance(history, dict):
            raise TypeError(f"Invalid history for run {run_id}.")

        for metric in HISTORY_METRICS:
            values = history.get(metric)

            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Run {run_id} is missing history metric {metric!r}."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Run {run_id} has non-finite values in {metric!r}."
                )

        epoch_counts = {
            len(history[metric]) for metric in HISTORY_METRICS
        }
        if len(epoch_counts) != 1:
            raise ValueError(
                f"History metrics have inconsistent lengths for {run_id}."
            )

        histories[run_id] = history

    for model_name in MODEL_NAMES:
        model_runs = [run for run in runs if run["model"] == model_name]

        if len(model_runs) != 3:
            raise ValueError(
                f"Expected three original runs for {model_name}; "
                f"found {len(model_runs)}."
            )

    predictions: dict[str, PredictionData] = {}

    for model_name in MODEL_NAMES:
        prediction_file = (
            diagnostic_path
            / "predictions"
            / f"{model_name}_seed42_run01_predictions.npz"
        )

        if not prediction_file.is_file():
            raise FileNotFoundError(
                f"Diagnostic predictions not found: {prediction_file}"
            )

        with np.load(prediction_file) as archive:
            required_arrays = {"true_labels", "predicted_labels"}

            if not required_arrays.issubset(archive.files):
                raise ValueError(
                    f"Prediction archive {prediction_file.name} must contain "
                    "'true_labels' and 'predicted_labels'."
                )

            true_labels = archive["true_labels"].copy()
            predicted_labels = archive["predicted_labels"].copy()

        _validate_labels(true_labels, name=f"{model_name} true labels")
        _validate_labels(
            predicted_labels,
            name=f"{model_name} predicted labels",
        )

        if true_labels.shape != predicted_labels.shape:
            raise ValueError(
                f"True and predicted labels have different shapes for "
                f"{model_name}."
            )

        predictions[model_name] = PredictionData(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        )

    lecun_labels = predictions["lecun"].true_labels
    chollet_labels = predictions["chollet"].true_labels

    if not np.array_equal(lecun_labels, chollet_labels):
        raise ValueError(
            "The diagnostic prediction archives use different test labels."
        )

    return AnalysisInputs(
        runs=runs,
        histories=histories,
        predictions=predictions,
    )


def aggregate_histories(
    inputs: AnalysisInputs,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Calculate per-epoch means and sample standard deviations.

    The three independent runs of each model are aggregated separately.
    Standard deviation uses ddof=1 (sample standard deviation).
    """

    aggregated: dict[
        str, dict[str, dict[str, list[float]]]
    ] = {}

    for model_name in MODEL_NAMES:
        model_runs = [
            run for run in inputs.runs
            if run["model"] == model_name
        ]

        model_result: dict[str, dict[str, list[float]]] = {}

        for metric in HISTORY_METRICS:
            values = np.asarray(
                [
                    inputs.histories[run["run_id"]][metric]
                    for run in model_runs
                ],
                dtype=np.float64,
            )

            if values.ndim != 2 or values.shape[0] != 3:
                raise ValueError(
                    f"Expected three equally sized histories for "
                    f"{model_name}/{metric}."
                )

            model_result[metric] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0, ddof=1).tolist(),
            }

        aggregated[model_name] = model_result

    return aggregated


def aggregate_run_metrics(
    inputs: AnalysisInputs,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate final test accuracy and training duration by model."""

    result: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in MODEL_NAMES:
        model_runs = [
            run for run in inputs.runs
            if run["model"] == model_name
        ]

        metrics = {
            "test_accuracy": np.asarray(
                [run["test_accuracy"] for run in model_runs],
                dtype=np.float64,
            ),
            "training_duration_seconds": np.asarray(
                [
                    run["training_duration_seconds"]
                    for run in model_runs
                ],
                dtype=np.float64,
            ),
        }

        result[model_name] = {
            metric: {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
            for metric, values in metrics.items()
        }

    return result


def plot_learning_curve(
    aggregated: dict[str, dict[str, dict[str, list[float]]]],
    output_dir: str | Path,
    show: bool = False,
) -> dict[str, Path]:
    """Create aggregated accuracy and loss learning-curve figures."""

    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}

    # Accuracy comparison: directly comparable across models.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, model_name in zip(axes, MODEL_NAMES, strict=True):
        metrics = aggregated[model_name]

        for metric, label in (
            ("accuracy", "Training"),
            ("val_accuracy", "Validation"),
        ):
            mean = np.asarray(metrics[metric]["mean"])
            std = np.asarray(metrics[metric]["std"])
            epochs = np.arange(1, len(mean) + 1)

            axis.plot(epochs, mean, label=label)
            axis.fill_between(
                epochs,
                mean - std,
                mean + std,
                alpha=0.2,
            )

        axis.set_title(f"{model_name.capitalize()} CNN")
        axis.set_xlabel("Epoch")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[0].set_ylabel("Categorical accuracy")
    fig.suptitle("Training and validation accuracy (mean ± 1 SD)")
    fig.tight_layout()

    accuracy_path = figure_dir / "learning_curves_accuracy.png"
    fig.savefig(accuracy_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    figure_paths["accuracy"] = accuracy_path

    # Loss comparison: separate panels because loss functions differ.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, model_name in zip(axes, MODEL_NAMES, strict=True):
        metrics = aggregated[model_name]

        for metric, label in (
            ("loss", "Training"),
            ("val_loss", "Validation"),
        ):
            mean = np.asarray(metrics[metric]["mean"])
            std = np.asarray(metrics[metric]["std"])
            epochs = np.arange(1, len(mean) + 1)

            axis.plot(epochs, mean, label=label)
            axis.fill_between(
                epochs,
                np.maximum(mean - std, 0),
                mean + std,
                alpha=0.2,
            )

        axis.set_title(f"{model_name.capitalize()} CNN")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("Training and validation loss (mean ± 1 SD)")
    fig.tight_layout()

    loss_path = figure_dir / "learning_curves_loss.png"
    fig.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    figure_paths["loss"] = loss_path

    return figure_paths


def plot_confusion_matrices(
    inputs: AnalysisInputs,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot test-set confusion matrices for both CNN models."""

    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}

    for model_name in MODEL_NAMES:
        prediction = inputs.predictions[model_name]

        matrix = confusion_matrix(
            prediction.true_labels,
            prediction.predicted_labels,
            labels=np.arange(NUM_CLASSES),
        )

        fig, axis = plt.subplots(figsize=(9, 8))

        image = axis.imshow(matrix, interpolation="nearest")
        fig.colorbar(image, ax=axis, label="Number of images")

        axis.set(
            xticks=np.arange(NUM_CLASSES),
            yticks=np.arange(NUM_CLASSES),
            xlabel="Predicted label",
            ylabel="True label",
            title=f"{model_name.capitalize()} CNN — Test Confusion Matrix",
        )

        threshold = matrix.max() / 2

        for row in range(NUM_CLASSES):
            for column in range(NUM_CLASSES):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}",
                    ha="center",
                    va="center",
                    color="white"
                    if matrix[row,  column] > threshold
                    else "black",
                )

        fig.tight_layout()

        figure_path = (
            figure_dir / f"confusion_matrix_{model_name}.png"
        )

        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        figure_paths[model_name] = figure_path

    return figure_paths


def analyse_class_performance(
    inputs: AnalysisInputs,
) -> dict[str, dict[str, Any]]:
    """Calculate per-class performance and common prediction errors."""

    results: dict[str, dict[str, Any]] = {}

    for model_name in MODEL_NAMES:
        prediction = inputs.predictions[model_name]

        matrix = confusion_matrix(
            prediction.true_labels,
            prediction.predicted_labels,
            labels=np.arange(NUM_CLASSES),
        )

        report = classification_report(
            prediction.true_labels,
            prediction.predicted_labels,
            labels=np.arange(NUM_CLASSES),
            output_dict=True,
            zero_division=0,
        )

        class_metrics = {}

        for digit in range(NUM_CLASSES):
            digit_report = report[str(digit)]

            class_metrics[str(digit)] = {
                "precision": float(digit_report["precision"]),
                "recall": float(digit_report["recall"]),
                "f1_score": float(digit_report["f1-score"]),
                "support": int(digit_report["support"]),
                "errors": int(
                    matrix[digit, :].sum() - matrix[digit, digit]
                ),
            }

        confusion_pairs = []

        for true_digit in range(NUM_CLASSES):
            for predicted_digit in range(NUM_CLASSES):
                if true_digit == predicted_digit:
                    continue

                count = int(matrix[true_digit, predicted_digit])

                if count > 0:
                    confusion_pairs.append(
                        {
                            "true_digit": true_digit,
                            "predicted_digit": predicted_digit,
                            "count": count,
                        }
                    )

        confusion_pairs.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        results[model_name] = {
            "accuracy": float(report["accuracy"]),
            "macro_avg": {
                "precision": float(report["macro avg"]["precision"]),
                "recall": float(report["macro avg"]["recall"]),
                "f1_score": float(report["macro avg"]["f1-score"]),
            },
            "weighted_avg": {
                "precision": float(report["weighted avg"]["precision"]),
                "recall": float(report["weighted avg"]["recall"]),
                "f1_score": float(report["weighted avg"]["f1-score"]),
            },
            "per_class": class_metrics,
            "top_confusions": confusion_pairs[:10],
        }

    return results


def plot_misclassified_examples(
    inputs: AnalysisInputs,
    test_images: NDArray,
    output_dir: str | Path,
    *,
    examples_per_model: int = 20,
) -> dict[str, Path]:
    """Plot representative misclassified test images for both models."""

    if test_images.ndim != 3 or test_images.shape[1:] != (28, 28):
        raise ValueError("test_images must have shape (n, 28, 28).")

    if examples_per_model < 1:
        raise ValueError("examples_per_model must be positive.")

    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}

    for model_name in MODEL_NAMES:
        prediction = inputs.predictions[model_name]

        if len(test_images) != len(prediction.true_labels):
            raise ValueError(
                f"Test images and predictions have different lengths "
                f"for {model_name}."
            )

        error_indices = np.flatnonzero(
            prediction.true_labels != prediction.predicted_labels
        )

        if error_indices.size == 0:
            raise ValueError(
                f"No misclassified examples found for {model_name}."
            )

        selected_indices = np.linspace(
            0,
            error_indices.size - 1,
            num=min(examples_per_model, error_indices.size),
            dtype=int,
        )

        selected_indices = error_indices[selected_indices]

        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        axes_flat = axes.ravel()

        for axis, index in zip(
            axes_flat,
            selected_indices,
            strict=False,
        ):
            axis.imshow(test_images[index], cmap="gray")
            axis.set_title(
                f"True: {prediction.true_labels[index]} | "
                f"Pred: {prediction.predicted_labels[index]}",
                fontsize=10,
            )
            axis.axis("off")

        for axis in axes_flat[len(selected_indices):]:
            axis.axis("off")

        fig.suptitle(
            f"{model_name.capitalize()} CNN — Misclassified Test Images",
            fontsize=14,
        )
        fig.tight_layout()

        figure_path = (
            figure_dir / f"misclassified_examples_{model_name}.png"
        )

        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        figure_paths[model_name] = figure_path

    return figure_paths
