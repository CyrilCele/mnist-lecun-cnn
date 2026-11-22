
"""Tests for MNIST experiment analysis and visualisation."""

import numpy as np
import pytest

from mnist_lecun_cnn.analysis import (
    AnalysisInputs,
    PredictionData,
    aggregate_histories,
    aggregate_run_metrics,
    analyse_class_performance,
    plot_confusion_matrices,
    plot_learning_curve,
    plot_misclassified_examples,
)


@pytest.fixture
def analysis_inputs() -> AnalysisInputs:
    """Create synthetic results for three runs of each model."""

    runs = []
    histories = {}

    for model_index, model_name in enumerate(("lecun", "chollet")):
        for run_index, seed in enumerate((42, 43, 44)):
            run_id = f"{model_name}_seed{seed}_run{run_index + 1}"

            runs.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "test_accuracy": (
                        0.95 + model_index * 0.02 + run_index * 0.001
                    ),
                    "training_duration_seconds": (
                        200.0 + model_index * 50 + run_index * 10
                    ),
                }
            )

            histories[run_id] = {
                "loss": [1.0 - run_index * 0.1, 0.5 - run_index * 0.1],
                "val_loss": [
                    1.1 - run_index * 0.1,
                    0.6 - run_index * 0.1,
                ],
                "accuracy": [
                    0.8 + run_index * 0.01,
                    0.9 + run_index * 0.01,
                ],
                "val_accuracy": [
                    0.79 + run_index * 0.01,
                    0.89 + run_index * 0.01,
                ],
            }

    true_labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)

    predictions = {
        "lecun": PredictionData(
            true_labels=true_labels,
            predicted_labels=np.array(
                [0, 1, 2, 3, 5, 5],
                dtype=np.int64,
            ),
        ),
        "chollet": PredictionData(
            true_labels=true_labels,
            predicted_labels=np.array(
                [0, 1, 2, 3, 4, 5],
                dtype=np.int64,
            ),
        ),
    }

    return AnalysisInputs(
        runs=runs,
        histories=histories,
        predictions=predictions,
    )


def test_aggregate_histories_calculates_mean_and_sample_std(
    analysis_inputs: AnalysisInputs,
) -> None:
    """History aggregation should calculate per-epoch statistics."""

    result = aggregate_histories(analysis_inputs)

    assert set(result) == {"lecun", "chollet"}

    values = np.array(
        [
            [1.0, 0.5],
            [0.9, 0.4],
            [0.8, 0.3],
        ]
    )

    np.testing.assert_allclose(
        result["lecun"]["loss"]["mean"],
        values.mean(axis=0),
    )

    np.testing.assert_allclose(
        result["lecun"]["loss"]["std"],
        values.std(axis=0, ddof=1),
    )


def test_aggregate_run_metrics_calculates_mean_and_sample_std(
    analysis_inputs: AnalysisInputs,
) -> None:
    """Final metrics should be aggregated separately by model."""

    result = aggregate_run_metrics(analysis_inputs)

    assert set(result) == {"lecun", "chollet"}

    expected_accuracy = np.array([0.95, 0.951, 0.952])

    assert result["lecun"]["test_accuracy"]["mean"] == pytest.approx(
        expected_accuracy.mean()
    )

    assert result["lecun"]["test_accuracy"]["std"] == pytest.approx(
        expected_accuracy.std(ddof=1)
    )

    assert result["chollet"]["test_accuracy"]["mean"] > (
        result["lecun"]["test_accuracy"]["mean"]
    )


def test_plot_learning_curve_creates_both_figures(
    analysis_inputs: AnalysisInputs,
    tmp_path,
) -> None:
    """Learning-curve plotting should save accuracy and loss figures."""

    aggregated = aggregate_histories(analysis_inputs)

    paths = plot_learning_curve(aggregated, tmp_path)

    assert set(paths) == {"accuracy", "loss"}

    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_plot_confusion_matrices_creates_both_figures(
    analysis_inputs: AnalysisInputs,
    tmp_path,
) -> None:
    """Confusion-matrix plotting should process both model predictions."""

    paths = plot_confusion_matrices(analysis_inputs, tmp_path)

    assert set(paths) == {"lecun", "chollet"}

    for model_name, path in paths.items():
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.name == f"confusion_matrix_{model_name}.png"


def test_analyse_class_performance_reports_class_metrics(
    analysis_inputs: AnalysisInputs,
) -> None:
    """Class-wise metrics and confusion pairs should be calculated."""

    results = analyse_class_performance(analysis_inputs)

    assert set(results) == {"lecun", "chollet"}

    lecun = results["lecun"]

    assert lecun["accuracy"] == pytest.approx(5 / 6)
    assert lecun["per_class"]["4"]["errors"] == 1
    assert lecun["per_class"]["4"]["recall"] == pytest.approx(0.0)

    assert lecun["top_confusions"][0] == {
        "true_digit": 4,
        "predicted_digit": 5,
        "count": 1,
    }

    chollet = results["chollet"]

    assert chollet["accuracy"] == pytest.approx(1.0)
    assert chollet["top_confusions"] == []


def test_plot_misclassified_examples_creates_both_figures(
    analysis_inputs: AnalysisInputs,
    tmp_path,
) -> None:
    """Misclassification plotting should save figures for both models."""

    true_labels = analysis_inputs.predictions["chollet"].true_labels
    chollet_predictions = np.array(
        [1, 1, 2, 3, 4, 5],
        dtype=np.int64,
    )

    inputs = AnalysisInputs(
        runs=analysis_inputs.runs,
        histories=analysis_inputs.histories,
        predictions={
            **analysis_inputs.predictions,
            "chollet": PredictionData(
                true_labels=true_labels,
                predicted_labels=chollet_predictions,
            ),
        },
    )

    test_images = np.random.default_rng(42).random((6, 28, 28))

    paths = plot_misclassified_examples(
        inputs,
        test_images,
        tmp_path,
        examples_per_model=5,
    )

    assert set(paths) == {"lecun", "chollet"}

    for model_name, path in paths.items():
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.name == f"misclassified_examples_{model_name}.png"


def test_plot_misclassified_examples_rejects_invalid_image_shape(
    analysis_inputs: AnalysisInputs,
    tmp_path,
) -> None:
    """Misclassification plotting should reject incorrectly shaped images."""

    test_images = np.zeros((6, 28, 28, 1))

    with pytest.raises(ValueError, match="shape"):
        plot_misclassified_examples(
            analysis_inputs,
            test_images,
            tmp_path,
        )
