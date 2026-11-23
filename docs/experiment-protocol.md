# Experimental Protocol and Execution Record

## 1. Purpose

This document records the procedure used to compare the LeCun-inspired CNN and the Chollet Keras MNIST baseline. It describes dataset handling, preprocessing, training, evaluation, timing, repeated runs and quality controls.

## 2. Experimental design

The experiment compares two complete model pipelines. Both use the same MNIST source, training/validation/test partitions, number of epochs, batch size, optimizer family, machine and final evaluation dataset. Their architecture, preprocessing, target encoding, output activation and loss functions differ by design.

The results therefore do not isolate architecture as the only independent variable.

## 3. Dataset and partitioning

MNIST is loaded using `tf.keras.datasets.mnist.load_data()`.

| Dataset component | Shape           |
| ----------------- | --------------- |
| Training images   | (60000, 28, 28) |
| Training labels   | (60000,)        |
| Test images       | (10000, 28, 28) |
| Test labels       | (10000,)        |

The 60,000 training examples are stratified into 54,000 training and 6,000 validation samples using split seed 42. Both models use the same partition. The official 10,000-image test set remains held out for final evaluation.

The source arrays are retained as the original dataset representation; model-specific transformations are applied through separate preprocessing functions.

## 4. Preprocessing

### 4.1 LeCun-inspired pipeline

- Identify nonzero foreground pixels and determine the bounding box.
- Crop the digit.
- Resize proportionally to fit a 16 × 16 region using bilinear interpolation.
- Apply half-up dimension rounding and integer-floor centring on a 28 × 28 canvas.
- Use −1.0 for an empty-image canvas.
- Normalize with `pixel / 127.5 - 1.0`.
- Add a channel dimension.

### 4.2 Chollet pipeline

- Convert images to `float32`.
- Add a channel dimension.
- Normalize with `pixel / 255.0`.
- Encode labels as ten-class one-hot vectors.

### 4.3 Validation

Preprocessing checks cover expected shapes, data types, finite values, value ranges, label representation and preservation of image/label correspondence. Empty foreground handling is tested for the LeCun-inspired pipeline.

## 5. Model and training configuration

The machine-readable configuration is `configs/experiment.json`.

| Setting                             | Value                   |
| ----------------------------------- | ----------------------- |
| Training seeds                      | 42, 43, 44              |
| Repetitions                         | 3 per model             |
| Split seed                          | 42                      |
| Training samples                    | 54,000                  |
| Validation samples                  | 6,000                   |
| Test samples                        | 10,000                  |
| Epochs                              | 15                      |
| Batch size                          | 128                     |
| Optimizer                           | Adam                    |
| Learning rate                       | 0.001                   |
| TensorFlow deterministic operations | Enabled where supported |

Model-specific configuration:

| Setting              | LeCun-inspired CNN | Chollet CNN              |
| -------------------- | ------------------ | ------------------------ |
| Trainable parameters | 2,578              | 34,826                   |
| Output               | 10 tanh units      | 10 softmax units         |
| Targets              | Bipolar            | One-hot                  |
| Loss                 | Mean squared error | Categorical crossentropy |
| Dropout              | None               | 0.5                      |

Each run initializes a new model. No weights are carried between runs. The final epoch weights are evaluated on the official test set; validation performance is monitored during training.

## 6. Training-time measurement

Training duration is recorded in seconds for the model-fitting procedure. It excludes dataset downloading, preprocessing, model construction and compilation. The same computer and software environment were used for both models.

The recorded environment was Windows 11, Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.1 and NumPy 2.5.5. TensorFlow reported CPU execution and no available GPU. Timing is hardware- and environment-dependent.

**Important data reconciliation note:** the repository contains two run-record collections:

- `outputs/metrics/run_records.json`
- `notebooks/outputs/notebook_reproduction/run_records.json`

The recorded test accuracies and losses agree, but the training durations differ. The notebook's aggregate table is calculated from the second collection. These collections must not be mixed. Before reporting a single timing mean and standard deviation in the assignment report, identify which execution is authoritative and align the notebook, logs and report with that source. Do not overwrite records merely to make the values agree.

## 7. Evaluation metrics

### 7.1 Accuracy

Accuracy is the proportion of correctly classified test images:

\[
\mathrm{Accuracy} =
\frac{\mathrm{Correct\ predictions}}{\mathrm{Total\ predictions}}
\]

For each image, the predicted class is the index of the largest output value.

### 7.2 Test loss

The LeCun-inspired model uses MSE with bipolar targets. The Chollet model uses categorical crossentropy with one-hot targets. Their numerical loss values have different meanings and must not be treated as directly comparable.

### 7.3 Training time and parameter count

Record the model-fitting duration and Keras trainable parameter count for each run. The expected parameter counts are 2,578 and 34,826, respectively.

## 8. Diagnostic analysis

The completed analysis includes:

- Training and validation loss/accuracy curves aggregated across repeated runs.
- Test confusion matrices.
- Selected misclassified test examples with true and predicted labels.
- Per-class precision, recall, F1-score, support and error counts.

Confusion matrices, misclassified examples and per-class metrics in the notebook are based on the representative seed-42 run. The aggregate accuracy table summarizes all three runs. These are distinct levels of analysis and should be labelled accordingly.

## 9. Repeated-run aggregation

For three runs, report the mean and sample standard deviation (`ddof=1`) for test accuracy, test loss and training duration. Preserve individual run records and histories.

The notebook's aggregate test results are:

| Metric        | LeCun-inspired CNN |       Chollet CNN |
| ------------- | -----------------: | ----------------: |
| Test accuracy |   97.477% ± 0.071% |  99.117% ± 0.042% |
| Test loss     |  0.02065 ± 0.00059 | 0.02609 ± 0.00182 |

The training-time aggregate is intentionally not repeated here because the repository currently retains two different timing-record sets. Reconcile those records before using a timing figure in the final report.

Three runs provide descriptive evidence only; no statistical-significance claim is made.

## 10. Run records and outputs

Each run has a unique identifier such as `lecun_seed42_run01` or `chollet_seed44_run03`. Records include model, seed, timestamps, duration, epochs, batch size, parameter count, final training/validation loss, test loss, test accuracy, software, hardware and history-file reference.

| Location                                   | Contents                                                   |
| ------------------------------------------ | ---------------------------------------------------------- |
| `outputs/metrics/`                         | Project-level run records and histories                    |
| `outputs/logs/`                            | Experiment execution log                                   |
| `outputs/figures/`                         | Project-level figures                                      |
| `outputs/diagnostics/`                     | Experiment diagnostics                                     |
| `notebooks/outputs/notebook_reproduction/` | Notebook run records, histories and prediction diagnostics |

## 11. Quality assurance record

The repository includes automated tests for data handling, preprocessing, model construction, training and analysis. The recorded experiment results verify the expected model parameter counts and include six completed runs.

## 12. Limitations

- Model-specific preprocessing and loss functions mean this is a comparison of complete pipelines.
- The LeCun-inspired network is not an exact historical reproduction.
- Three runs are insufficient to support broad statistical claims.
- Training-time measurements apply to the recorded CPU environment only.
- Confusion matrices and detailed per-class results are representative single-run analyses, not three-run averages.
