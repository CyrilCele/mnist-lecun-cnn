# Experimental Protocol

## 1. Purpose

This document defines the procedure for conducting reproducible experiments comparing:

- Model A: LeCun-inspired CNN.
- Model B: Chollet Keras MNIST CNN baseline.

The protocol establishes dataset handling, preprocessing, data partitioning, training configuration, evaluation metrics, timing procedures, experiment records and quality controls.

The protocol must be followed consistently. Any deviation must be documented with its reason and the affected experiments.

## 2. Experimental objectives

The experiments aim to:

1. Measure the classification performance of both models on MNIST.
2. Compare their test accuracy and test loss.
3. Measure and compare their training times.
4. Compare their trainable parameter counts.
5. Examine their learning behaviour using training and validation curves.
6. Investigate classification errors using confusion matrices and selected examples.
7. Interpret results in relation to architectural and preprocessing differences.

## 3. Experimental design

### 3.1 Comparison approach

The project uses a comparative experimental design.

Both models will use:

- The same MNIST source dataset.
- Identical training, validation and test sample indices.
- The same number of training epochs.
- The same batch size.
- The same hardware and software environment.
- Recorded random seeds.
- The same final evaluation procedure.

Model-specific architectural, preprocessing, activation, label encoding, loss and optimization choices remain as specified in the technical specification.

Because these model-specific choices differ, the experiment compares complete model pipelines rather than isolating architecture as the only independent variable.

### 3.2 Experimental variables

**Controlled variables**

- Dataset source.
- Dataset partition.
- Training sample order policy.
- Number of epochs.
- Batch size.
- Hardware.
- Software environment.
- Evaluation dataset.

**Model-specific variables**

- Network architecture.
- Number of trainable parameters.
- Input normalization.
- Geometric preprocessing.
- Target encoding.
- Output activation.
- Loss function.

**Measured outcomes**

- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.
- Test loss.
- Test accuracy.
- Training time.
- Trainable parameter count.

## 4. Dataset loading and validation

### 4.1 Dataset source

Load MNIST using the TensorFlow/Keras dataset loader:

`tf.keras.datasets.mnist.load_data()`

The expected data dimensions are:

| Dataset component | Expected shape |
|---|---|
| Training images | (60000, 28, 28) |
| Training labels | (60000,) |
| Test images | (10000, 28, 28) |
| Test labels | (10000,) |

### 4.2 Data integrity checks

Before preprocessing, verify:

- Training image count is 60,000.
- Test image count is 10,000.
- Image dimensions are 28 × 28.
- Images contain one grayscale channel implicitly.
- Pixel values are within 0–255.
- Labels are integers from 0 through 9.
- There are no unexpected NaN or infinite values.
- Label distributions are recorded.

The original loaded arrays must not be modified in place.

## 5. Dataset partitioning

### 5.1 Training and validation split

Use the original 60,000 training examples to create:

| Subset | Number of samples |
|---|---:|
| Training | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

Use a fixed random seed and stratified sampling to preserve the approximate class distribution.

The split must be generated once and reused for both models and all repeated runs.

### 5.2 Test-set isolation

The 10,000 official MNIST test examples must remain independent of training and model selection.

The test set must not be used to:

- Select model architecture.
- Tune hyperparameters.
- Select preprocessing parameters.
- Select training epochs.
- Choose between checkpoints.

Test evaluation occurs only after the training procedure has been completed.

## 6. Preprocessing procedures

### 6.1 Model A: LeCun-inspired preprocessing

Apply the procedure defined in the technical specification:

1. Identify the foreground bounding box.
2. Crop the digit.
3. Resize it proportionally to fit within 16 × 16 pixels.
4. Place it on a centred 28 × 28 canvas.
5. Normalize pixel values to [−1, +1].
6. Add the channel dimension.

The following implementation details must be fixed and recorded:

- Foreground threshold.
- Bounding-box boundary handling.
- Resize interpolation method.
- Rounding behaviour.
- Centring procedure.
- Empty-image handling.
- Pixel normalization formula.

An all-background image must be handled safely without division-by-zero or invalid dimensions.

### 6.2 Model B: Chollet preprocessing

Apply the standard baseline preprocessing:

1. Convert images to float32.
2. Add a channel dimension.
3. Divide pixel intensities by 255.
4. Convert labels into ten-class one-hot vectors.

The expected image shape is:

`(number_of_samples, 28, 28, 1)`

The expected label shape is:

`(number_of_samples, 10)`

### 6.3 Preprocessing validation

For both pipelines, verify:

- Expected output dimensions.
- Expected data type.
- Finite values.
- Correct value ranges.
- Correct label representations.
- Correct correspondence between each image and its original label.

Preprocessing must be deterministic for a given input and configuration.

## 7. Model configurations

### 7.1 Model A

| Parameter | Value |
|---|---|
| Architecture | LeCun-inspired CNN |
| Input | 28 × 28 × 1 |
| Output | 10 tanh units |
| Target encoding | Bipolar |
| Loss | Mean squared error |
| Optimizer | Adam |
| Epochs | 15 |
| Batch size | 128 |
| Expected trainable parameters | 2,578 |

### 7.2 Model B

| Parameter | Value |
|---|---|
| Architecture | Chollet Keras CNN |
| Input | 28 × 28 × 1 |
| Output | 10 softmax units |
| Target encoding | One-hot |
| Loss | Categorical crossentropy |
| Optimizer | Adam |
| Epochs | 15 |
| Batch size | 128 |
| Dropout | 0.5 |
| Expected trainable parameters | 34,826 |

### 7.3 Configuration management

The experiment settings must be stored in a machine-readable configuration file.

The configuration should include:

- Dataset source.
- Split seed.
- Training seeds.
- Number of epochs.
- Batch size.
- Optimizer settings.
- Model-specific loss functions.
- Preprocessing parameters.
- Repetition count.
- Output locations.

A change to the configuration must be traceable to the experiment that used it.

## 8. Randomness and reproducibility

### 8.1 Random seeds

Record the seeds used for:

- Python random number generation.
- NumPy.
- TensorFlow.
- Training/validation splitting.

Use TensorFlow's supported seed-setting utilities where appropriate.

### 8.2 Repeated experiments

The target is three independent training runs per model, using the same predefined training and validation partitions.

Use a predefined list of training seeds, shared between the two models.

Each run must start from newly initialized model weights. Weights must not carry over between runs.

If hardware or time limitations prevent three complete runs, document the limitation and report the actual number of completed runs. A single run must not be presented as a repeated experiment.

### 8.3 Determinism

Enable deterministic TensorFlow operations where supported and practical.

Record whether deterministic execution was enabled successfully.

Some hardware operations and numerical kernels may still produce small floating-point differences. Reproducibility should therefore be described in terms of controlled experimental conditions rather than a guarantee of bit-for-bit identical results.

## 9. Training procedure

For each model and each predefined seed:

1. Set the random seeds.
2. Load or access the fixed dataset partition.
3. Apply the relevant preprocessing.
4. Construct a new model instance.
5. Compile using the model-specific optimizer and loss.
6. Verify the model's input and output shapes.
7. Verify the trainable parameter count.
8. Train using the training subset.
9. Record training and validation loss and accuracy for every epoch.
10. Preserve the training history and experiment metadata.

The training configuration is fixed at 15 epochs and batch size 128.

No early stopping or test-driven training decisions are permitted in the primary experiment.

The final epoch model will be used for test evaluation. Validation data is used for monitoring and descriptive analysis, not for selecting a different epoch in the primary comparison.

## 10. Evaluation metrics

### 10.1 Test accuracy

Accuracy measures the proportion of correctly classified examples.

$$
\text{Accuracy} =
\frac{\text{Correct predictions}}{\text{Total predictions}}
$$

For both models, convert the ten output values into predicted class indices using the index of the maximum output value.

Accuracy is the principal common metric for comparing classification performance.

### 10.2 Test loss

Record the compiled test loss for each model:

- Model A: mean squared error with bipolar targets.
- Model B: categorical crossentropy with one-hot targets.

The numerical loss values must be reported separately because the loss functions and target representations differ.

They must not be treated as directly equivalent measures.

### 10.3 Training time

Measure the duration of the model-fitting procedure.

The measurement must:

- Use the same computer.
- Use the same software environment.
- Include the same number of epochs.
- Exclude unrelated data downloading and preprocessing.
- Exclude model construction and compilation.
- Record the hardware and device used.

Record the duration of each run in seconds.

The first run may include one-time initialization overhead. This must be noted and handled consistently when interpreting results.

### 10.4 Parameter count

Record the trainable parameter count reported by Keras.

Expected values:

| Model | Expected trainable parameters |
|---|---:|
| LeCun-inspired CNN | 2,578 |
| Chollet CNN | 34,826 |

Any discrepancy must be resolved before training.

## 11. Additional diagnostic analysis

### 11.1 Learning curves

For every run, retain:

- Training loss per epoch.
- Validation loss per epoch.
- Training accuracy per epoch.
- Validation accuracy per epoch.

Produce plots that show the evolution of these metrics across epochs.

For repeated runs, individual run curves may be shown alongside an aggregated summary where appropriate.

### 11.2 Confusion matrices

Generate a confusion matrix for each trained model using its final test predictions.

The matrix must contain ten rows and ten columns, corresponding to digits zero through nine.

Use the matrices to identify:

- Frequently confused digit pairs.
- Classes with relatively high classification error.
- Differences in error patterns between the models.

### 11.3 Misclassified examples

Select representative incorrect predictions for qualitative analysis.

For each example, retain:

- Original image.
- True label.
- Predicted label.
- Model identity.

Do not selectively present examples in a way that misrepresents overall performance.

## 12. Repeated-run result aggregation

If three runs are completed, report:

- Mean test accuracy.
- Standard deviation of test accuracy.
- Mean training time.
- Standard deviation of training time.

The same aggregation approach should be applied consistently to both models.

Individual run results must also be preserved.

If fewer runs are completed, clearly report the actual number and avoid implying stronger statistical evidence than the experiment supports.

Test loss may also be summarized across runs, but the different loss definitions must remain explicit.

## 13. Experiment record structure

Every experiment run must have a unique identifier.

A recommended naming convention is:

`{model}_{seed}_{run}`

Examples:

- `lecun_seed42_run01`
- `chollet_seed42_run01`

Each run record should include:

| Field | Description |
|---|---|
| Experiment ID | Unique run identifier |
| Model | LeCun-inspired or Chollet |
| Seed | Training seed |
| Start time | Experiment start |
| End time | Experiment completion |
| Training duration | Seconds |
| Epochs completed | Number of epochs |
| Batch size | Training batch size |
| Trainable parameters | Keras parameter count |
| Final training loss | Last-epoch training loss |
| Final validation loss | Last-epoch validation loss |
| Test loss | Final test loss |
| Test accuracy | Final test accuracy |
| Software versions | Python, TensorFlow and dependencies |
| Hardware | CPU, GPU and memory details |
| Notes | Deviations or relevant observations |

The experiment record must be sufficient to trace every reported metric to the code, configuration and model that generated it.

## 14. Output organization

The repository will use the following output categories:

| Directory | Contents |
|---|---|
| `outputs/metrics/` | Run metrics and aggregated results |
| `outputs/logs/` | Training and execution logs |
| `outputs/figures/` | Learning curves, confusion matrices and diagnostic figures |
| `outputs/` | Other generated experiment artifacts where needed |

Generated model checkpoints and large binary artifacts should remain outside version control unless there is a specific reason to publish them.

Curated, relevant final results may be selectively added to Git after review.

## 15. Quality assurance

The following checks must pass before results are accepted:

### Dataset

- [ ] MNIST sample counts are correct.
- [ ] Image and label dimensions are correct.
- [ ] Labels are within the expected class range.
- [ ] Training and validation partitions do not overlap.
- [ ] Test samples remain independent.
- [ ] Class distributions are recorded.

### Preprocessing

- [ ] Both preprocessing pipelines produce the expected shapes.
- [ ] Pixel value ranges are correct.
- [ ] Output data types are correct.
- [ ] No NaN or infinite values are present.
- [ ] Labels remain correctly associated with their images.
- [ ] Empty-image handling is tested for Model A.

### Models

- [ ] Model A has 2,578 trainable parameters.
- [ ] Model B has 34,826 trainable parameters.
- [ ] Input and output dimensions are correct.
- [ ] Restricted connectivity is verified.
- [ ] Model outputs are finite.
- [ ] Model compilation succeeds.

### Experiments

- [ ] Seeds and configurations are recorded.
- [ ] Each run starts from newly initialized weights.
- [ ] Training histories are saved.
- [ ] Training times are recorded.
- [ ] Test evaluation follows the documented procedure.
- [ ] All deviations are documented.

### Analysis

- [ ] Results can be traced to individual runs.
- [ ] Accuracy is computed consistently.
- [ ] Loss functions are interpreted correctly.
- [ ] Confusion matrices use the correct labels.
- [ ] Figures accurately represent recorded data.
- [ ] No experimental result has been invented or substituted.

## 16. Protocol deviations

Any deviation from this protocol must be recorded with:

1. Description of the original requirement.
2. Description of the actual procedure.
3. Reason for the deviation.
4. Experiments affected.
5. Potential impact on comparison and interpretation.

A protocol change must not be silently applied to previously completed experiments. If a change affects comparability, the affected experiments must be repeated or the limitation explicitly reported.

## 17. Completion criteria

The experimental phase is complete when:

1. Both models have completed the predefined training procedure.
2. The actual number of runs is documented.
3. Training and validation histories are preserved.
4. Training times are recorded consistently.
5. Final test loss and accuracy are available.
6. Parameter counts have been verified.
7. Diagnostic visualizations have been generated.
8. Results are traceable to experiment configurations and run records.
9. All deviations and limitations are documented.