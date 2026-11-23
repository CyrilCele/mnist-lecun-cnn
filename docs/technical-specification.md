# Technical Specification — Implemented System

## 1. Purpose and implementation status

This document describes the implemented MNIST classification pipelines. It records the architecture, preprocessing, target representations, optimization and verified parameter counts. Model A is a LeCun-inspired modern adaptation, not an exact reproduction of the 1989 postal-digit experiment.

## 2. Software environment

| Component | Recorded environment |
|---|---|
| Operating system | Windows 11 |
| Python | 3.12.13 |
| Environment manager | UV |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.5.5 |
| Data analysis | pandas, scikit-learn |
| Visualization | Matplotlib |
| Testing | pytest |
| Linting | Ruff |
| Execution device | CPU; no GPU available |

The dependency constraints are in `pyproject.toml`; exact resolved dependencies are in `uv.lock`.

## 3. Dataset and partitions

MNIST contains 70,000 grayscale images of handwritten digits.

| Property | Value |
|---|---|
| Original training set | 60,000 images |
| Official test set | 10,000 images |
| Image dimensions | 28 × 28 |
| Channels | 1 |
| Classes | 10 (digits 0–9) |
| Original pixel range | 0–255 |

The original training set is split into 54,000 training and 6,000 validation samples using a stratified split with seed 42. The same split is used for both models. The official test set is held out for final evaluation.

## 4. Model A — LeCun-inspired CNN

### 4.1 Design rationale

The model incorporates convolutional feature extraction, subsampling stages, restricted connectivity between feature maps and a compact classification layer. The layer sizes and implementation are adapted to this project and must not be described as an exact reproduction of the original historical network.

### 4.2 Implemented architecture

| Stage | Operation | Output dimensions | Trainable parameters |
|---|---|---:|---:|
| Input | Grayscale image | 28 × 28 × 1 | 0 |
| H1 | 4 valid 5 × 5 convolutional maps, tanh | 24 × 24 × 4 | 104 |
| H2 | 2 × 2 average subsampling; trainable scale and bias per map | 12 × 12 × 4 | 8 |
| H3 | 12 valid 5 × 5 convolutional maps, restricted connectivity, tanh | 8 × 8 × 12 | 512 |
| H4 | 2 × 2 average subsampling; trainable scale and bias per map | 4 × 4 × 12 | 24 |
| Output | Flatten and dense 10-unit tanh layer | 10 | 1,930 |
| **Total** | | | **2,578** |

### 4.3 Parameter calculations

- H1: `4 × (5 × 5 × 1 + 1) = 104`
- H2: `4 × (1 scale + 1 bias) = 8`
- H3: `20 permitted connections × 25 weights + 12 biases = 512`
- H4: `12 × (1 scale + 1 bias) = 24`
- Output: `(4 × 4 × 12 × 10) + 10 = 1,930`

Total: **2,578 trainable parameters**.

The H2-to-H3 connectivity pattern uses 20 permitted input-map/output-map connections:

| H2 input map | Connected H3 output maps |
|---|---|
| H2.1 | H3.1, H3.2, H3.3, H3.5, H3.6 |
| H2.2 | H3.2, H3.3, H3.4, H3.5, H3.6 |
| H2.3 | H3.7, H3.8, H3.9, H3.11, H3.12 |
| H2.4 | H3.8, H3.9, H3.10, H3.11, H3.12 |

### 4.4 Training configuration

| Setting | Implemented value |
|---|---|
| Output activation | tanh |
| Target encoding | Bipolar: +1 for the correct class, −1 otherwise |
| Loss | Mean squared error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 15 |
| Batch size | 128 |
| Reported metric | Categorical accuracy |

Adam and the MSE/bipolar objective are modern project choices, not claims about the original training procedure.

## 5. Model B — Chollet Keras CNN baseline

### 5.1 Implemented architecture

| Stage | Operation | Output dimensions | Parameters |
|---|---|---:|---:|
| Input | Grayscale image | 28 × 28 × 1 | 0 |
| Conv1 | 32 filters, 3 × 3, valid, ReLU | 26 × 26 × 32 | 320 |
| Pool1 | 2 × 2 max pooling | 13 × 13 × 32 | 0 |
| Conv2 | 64 filters, 3 × 3, valid, ReLU | 11 × 11 × 64 | 18,496 |
| Pool2 | 2 × 2 max pooling | 5 × 5 × 64 | 0 |
| Flatten | Flatten feature maps | 1,600 | 0 |
| Dropout | Rate 0.5 | 1,600 | 0 |
| Output | Dense 10, softmax | 10 | 16,010 |
| **Total** | | | **34,826** |

Parameter calculations:

- Conv1: `(3 × 3 × 1 × 32) + 32 = 320`
- Conv2: `(3 × 3 × 32 × 64) + 64 = 18,496`
- Output: `(1,600 × 10) + 10 = 16,010`

Total: **34,826 trainable parameters**.

### 5.2 Training configuration

| Setting | Implemented value |
|---|---|
| Output activation | Softmax |
| Target encoding | Ten-class one-hot |
| Loss | Categorical crossentropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 15 |
| Batch size | 128 |
| Dropout | 0.5 |
| Reported metric | Categorical accuracy |

The implementation uses the same explicit 54,000/6,000 training/validation partition as Model A. This is a controlled adaptation of the reference example's validation handling.

## 6. Preprocessing

### 6.1 LeCun-inspired pipeline

1. Find the foreground bounding box using nonzero pixels.
2. Crop the image to the foreground.
3. Resize proportionally to fit within 16 × 16 pixels.
4. Centre the resized digit on a 28 × 28 canvas.
5. Normalize pixels using `pixel / 127.5 - 1.0`, producing values in [−1, +1].
6. Add the channel dimension.

The recorded configuration specifies bilinear interpolation, half-up dimension rounding, integer-floor centring and an empty-image canvas value of −1.0. This bounding-box procedure is a project-defined adaptation, not an assertion of exact historical preprocessing.

### 6.2 Chollet pipeline

1. Convert images to `float32`.
2. Add the channel dimension.
3. Normalize using `pixel / 255.0`, producing values in [0, 1].
4. Encode labels as ten-class one-hot vectors.

### 6.3 Comparison and implications

| Property | LeCun-inspired | Chollet |
|---|---|---|
| Input shape | 28 × 28 × 1 | 28 × 28 × 1 |
| Geometric processing | Crop, aspect-preserving resize and centre | None |
| Pixel range | [−1, +1] | [0, 1] |
| Targets | Bipolar | One-hot |
| Output activation | tanh | softmax |
| Loss | MSE | Categorical crossentropy |

These pipeline differences are relevant when interpreting performance. The experiment does not isolate architecture as a single independent variable.

## 7. Evaluation and interpretation

Both models are evaluated on the same official MNIST test set. The common classification measure is accuracy. Test loss is recorded using each model's compiled loss, but the values are not directly comparable because their objectives and target encodings differ.

The analysis includes trainable parameter counts, training/validation curves, confusion matrices, selected misclassified examples and per-class precision, recall and F1-score.

## 8. Historical facts and project adaptations

| Design element | Status |
|---|---|
| Convolutional feature extraction | Historical design principle |
| Subsampling stages | Historical design principle |
| Restricted feature-map connectivity | Historically motivated |
| MNIST dataset | Project adaptation |
| Specific reduced feature-map dimensions | Project implementation choice |
| Bounding-box crop and 16 × 16 resize | Project-defined preprocessing |
| tanh output and bipolar targets | Project design choice |
| Adam optimizer | Modern choice |
| TensorFlow/Keras implementation | Modern implementation |
| 15 epochs and batch size 128 | Experimental configuration |

## 9. Configuration source

`configs/experiment.json` is the machine-readable source for seeds, repetitions, optimizer settings, model losses, preprocessing parameters, expected parameter counts and output locations. The executed notebook and saved run records document the completed experiment.
