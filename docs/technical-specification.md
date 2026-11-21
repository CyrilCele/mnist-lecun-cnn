# Technical Specification

## 1. Purpose

This document defines the technical design of the two convolutional neural networks (CNNs) used in the MNIST handwritten-digit classification project.

It specifies:

- The dataset and model input requirements.
- The LeCun-inspired historical architecture.
- The Chollet Keras baseline.
- Preprocessing and label representations.
- Activation functions, loss functions and optimization.
- Expected tensor dimensions and trainable parameter counts.
- Implementation assumptions and acceptance criteria.

The implementation must follow this specification unless a justified change is documented before the affected experiment is executed.

## 2. Technical environment

| Component               | Specification           |
| ----------------------- | ----------------------- |
| Operating system        | Windows                 |
| Shell                   | Git Bash (MINGW64)      |
| Python                  | 3.12                    |
| Environment manager     | UV                      |
| Deep-learning framework | TensorFlow 2.21 / Keras |
| Numerical computation   | NumPy                   |
| Data analysis           | pandas, scikit-learn    |
| Visualization           | Matplotlib              |
| Testing                 | pytest                  |
| Code quality            | Ruff                    |
| Version control         | Git                     |
| Remote repository       | GitHub                  |

The exact installed package versions are controlled by `uv.lock`. Hardware details, TensorFlow device information and other relevant environment characteristics will be recorded before the experiments.

## 3. Dataset specification

### 3.1 MNIST

The project uses the MNIST handwritten-digit dataset.

| Property                  | Specification  |
| ------------------------- | -------------- |
| Number of training images | 60,000         |
| Number of test images     | 10,000         |
| Image dimensions          | 28 × 28 pixels |
| Channels                  | 1 (grayscale)  |
| Classes                   | 10             |
| Class labels              | Integers 0–9   |
| Original pixel range      | 0–255          |

MNIST is a benchmark dataset for handwritten-digit classification. Each image represents one digit from zero to nine.

### 3.2 Dataset partitioning

The 60,000 training images will be divided into:

| Subset     | Samples | Purpose                        |
| ---------- | ------: | ------------------------------ |
| Training   |  54,000 | Model parameter optimization   |
| Validation |   6,000 | Monitoring and model selection |
| Test       |  10,000 | Final independent evaluation   |

A fixed random seed will determine the training/validation split. The same sample indices will be used for both models.

The original test set will remain separate and will not be used for training, hyperparameter selection or other model-development decisions.

## 4. Model A — LeCun-inspired CNN

### 4.1 Historical foundation

Model A is based on the convolutional architecture described by LeCun et al. (1989), particularly its use of:

- Convolutional feature extraction.
- Subsampling between convolutional stages.
- Restricted connectivity between feature maps.
- A compact network with a small number of trainable parameters.
- A final classification stage.

The original research addressed postal ZIP-code digit recognition. This project applies a historically informed architectural approximation to MNIST instead.

Consequently, Model A must not be described as an exact reproduction of the original experiment.

### 4.2 Architecture

The planned network consists of an input layer, two convolutional stages, two subsampling stages and a final classification layer.

| Stage     | Operation                                                   | Output dimensions | Trainable parameters |
| --------- | ----------------------------------------------------------- | ----------------- | -------------------: |
| Input     | Grayscale image                                             | 28 × 28 × 1       |                    0 |
| H1        | 4 convolutional maps, 5 × 5, valid                          | 24 × 24 × 4       |                  104 |
| H2        | 2 × 2 average subsampling, trainable scale and bias per map | 12 × 12 × 4       |                    8 |
| H3        | 12 convolutional maps, 5 × 5, restricted connectivity       | 8 × 8 × 12        |                  512 |
| H4        | 2 × 2 average subsampling, trainable scale and bias per map | 4 × 4 × 12        |                   24 |
| Output    | Fully connected classification layer                        | 10 outputs        |                1,930 |
| **Total** |                                                             |                   |            **2,578** |

All spatial dimensions assume valid convolution and non-overlapping 2 × 2 subsampling.

### 4.3 H1: first convolutional stage

- Number of feature maps: 4.
- Kernel size: 5 × 5.
- Padding: valid.
- Input channels: 1.
- Activation: tanh.

Output dimensions:

28 − 5 + 1 = 24

Parameter calculation:

4 × (5 × 5 × 1 + 1) = 104

### 4.4 H2: first subsampling stage

- Operation: 2 × 2 average pooling.
- Pooling stride: 2.
- Number of feature maps: 4.
- Trainable parameters: one scale and one bias per feature map.

The average pooling operation reduces each 24 × 24 feature map to 12 × 12.

The trainable scale and bias approximate the trainable subsampling behaviour of the historical architecture.

Parameter calculation:

4 × (1 + 1) = 8

The implementation will use a dedicated Keras layer or an equivalent explicitly tested construction to apply the trainable transformation consistently across each feature map.

### 4.5 H3: second convolutional stage

- Number of output feature maps: 12.
- Kernel size: 5 × 5.
- Padding: valid.
- Input feature maps: 4.
- Connectivity: restricted according to LeCun et al. (1989), Table 1.
- Activation: tanh.

Output dimensions:

12 − 5 + 1 = 8

#### Restricted connectivity

The connections between H2 and H3 are specified as follows:

| H2 input map | Connected H3 output maps        |
| ------------ | ------------------------------- |
| H2.1         | H3.1, H3.2, H3.3, H3.5, H3.6    |
| H2.2         | H3.2, H3.3, H3.4, H3.5, H3.6    |
| H2.3         | H3.7, H3.8, H3.9, H3.11, H3.12  |
| H2.4         | H3.8, H3.9, H3.10, H3.11, H3.12 |

There are 20 permitted input-to-output feature-map connections.

Each connection has a 5 × 5 kernel containing 25 weights. Each of the 12 output maps has one bias.

Parameter calculation:

(20 × 25) + 12 = 512

The restricted connectivity will be implemented using separate convolution branches or an equivalent explicit connectivity mechanism. It must not be replaced by an unrestricted convolution whose parameter count includes connections that should not exist.

### 4.6 H4: second subsampling stage

- Operation: 2 × 2 average pooling.
- Pooling stride: 2.
- Number of feature maps: 12.
- Trainable parameters: one scale and one bias per feature map.

The 12 feature maps are reduced from 8 × 8 to 4 × 4.

Parameter calculation:

12 × (1 + 1) = 24

### 4.7 Output layer

The H4 feature maps are flattened:

4 × 4 × 12 = 192 features

The output layer contains 10 units, one for each digit class.

Parameter calculation:

(192 × 10) + 10 = 1,930

The planned output activation is tanh, with bipolar target encoding.

### 4.8 Total parameter count

| Component | Parameters |
| --------- | ---------: |
| H1        |        104 |
| H2        |          8 |
| H3        |        512 |
| H4        |         24 |
| Output    |      1,930 |
| **Total** |  **2,578** |

**Implementation acceptance criterion:** Keras must report exactly 2,578 trainable parameters for Model A. Any discrepancy must be investigated before training.

### 4.9 Activation and target encoding

The planned activation function is tanh.

The target representation is bipolar:

- Correct class: +1.
- Incorrect classes: −1.

For example, the target for digit 3 is:

[-1, -1, -1, +1, -1, -1, -1, -1, -1, -1]

The model's predicted class is the index of the output unit with the highest activation.

### 4.10 Loss and optimization

| Setting           | Specification            |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Loss              | Mean squared error (MSE) |
| Epochs            | 15                       |
| Batch size        | 128                      |
| Output activation | tanh                     |
| Target encoding   | Bipolar                  |

Adam is a modern optimization choice. It is not claimed to reproduce the original optimization procedure used in the historical paper.

The use of MSE with bipolar targets is an explicit project design decision.

## 5. Model B — Chollet Keras CNN baseline

### 5.1 Reference architecture

Model B follows the CNN example described by Chollet in the Keras MNIST tutorial.

It uses conventional convolutional feature extraction, max pooling, dropout and a softmax classification layer.

### 5.2 Architecture

| Stage     | Operation                      | Output dimensions | Parameters |
| --------- | ------------------------------ | ----------------- | ---------: |
| Input     | Grayscale image                | 28 × 28 × 1       |          0 |
| Conv1     | 32 filters, 3 × 3, ReLU, valid | 26 × 26 × 32      |        320 |
| Pool1     | 2 × 2 max pooling              | 13 × 13 × 32      |          0 |
| Conv2     | 64 filters, 3 × 3, ReLU, valid | 11 × 11 × 64      |     18,496 |
| Pool2     | 2 × 2 max pooling              | 5 × 5 × 64        |          0 |
| Flatten   | Flatten feature maps           | 1,600             |          0 |
| Dropout   | Rate 0.5                       | 1,600             |          0 |
| Output    | Dense 10, softmax              | 10                |     16,010 |
| **Total** |                                |                   | **34,826** |

### 5.3 Parameter calculations

First convolution:

(3 × 3 × 1 × 32) + 32 = 320

Second convolution:

(3 × 3 × 32 × 64) + 64 = 18,496

Output dense layer:

(1,600 × 10) + 10 = 16,010

Total:

320 + 18,496 + 16,010 = 34,826

**Implementation acceptance criterion:** Keras must report exactly 34,826 trainable parameters for Model B.

### 5.4 Training configuration

| Setting              | Specification            |
| -------------------- | ------------------------ |
| Optimizer            | Adam                     |
| Loss                 | Categorical crossentropy |
| Output activation    | Softmax                  |
| Label representation | One-hot encoded          |
| Epochs               | 15                       |
| Batch size           | 128                      |
| Dropout rate         | 0.5                      |
| Primary metric       | Accuracy                 |

The implementation will use an explicit training/validation split so that both models use the same validation samples. This is a documented adaptation of the original example's use of `validation_split=0.1`.

## 6. Preprocessing specifications

### 6.1 Model A preprocessing

The LeCun-inspired model uses a historically motivated normalization procedure.

The planned procedure is:

1. Load the original 28 × 28 grayscale MNIST image.
2. Identify the digit's foreground bounding box using nonzero pixels.
3. Crop the image to the foreground bounding box.
4. Resize the cropped digit, preserving its aspect ratio, so that it fits within a 16 × 16 region.
5. Place the resized digit in the centre of a 28 × 28 canvas.
6. Normalize pixel intensities to the range [−1, +1].

The bounding-box-based geometric centring is a project-defined adaptation. It must not be presented as an exact reproduction of the original preprocessing algorithm.

Any implementation-specific details, such as interpolation method, rounding, empty-image handling and pixel normalization, must be recorded and tested.

### 6.2 Model B preprocessing

The baseline uses standard MNIST preprocessing:

1. Load the 28 × 28 grayscale images.
2. Add the channel dimension.
3. Convert pixel values to float32.
4. Divide pixel intensities by 255.
5. Convert labels to categorical one-hot vectors.

Pixel values therefore lie in [0, 1].

### 6.3 Preprocessing comparison

| Property                | Model A                          | Model B                  |
| ----------------------- | -------------------------------- | ------------------------ |
| Input dimensions        | 28 × 28 × 1                      | 28 × 28 × 1              |
| Geometric normalization | Digit fitted into 16 × 16 region | None                     |
| Pixel normalization     | [−1, +1]                         | [0, 1]                   |
| Label representation    | Bipolar                          | One-hot                  |
| Output activation       | tanh                             | softmax                  |
| Loss                    | MSE                              | Categorical crossentropy |

The different preprocessing and training objectives are deliberate parts of the two model pipelines.

Consequently, the primary experiment compares complete model pipelines rather than isolating architecture as the only changing variable.

## 7. Comparison framework

### 7.1 Required evaluation metrics

Both models will be evaluated using:

- Test loss.
- Test accuracy.
- Training time.

Additional analysis will include:

- Trainable parameter count.
- Training and validation learning curves.
- Confusion matrices.
- Selected misclassified examples.

### 7.2 Interpretation of loss

The two models use different target representations and loss functions.

Model A uses MSE with bipolar targets, whereas Model B uses categorical crossentropy with one-hot targets.

Their numerical loss values therefore have different meanings and should not be interpreted as directly equivalent measures of model quality.

Accuracy provides a common classification metric. Loss values will be reported separately with this limitation explained.

### 7.3 Training-time comparison

Training time will be measured on the same computer and under the same software environment.

The measurement procedure will be defined in the experiment protocol. It will distinguish model construction, data preparation, model fitting and final evaluation so that timing results are interpreted consistently.

## 8. Historical facts versus project adaptations

| Design decision                        | Classification                        |
| -------------------------------------- | ------------------------------------- |
| Convolutional feature extraction       | Historical design principle           |
| Subsampling stages                     | Historical design principle           |
| Restricted H2–H3 connectivity          | Based on historical architecture      |
| Four H1 maps and twelve H3 maps        | Historical architecture specification |
| MNIST as the experimental dataset      | Project adaptation                    |
| Bounding-box-based MNIST normalization | Project-defined adaptation            |
| tanh activation                        | Explicit modern approximation         |
| Adam optimizer                         | Modern replacement                    |
| MSE with bipolar targets               | Project implementation decision       |
| Modern TensorFlow/Keras implementation | Modern replacement                    |
| Fifteen epochs and batch size 128      | Experimental configuration            |

The original LeCun work used a postal-digit recognition dataset, not MNIST. Its reported historical performance must not be presented as a result of this project.

## 9. Implementation acceptance criteria

Before experimental training begins, the following requirements must be satisfied.

### Model A

- [ ] Input shape is 28 × 28 × 1.
- [ ] H1 has four 5 × 5 valid convolutional feature maps.
- [ ] H2 produces four 12 × 12 maps.
- [ ] H2 has trainable per-map scale and bias parameters.
- [ ] H3 has twelve 5 × 5 valid convolutional feature maps.
- [ ] H2–H3 restricted connectivity matches the specified table.
- [ ] H4 produces twelve 4 × 4 maps.
- [ ] H4 has trainable per-map scale and bias parameters.
- [ ] Output contains ten tanh units.
- [ ] Bipolar target encoding is correct.
- [ ] Total trainable parameter count equals 2,578.

### Model B

- [ ] Input shape is 28 × 28 × 1.
- [ ] First convolution has 32 filters and a 3 × 3 kernel.
- [ ] First max-pooling operation is 2 × 2.
- [ ] Second convolution has 64 filters and a 3 × 3 kernel.
- [ ] Second max-pooling operation is 2 × 2.
- [ ] Dropout rate is 0.5.
- [ ] Output contains ten softmax units.
- [ ] One-hot target encoding is correct.
- [ ] Total trainable parameter count equals 34,826.

### Shared requirements

- [ ] Both models receive the same training, validation and test sample indices.
- [ ] Random seeds are recorded.
- [ ] Preprocessing is covered by automated tests.
- [ ] Model input and output shapes are verified.
- [ ] Parameter counts are verified programmatically.
- [ ] The test set remains independent until final evaluation.
- [ ] The experiment configuration is documented.

## 10. References

- LeCun, Y., et al. (1989). _Handwritten Digit Recognition with a Back-Propagation Network._
- Chollet, F. (2021). _Simple MNIST convnet._ Keras documentation.
- IU (2024). _Introduction to Computer Vision_. DLBAIICV01, Unit 4: High-Level Vision, Sections 4.1 and 4.2.
