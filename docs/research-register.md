# Research Register

## 1. Purpose and scope

This register records the principal academic and technical sources used to design and interpret the MNIST CNN comparison. It distinguishes source-supported historical and theoretical ideas from project-specific implementation choices.

The project compares a LeCun-inspired modern CNN with the Keras MNIST CNN example described by Chollet. It does not claim to reproduce the original postal-digit experiment.

## 2. Academic and course sources

### R1 — LeCun et al. (1989)

**Reference**

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W. and Jackel, L. D. (1989). _Handwritten Digit Recognition with a Back-Propagation Network_. Advances in Neural Information Processing Systems 2, 396–404.

**Source:** https://dl.acm.org/doi/10.5555/2969830.2969879

**Use in this project**

- Historical context for back-propagation and convolutional approaches to handwritten-digit recognition.
- Motivation for local receptive fields, feature maps and subsampling.
- Historical reference for restricted connectivity.

**Interpretive limitation**

The original work used postal-digit data and its own experimental setup. This project uses MNIST and a modern TensorFlow/Keras implementation. Its results are not a replication of the original experiment.

### R2 — LeCun et al. (1998)

**Reference**

LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). _Gradient-Based Learning Applied to Document Recognition_. Proceedings of the IEEE, 86(11), 2278–2324.

**DOI:** https://doi.org/10.1109/5.726791

**Use in this project**

- Additional explanation of convolutional networks, feature maps, subsampling and document recognition.
- Architectural context for LeCun-family network designs.

**Interpretive limitation**

The network implemented here is a reduced MNIST adaptation. It is not an exact reproduction of LeNet-5 or of the complete historical document-recognition system.

### R3 — Chollet (2021), Keras MNIST CNN example

**Reference**

Chollet, F. (2021). _Simple MNIST convnet_. Keras example.

**Source:** https://keras.io/examples/vision/mnist_convnet/

**Use in this project**

- Reference for the modern CNN baseline.
- Architectural guidance for convolution, ReLU, max pooling, dropout and softmax classification.

**Interpretive limitation**

The implemented experiment uses a shared explicit training/validation split and a controlled configuration. Differences from the reference example are project adaptations and should not be attributed to the original example.

### R4 — IU course textbook

**Reference**

IU International University of Applied Sciences. (2024). _Introduction to Computer Vision_ (DLBAIICV01), Unit 4: High-Level Vision, Sections 4.1 and 4.2.

**Source:** Course-provided textbook.

**Use in this project**

- Course-level theoretical background for deep learning, convolutional neural networks, convolution, pooling and classification.
- Terminology and conceptual explanations used in the project documentation and report.

## 3. Technical implementation sources

### R5 — TensorFlow and Keras

**Sources:**

- https://www.tensorflow.org/api_docs/python/tf/keras
- https://keras.io/api/layers/

Used for model construction, layers, compilation, training, evaluation and parameter inspection. Framework documentation establishes API behaviour, not historical equivalence.

### R6 — scikit-learn

**Source:** https://scikit-learn.org/stable/

Used for stratified data splitting and evaluation utilities.

### R7 — NumPy

**Source:** https://numpy.org/doc/

Used for array operations, numerical preprocessing and metric calculations.

## 4. Project-specific decisions

| Decision               | Treatment                                                          | Classification                      |
| ---------------------- | ------------------------------------------------------------------ | ----------------------------------- |
| Dataset                | MNIST                                                              | Project adaptation                  |
| LeCun-inspired model   | Reduced feature-map design with restricted connectivity            | Historically informed approximation |
| Subsampling            | Average pooling with trainable per-map scale and bias              | Modern approximation                |
| Preprocessing          | Bounding-box crop, proportional resize, centring and normalization | Project-defined                     |
| Activation and targets | tanh and bipolar targets for Model A                               | Project decision                    |
| Optimization           | Adam                                                               | Modern choice                       |
| Baseline               | Keras CNN based on Chollet's example                               | Reference-inspired implementation   |
| Evaluation             | Shared split, independent test set, three seeded runs              | Experimental design                 |

## 5. Register status

This register records the principal sources used for the completed implementation and analysis.
