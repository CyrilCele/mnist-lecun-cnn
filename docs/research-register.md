# Research Register

## 1. Purpose

This register records the academic, technical, and implementation sources used to define the project. It links each source to the decisions it supports and identifies limitations that matter when interpreting the results.

The project compares a historically informed, LeCun-inspired CNN with a modern Keras MNIST CNN baseline. It does **not** claim to reproduce the original historical experiments exactly.

## 2. Primary academic and course sources

### R1 — LeCun et al. (1989)

**Reference**

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., and Jackel, L. D. (1989). _Handwritten Digit Recognition with a Back-Propagation Network_. Advances in Neural Information Processing Systems 2, 396–404.

**Source:** https://dl.acm.org/doi/10.5555/2969830.2969879

**Relevance to the project**

- Provides an early foundation for applying convolutional neural networks and back-propagation to handwritten-digit recognition.
- Informs the historically motivated design of local receptive fields, feature maps, subsampling, and trainable network parameters.
- Provides historical context for the use of digit-recognition tasks.

**Limitations and interpretation**

The original work used postal-digit data and its own experimental setup. This project uses MNIST and a modern TensorFlow/Keras implementation. Results must not be described as a replication of the original experiment.

### R2 — LeCun et al. (1998)

**Reference**

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). _Gradient-Based Learning Applied to Document Recognition_. Proceedings of the IEEE, 86(11), 2278–2324.

**DOI:** https://doi.org/10.1109/5.726791

**Relevance to the project**

- Provides a detailed account of gradient-based learning for document recognition.
- Discusses convolutional network architecture, feature maps, subsampling, and connectivity patterns.
- Provides additional architectural context for interpreting LeCun-inspired network designs.

**Limitations and interpretation**

The architecture implemented in this project is a reduced, MNIST-adapted design. Its layer dimensions and parameter count must be treated as project-specific unless they are independently verified against the original publication. The implementation is not an exact reproduction of LeNet-5.

### R3 — Chollet (2021), Keras MNIST CNN example

**Reference**

Chollet, F. (2021). _Simple MNIST convnet_. Keras documentation / example implementation.

**Source:** https://keras.io/examples/vision/mnist_convnet/

**Relevance to the project**

- Serves as the reference for the modern Keras CNN baseline.
- Informs the use of standard convolutional layers, ReLU activation, max pooling, dropout, and a softmax classification output.
- Provides a practical reference for implementing and training a CNN on MNIST.

**Limitations and interpretation**

The project records its actual baseline architecture and training configuration in the technical specification. Any differences between the reference example and this controlled implementation must be documented rather than silently attributed to Chollet's original example.

### R4 — IU (2024), course textbook

**Reference**

IU International University of Applied Sciences. (2024). _Introduction to Computer Vision_ (DLBAIICV01), Unit 4: Deep Learning, particularly Sections 4.1 and 4.2.

**Source:** Course-provided textbook.

**Relevance to the project**

- Provides course context for deep learning and image classification.
- Supports explanations of convolutional neural networks, convolution, pooling, and classification.
- Informs the theoretical background and terminology used in the project documentation and eventual report.

**Limitations and interpretation**

The textbook is used as a course reference, not as evidence that the specific experimental architecture or results have been independently validated.

## 3. Technical implementation sources

### R5 — TensorFlow / Keras documentation

**Reference**

TensorFlow and Keras documentation, including the API documentation for convolutional, pooling, dense, dropout, and model-training functionality.

**Sources:**

- https://www.tensorflow.org/api_docs/python/tf/keras
- https://keras.io/api/layers/

**Relevance to the project**

- Guides implementation of the models using supported Keras layers and APIs.
- Supports model construction, compilation, training, evaluation, and parameter inspection.
- Provides the technical basis for implementing custom layers or operations where standard layers alone do not express the specified architecture.

**Limitations and interpretation**

Framework documentation describes software behavior and APIs. It does not establish historical equivalence between a modern implementation and an original research architecture.

### R6 — scikit-learn documentation

**Reference**

scikit-learn developers. _scikit-learn documentation_.

**Source:** https://scikit-learn.org/stable/

**Relevance to the project**

- Supports reproducible stratified splitting of the training data.
- Provides evaluation utilities such as confusion matrices and classification metrics.
- Helps validate the experimental evaluation pipeline.

### R7 — NumPy documentation

**Reference**

NumPy developers. _NumPy documentation_.

**Source:** https://numpy.org/doc/

**Relevance to the project**

- Supports numerical data handling and preprocessing.
- Provides array operations used for dataset validation, label conversion, and metric calculations.

## 4. Project-specific design decisions

The following decisions are adaptations made for this project. They should not be represented as direct historical specifications unless supported by the original sources.

| Decision                    | Project treatment                                                         | Interpretation                                                      |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Dataset                     | MNIST                                                                     | Modern benchmark dataset, not the original postal-digit dataset     |
| LeCun-inspired architecture | Reduced feature-map counts and specified restricted connectivity          | Historically informed adaptation                                    |
| Subsampling                 | Average pooling with trainable per-map scale and bias                     | Modern approximation of trainable subsampling                       |
| Activation                  | Tanh in the LeCun-inspired model                                          | Explicit modern design choice                                       |
| Optimization                | Adam                                                                      | Modern optimizer, not a claim about the original training procedure |
| Input preprocessing         | Bounding-box crop, aspect-preserving resize, centering, and normalization | Project-defined preprocessing                                       |
| Baseline                    | Keras CNN based on the specified Chollet example                          | Reference-inspired modern baseline                                  |
| Evaluation                  | Shared training/validation split and independent test set                 | Controlled comparative experiment                                   |
| Repeated runs               | Multiple runs with documented random seeds                                | Project reproducibility measure                                     |

## 5. Evidence and verification requirements

Before final analysis and report preparation:

1. Verify all architecture claims against the cited publications or implementation source.
2. Keep historical facts separate from project-specific adaptations.
3. Record any differences between the cited Keras example and the implemented baseline.
4. Record software versions and the final dependency lockfile.
5. Cite sources for theoretical claims in the final report.
6. Report measured experimental results only after executing and validating the experiments.

## 6. Register maintenance

This register should be updated whenever a new source materially influences the architecture, preprocessing, experimental methodology, or interpretation of results.

The final report bibliography should be checked against this register. All sources cited in the report should appear in the bibliography, and all bibliography entries should correspond to sources actually consulted.
