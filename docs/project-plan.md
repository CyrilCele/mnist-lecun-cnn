# Project Plan and Completion Record

## 1. Project identification

- **Project:** Comparative Implementation and Evaluation of a LeCun-Inspired CNN and a Modern Keras MNIST Baseline
- **Course:** DLBAIICV01 – Introduction to Computer Vision
- **Assignment:** Task 1 – Handwritten Digit Recognition
- **Repository:** `CyrilCele/mnist-lecun-cnn`
- **Implementation status:** Complete
- **Experiment status:** Complete (three runs per model)

## 2. Background and purpose

LeCun et al. (1989) demonstrated back-propagation for handwritten-digit recognition using a convolutional network. Chollet's Keras MNIST example provides a modern CNN baseline. This project implements a historically informed approximation of the former and compares it with the latter on MNIST.

The project does not reproduce the original postal ZIP-code experiment. The dataset, software framework, optimization method and some implementation details differ from the historical work. These adaptations are documented in the technical specification and notebook.

## 3. Objectives

1. Describe the MNIST dataset and its suitability for handwritten-digit classification.
2. Examine the historical CNN design and identify its relevant structural ideas.
3. Implement a LeCun-inspired network in TensorFlow/Keras, including restricted feature-map connectivity.
4. Implement the Chollet-style Keras MNIST baseline.
5. Apply documented, reproducible data preparation and training procedures.
6. Compare predictive performance, parameter count and training time.
7. Interpret the results in light of architectural and preprocessing differences.
8. Preserve an executed notebook, source code, tests and experiment records.

## 4. Research questions

- **RQ1:** What are the principal architectural differences between the LeCun-inspired network and the Chollet CNN baseline?
- **RQ2:** How can key structural characteristics of the historical network be approximated using modern Keras layers?
- **RQ3:** How do the implemented pipelines compare on MNIST under the documented experimental procedure?
- **RQ4:** How do parameter count, preprocessing and training configuration help explain the observed results?

## 5. Scope

### Included

- MNIST handwritten-digit classification.
- LeCun-inspired CNN with restricted connectivity and trainable subsampling transformations.
- Chollet-style Keras CNN baseline.
- Model-specific image preprocessing and target encoding.
- Three training runs per model using seeds 42, 43 and 44.
- Test evaluation, learning curves, confusion matrices, misclassified examples and per-class metrics.
- Modular Python implementation, configuration, automated tests and executed notebook.

### Excluded

- Reproduction of the original postal ZIP-code dataset experiment.
- Claims of exact historical or LeNet-5 reproduction.
- Test-set-driven model selection or hyperparameter tuning.
- Statistical-significance claims based on only three runs.
- Production deployment.

## 6. Completed experimental design

| Setting | Implemented value |
|---|---|
| Dataset | MNIST |
| Training partition | 54,000 |
| Validation partition | 6,000 |
| Test partition | 10,000 |
| Split | Stratified; seed 42 |
| Training seeds | 42, 43, 44 |
| Repetitions | 3 per model |
| Epochs | 15 |
| Batch size | 128 |
| Optimizer | Adam |
| LeCun-inspired loss/targets | MSE / bipolar |
| Chollet loss/targets | Categorical crossentropy / one-hot |

The two models use distinct preprocessing, target representations and loss functions. Results therefore compare complete pipelines rather than architecture in isolation.

## 7. Completion record

| Work package | Final status | Evidence |
|---|---|---|
| Project definition and research questions | Complete | This document |
| Literature and technical research | Complete | `research-register.md` |
| Architecture and preprocessing specification | Complete | `technical-specification.md` |
| Environment and repository setup | Complete | `pyproject.toml`, `uv.lock` |
| Data and model implementation | Complete | `src/mnist_lecun_cnn/` |
| Automated tests | Implemented | `tests/` |
| Experimental execution | Complete | Six run records and histories |
| Results and diagnostics | Complete | Notebook and output figures |
| Notebook interpretation | Complete | `notebooks/mnist_cnn_comparison.ipynb` |
| Documentation and licence | Finalisation package supplied | Repository documentation |

## 8. Final measured findings

The notebook's three-run aggregate reports:

| Metric | LeCun-inspired CNN | Chollet CNN |
|---|---:|---:|
| Mean test accuracy | 97.477% ± 0.071% | 99.117% ± 0.042% |
| Mean test loss | 0.02065 ± 0.00059 | 0.02609 ± 0.00182 |
| Trainable parameters | 2,578 | 34,826 |

The Chollet model achieved 1.64 percentage points higher mean accuracy and used approximately 13.51 times as many parameters. The loss values are not directly comparable because the models use different loss functions and target encodings. Training-time results must be taken from a reconciled, explicitly identified run-record set; see the experimental protocol.

## 9. Risks, limitations and interpretation

- **Historical equivalence:** Model A is a historically informed adaptation, not an exact reproduction.
- **Pipeline confounding:** Preprocessing, activation, target encoding and loss differ between models. Performance differences cannot be attributed solely to architecture.
- **Small number of repetitions:** Three runs provide descriptive variation, not strong statistical evidence.
- **Timing variability:** Training times depend on machine load, software and hardware. The repository currently contains two run-record sets with different durations; they must not be mixed.
- **Hardware scope:** The recorded runs used CPU execution on Windows 11; timing should not be generalized to GPU systems.
- **External submission files:** The repository's `reports/` directory is a placeholder. Any required report PDF and notebook PDF must be exported and checked separately according to the course instructions.