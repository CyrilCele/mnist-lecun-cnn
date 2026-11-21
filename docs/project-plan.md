# Project Plan

## 1. Project identification

**Project title:** Comparative Implementation and Evaluation of a LeCun-Inspired CNN and a Modern Keras MNIST Baseline

**Course:** DLBAIICV01 – Introduction to Computer Vision

**Assignment:** Task 1 – Handwritten Digit Recognition

**Repository:** `mnist-lecun-cnn`

**Project status:** In progress

## 2. Background and motivation

Convolutional neural networks (CNNs) are a central approach to image classification. LeCun et al. (1989) demonstrated the use of a back-propagation network for handwritten digit recognition. Modern deep-learning frameworks provide more accessible implementations and training methods, as illustrated by Chollet's Keras MNIST convolutional-network example.

This project investigates how a historically informed CNN architecture can be implemented using modern Keras and compared with a contemporary baseline on the MNIST handwritten-digit dataset.

The project is not intended to reproduce the original historical experiment exactly. It adapts the architecture to the MNIST dataset and a modern software environment. All deviations and assumptions will be documented.

## 3. Project objectives

The project has the following objectives:

1. Describe the MNIST dataset and its suitability for handwritten-digit classification.
2. Examine the historical CNN architecture described by LeCun et al. (1989).
3. Implement a historically informed approximation of that architecture using TensorFlow/Keras.
4. Implement the CNN baseline presented by Chollet (2021).
5. Define and execute a reproducible experimental protocol.
6. Compare both models using test loss, test accuracy and training time.
7. Interpret the results in terms of architecture, parameter count, preprocessing and computational cost.
8. Produce a professional project report and reproducible code artifacts.

## 4. Research questions

**RQ1:** What are the principal architectural differences between the historical CNN described by LeCun et al. (1989) and the modern Keras CNN presented by Chollet (2021)?

**RQ2:** How can the historical architecture be approximated using modern Keras layers while preserving its most important structural characteristics?

**RQ3:** How do the two implementations compare in test loss, test accuracy and training time when evaluated under a documented experimental protocol?

**RQ4:** How do architectural complexity, parameter count and preprocessing choices help explain the observed performance differences?

## 5. Scope

### 5.1 Included

- MNIST handwritten-digit classification.
- A LeCun-inspired CNN with historically motivated feature-map structure and restricted connectivity.
- A Chollet-style Keras CNN baseline.
- Documented preprocessing for both model pipelines.
- Reproducible training and evaluation.
- Model parameter counts.
- Test loss, test accuracy and training-time comparison.
- Learning curves, confusion matrices and selected prediction-error examples.
- A written report and executed notebook.

### 5.2 Excluded

- Reproducing the original postal ZIP-code dataset experiment.
- Claiming exact reproduction of the original 1989 hardware, software or optimization procedure.
- Hyperparameter searches using the test set.
- Claims of statistical significance without an appropriate statistical procedure.
- Production deployment or real-time digit recognition.

## 6. Research and implementation basis

The project uses three principal academic foundations:

1. LeCun et al. (1989), the historical handwritten-digit recognition network.
2. Chollet (2021), the Keras MNIST CNN example.
3. IU course book, _Introduction to Computer Vision_, particularly Unit 4: High-Level Vision, Sections 4.1 and 4.2.

The course book's discussion of convolution, pooling, dropout and fully connected layers provides the theoretical foundation for describing the model architectures and their operations.

## 7. Experimental design overview

The MNIST dataset provides 60,000 training images and 10,000 test images. The training portion will be divided into:

- Training: 54,000 images.
- Validation: 6,000 images.
- Test: 10,000 images.

The same sample indices will be used for both models. The test set will remain separate from training and model selection.

The initial experiment configuration is:

| Setting            | Planned value                           |
| ------------------ | --------------------------------------- |
| Dataset            | MNIST                                   |
| Training samples   | 54,000                                  |
| Validation samples | 6,000                                   |
| Test samples       | 10,000                                  |
| Epochs             | 15                                      |
| Batch size         | 128                                     |
| Optimizer          | Adam                                    |
| Randomness         | Fixed and recorded seeds                |
| Repeated runs      | Three where computationally practical   |
| Primary metrics    | Test loss, test accuracy, training time |

The exact model-specific loss functions, output activations, label encodings and preprocessing procedures are defined in the technical specification and experiment protocol.

## 8. Project phases and milestones

### Phase 1 — Project definition

**Status:** Completed

- Select Task 1.
- Define project objectives and research questions.
- Identify required deliverables.
- Establish the project phases and quality gates.

### Phase 2 — Research and technical specification

**Status:** Completed

- Review the LeCun architecture.
- Review the Chollet Keras baseline.
- Define the historical approximation.
- Specify the preprocessing and training assumptions.
- Calculate expected model parameter counts.

### Phase 3 — Environment and repository engineering

**Status:** In progress

- Configure Python and UV.
- Verify TensorFlow and supporting libraries.
- Configure Git and GitHub CLI.
- Establish the repository structure.
- Prepare project documentation.
- Verify dependency locking and repository hygiene.

**Exit criteria:** Environment checks pass, documentation is reviewed, and the repository is published with an initial milestone commit.

### Phase 4 — Dataset and model development

- Implement dataset loading and validation.
- Implement the two preprocessing pipelines.
- Implement the LeCun-inspired model.
- Verify its expected parameter count and tensor dimensions.
- Implement the Chollet baseline.
- Add automated tests.

**Exit criteria:** Both models build successfully and pass architecture and data-pipeline tests.

### Phase 5 — Experimental execution

- Configure reproducible training.
- Train both models.
- Record training histories and execution times.
- Evaluate the final models on the independent test set.
- Save model outputs and experiment metadata.

**Exit criteria:** Results are complete, traceable and generated without test-set leakage.

### Phase 6 — Results analysis

- Compare the required metrics.
- Produce tables and visualizations.
- Examine confusion matrices and selected errors.
- Interpret differences in model complexity and preprocessing.

**Exit criteria:** Every reported result can be traced to an experiment record.

### Phase 7 — Report preparation

- Write the introduction and literature review.
- Explain the methodology and architecture.
- Present experimental results.
- Discuss limitations and conclusions.
- Include references and code appendix.

**Exit criteria:** Report meets the assignment requirements and page limits.

### Phase 8 — Quality assurance

- Re-run tests.
- Validate notebook execution.
- Check numerical results against saved experiment records.
- Review citations, figures, tables and formatting.
- Check reproducibility instructions.

**Exit criteria:** No known blocking technical or reporting issues remain.

### Phase 9 — Submission preparation

- Export the final report as PDF.
- Export the executed notebook as PDF.
- Include the required source code.
- Verify all submission artifacts.
- Preserve the final project version in Git.

**Exit criteria:** All required submission artifacts are complete and readable.

## 9. Deliverables

| Deliverable              | Description                                                                      |
| ------------------------ | -------------------------------------------------------------------------------- |
| Project report           | 7–10-page PDF, excluding code appendix                                           |
| Executed notebook        | Notebook PDF displaying results                                                  |
| Notebook source          | `.ipynb` file                                                                    |
| Python source            | Reusable implementation modules                                                  |
| Environment definition   | `pyproject.toml` and `uv.lock`                                                   |
| Experiment configuration | Machine-readable settings                                                        |
| Test suite               | Automated checks for data and architecture                                       |
| Results                  | Metrics, logs and figures                                                        |
| Documentation            | Project plan, technical specification, experiment protocol and research register |

## 10. Risks and mitigation

| Risk                                               | Impact                                             | Mitigation                                                            |
| -------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------- |
| Historical details are ambiguous                   | Incorrect reproduction claims                      | Label assumptions and modern adaptations explicitly                   |
| Restricted connectivity is implemented incorrectly | Model no longer reflects the intended architecture | Test the connection pattern and parameter count                       |
| Preprocessing differs between models               | Confounded comparison                              | Document both pipelines and interpret results as pipeline comparisons |
| Randomness affects results                         | Unstable metrics                                   | Fix and record seeds; repeat runs where practical                     |
| Training time is inconsistent                      | Misleading comparison                              | Use the same hardware and a consistent timing procedure               |
| Test-set leakage                                   | Biased evaluation                                  | Keep the test set untouched until final evaluation                    |
| Environment changes                                | Poor reproducibility                               | Commit the UV lockfile and record software and hardware details       |
| Report exceeds page limit                          | Non-compliance                                     | Track the main report length separately from code appendix            |

## 11. Quality principles

The project will follow these principles:

- Reproducibility: preserve code, dependencies, configurations and random seeds.
- Traceability: connect every reported result to a recorded experiment.
- Transparency: distinguish historical facts from implementation assumptions.
- Fairness: use the same dataset partition and clearly state pipeline differences.
- Integrity: do not invent, estimate or substitute experimental results.
- Maintainability: use modular code, automated tests and milestone commits.
- Academic compliance: follow the assignment's report and notebook requirements.

## 12. Completion criteria

The project is complete when:

1. Both CNN implementations are implemented and tested.
2. The experiment protocol has been executed and documented.
3. Test loss, test accuracy and training times are reported for both models.
4. The comparison is supported by appropriate tables and visualizations.
5. The report addresses all five requirements in Task 1.
6. The report and executed notebook PDFs are complete.
7. The source code and environment configuration are included.
8. The final repository state is documented and reproducible.
