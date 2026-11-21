# LeCun-Inspired CNN vs. Keras MNIST Baseline

A reproducible computer vision project comparing a historically inspired convolutional neural network based on LeCun et al. (1989) with the modern Keras MNIST CNN example by Chollet (2021).

## Project objective

This project investigates the architectural differences between an early convolutional neural network and a modern CNN baseline using the MNIST handwritten-digit dataset.

The implementation aims to:

- Reconstruct a faithful, historically informed approximation of the LeCun architecture.
- Implement the Keras MNIST CNN baseline described by Chollet.
- Train and evaluate both models using a documented experimental protocol.
- Compare predictive performance, model complexity, and training time.
- Discuss the implications and limitations of adapting a historical architecture to MNIST.

## Models

**Model A — LeCun-inspired CNN**

A modern Keras implementation based on the architecture and connectivity described in LeCun et al. (1989). Any necessary modern adaptations will be explicitly documented.

**Model B — Keras MNIST CNN baseline**

A reproduction of the CNN architecture presented in Chollet's Keras MNIST example.

## Dataset

The project uses MNIST, containing 28 × 28 grayscale handwritten-digit images across ten classes (0–9).

The standard dataset provides 60,000 training images and 10,000 test images. The training portion will be divided into training and validation subsets using a documented, reproducible split.

## Experimental methodology

Both models will be evaluated using a consistent experimental protocol wherever their architectural and preprocessing requirements permit.

Planned evaluation measures include:

- Test accuracy
- Test loss
- Training time
- Number of trainable parameters
- Learning curves
- Confusion matrices and selected prediction errors

The test set will be reserved for final evaluation and will not be used for model selection.

## Reproducibility

The project uses:

- Python 3.12
- UV for dependency and environment management
- TensorFlow / Keras
- Git and GitHub for version control

The dependency lockfile, random seeds, model configurations, and experimental settings will be maintained in the repository.

## Repository structure

The source code, experiment configurations, tests, documentation, notebooks, and final report will be organized into separate directories as the project develops.

## Academic references

- LeCun, Y., et al. (1989). _Handwritten Digit Recognition with a Back-Propagation Network._
- Chollet, F. (2021). _Simple MNIST convnet._ Keras documentation.
- IU course book (2024). _Introduction to Computer Vision_, DLBAIICV01.

## Project status

Under development. Experimental results will be added only after the corresponding experiments have been executed and verified.
