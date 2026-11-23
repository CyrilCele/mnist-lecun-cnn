# LeCun-Inspired CNN vs. Keras MNIST Baseline

A reproducible computer vision project comparing a historically informed, LeCun-inspired convolutional neural network (CNN) with the Keras MNIST CNN example described by Chollet (2021).

The project implements both models in TensorFlow/Keras, evaluates them on MNIST, and examines the effects of architecture, preprocessing, model complexity and training configuration. The LeCun-inspired network is a modern adaptation; it is **not** an exact reproduction of the original 1989 experiment.

## Project overview

**Course:** DLBAIICV01 – Introduction to Computer Vision  
**Assignment:** Task 1 – Handwritten Digit Recognition  
**Language:** Python 3.12  
**Framework:** TensorFlow 2.21 / Keras  
**Environment management:** UV

### Models

- **Model A – LeCun-inspired CNN:** a compact network with convolutional feature maps, subsampling stages, restricted connectivity, tanh outputs and bipolar targets.
- **Model B – Chollet CNN:** a Keras MNIST baseline using ReLU convolutions, max pooling, dropout and a softmax output with one-hot targets.

The models use different preprocessing and loss functions. The experiment therefore compares two complete model pipelines; it does not isolate architecture as the only changed variable.

## Dataset and experimental design

The project uses the standard MNIST dataset: 60,000 training images and 10,000 test images, each a 28 × 28 grayscale image representing one of ten digits (0–9).

The 60,000 training examples are split once into 54,000 training and 6,000 validation examples using a stratified split. The same partitions are used for both models. The official 10,000-image test set is reserved for final evaluation.

Each model is trained for 15 epochs with a batch size of 128. Three runs use seeds 42, 43 and 44. The configuration is recorded in [`configs/experiment.json`](configs/experiment.json).

## Results

The notebook reports the mean and sample standard deviation across three runs:

| Metric | LeCun-inspired CNN | Chollet CNN |
|---|---:|---:|
| Test accuracy | 97.477% ± 0.071% | **99.117% ± 0.042%** |
| Test loss | 0.02065 ± 0.00059 | 0.02609 ± 0.00182 |
| Trainable parameters | **2,578** | 34,826 |

The Chollet baseline achieved 1.64 percentage points higher mean test accuracy and used approximately 13.51 times as many trainable parameters. The loss values are reported separately and **must not be compared directly**: the LeCun-inspired model uses mean squared error with bipolar targets, while the Chollet model uses categorical crossentropy with one-hot targets. The three-run results are descriptive; they are not a statistical-significance test.

Training-time measurements are recorded in the notebook and experiment records. **Before quoting a single timing result in a final report, reconcile the two retained run-record sets**: `outputs/metrics/run_records.json` and `notebooks/outputs/notebook_reproduction/run_records.json` contain the same test metrics but different measured durations. The notebook's aggregate table is calculated from the notebook-reproduction records. Do not combine timing values from the two locations.

The executed notebook contains the detailed analysis, learning curves, confusion matrices, selected misclassified examples and per-class metrics.

## Repository structure

```text
.
├── configs/
│   └── experiment.json
├── docs/
│   ├── experiment-protocol.md
│   ├── project-plan.md
│   ├── research-register.md
│   └── technical-specification.md
├── notebooks/
│   ├── mnist_cnn_comparison.ipynb
│   └── outputs/notebook_reproduction/
├── outputs/
│   ├── diagnostics/
│   ├── figures/
│   ├── logs/
│   └── metrics/
├── src/mnist_lecun_cnn/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

- `src/mnist_lecun_cnn/`: reusable data, preprocessing, model, training, experiment and analysis code.
- `notebooks/mnist_cnn_comparison.ipynb`: executed project notebook and narrative analysis.
- `notebooks/outputs/notebook_reproduction/`: notebook run records, training histories and diagnostic outputs.
- `outputs/`: project-level experiment records, logs, diagnostics and figures.
- `docs/`: project plan, technical specification, experimental protocol and research register.

## Getting started

### Requirements

- Python 3.12
- [UV](https://docs.astral.sh/uv/)
- Git

### Set up the environment

```bash
git clone https://github.com/CyrilCele/mnist-lecun-cnn.git
cd mnist-lecun-cnn
uv sync --all-groups
```

### Run tests and lint checks

```bash
uv run pytest
uv run ruff check .
```

### Open the notebook

```bash
uv run jupyter lab
```

Open `notebooks/mnist_cnn_comparison.ipynb` and execute it from top to bottom to reproduce the analysis. The experiment runner can reuse completed runs. Review the configuration and output paths before starting fresh training runs.

MNIST is loaded through the TensorFlow/Keras dataset loader when it is not already available locally. The recorded runs used Windows 11, Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.1 and NumPy 2.5.5 on CPU; no GPU was available. Training time will vary with hardware and software.

## Documentation

- [Project plan](docs/project-plan.md)
- [Technical specification](docs/technical-specification.md)
- [Experimental protocol](docs/experiment-protocol.md)
- [Research register](docs/research-register.md)

## Academic references

1. LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W. and Jackel, L. D. (1989). *Handwritten Digit Recognition with a Back-Propagation Network*. Advances in Neural Information Processing Systems 2, 396–404.
2. LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324. https://doi.org/10.1109/5.726791
3. Chollet, F. (2021). *Simple MNIST convnet*. Keras example. https://keras.io/examples/vision/mnist_convnet/
4. IU International University of Applied Sciences. (2024). *Introduction to Computer Vision* (DLBAIICV01), Unit 4.

## Project status

The implementation, six training runs, result analysis and executed notebook are complete. The repository contains the source code, configuration, experiment records, diagnostics and project documentation. The separate assignment report and any required PDF exports should be prepared and checked against the course submission instructions.

## License

The project code is released under the MIT License. See [`LICENSE`](LICENSE). This does not relicense third-party datasets, publications, documentation or dependencies; their respective terms continue to apply.
