# ML-from-scratch

**ML-from-scratch** is a Python package that implements fundamental machine learning algorithms and neural network building blocks from scratch using NumPy.

## Overview

* **Regression** – Linear regression (analytical and gradient based).
* **Neural Networks** – Modular framework with layers such as `Dense`, `Conv2D`, `MaxPooling2D`, `Flatten` and `Dropout` and activation functions such as `Leaky RelU`, `ReLU`, `Sigmoid`, `Softmax`.
* **Loss Functions** – `MeanSquareError`, `MeanAbsoluteError`, `CategoricalCrossentropy`, `BinaryCrossentropy`.
* **Optimizers** – `SGD`, `Adagrad`, `RMSprop`, `Adam`.
* **Examples** – Scripts for MNIST classification, regression on synthetic datasets, and more.
* **Datasets** – Utilities for loading common datasets.

> This project builds upon concepts and code examples from the book [*Neural Networks from Scratch*](https://github.com/Sentdex/nnfs_book) by Harrison Kinsley & Daniel Kukieła.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Jonatanprepuk/ML-from-scratch.git
   cd ML-from-scratch
   ```
2. Install the package:

   ```bash
   pip install .
   ```

## Quickstart

```python
from ml import Model, LayerInput, Dense, ActivationLayer, ReLU, Softmax
from ml.losses import CategoricalCrossentropy
from ml.optimizers import Adam
from ml.metrics import AccuracyCategorical

# Build and compile the model
model = Model([
    LayerInput(784),
    Dense(128), ActivationLayer(ReLU()),
    Dense(10), ActivationLayer(Softmax()),
])
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    accuracy=[AccuracyCategorical()]
)

# Train and evaluate
model.fit(x_train, y_train, epochs=5, batch_size=32)
model.evalute()
```

## Project Structure

```
ML-from-scratch/
├── ml/               # Core package
│   ├── base.py       # Trainable abstractions
│   ├── nn.py         # Layers and model definitions
│   ├── losses.py     # Loss function implementations
│   ├── optimizers.py # Optimization algorithms
│   ├── metrics.py    # Evaluation metrics
│   └── utils.py      # Helper functions and dataset loaders
├── examples/         # Example scripts
├── tests/            # Unit tests
├── requirements.txt  # Dependencies
└── LICENSE           # MIT License
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
