# ML-from-scratch

**ML-from-scratch** is a Python package that implements fundamental machine learning algorithms and neural network building blocks from scratch using NumPy.

## Overview

* **Regression** – Linear regression (analytical and gradient based).
* **Classification** – K-Nearest Neighbors and Gaussian Naive Bayes
* **Neural Networks** – Modular framework with layers such as `Dense`, `Conv2D`, `MaxPooling2D`, `Flatten` and `Dropout` and activation functions such as `Leaky ReLU`, `ReLU`, `Sigmoid`, `Softmax`.
* **Loss Functions** – `MeanSquareError`, `MeanAbsoluteError`, `CategoricalCrossentropy`, `BinaryCrossentropy`.
* **Optimizers** – `SGD`, `Adagrad`, `RMSprop`, `Adam`.
* **Examples** – Scripts for MNIST classification, regression on synthetic datasets, and more.
* **Datasets** – Functions for generating synthetic datasets

> This project builds upon concepts and code examples from the book [*Neural Networks from Scratch*](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła.

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
from ml.nn import *
from ml.optimizers import Adam
from ml.losses import CategoricalCrossentropy
from ml.datasets import blobs_data
import matplotlib.pyplot as plt

# Generate data
X, y, X_test, y_test = blobs_data(classes=5, samples_per_class=200, seed=1234, test_split=0.2)


# Build and compile the model
model = Model(
    Dense(X.shape[1], 64),
    LeakyReLU(alpha=0.3),
    Dense(64,64),
    LeakyReLU(alpha=0.3),
    Dense(64,64),
    LeakyReLU(alpha=0.3),
    Dense(64, 5),
    Softmax()
)

model.set(loss=CategoricalCrossentropy(), 
          optimizer=Adam(learning_rate=0.0001, decay=1e-2), 
          accuracy=AccuracyCategorical())

model.finalize()

# Train and plot loss 
model.train(X,y, epochs=500, print_every=100, batch_size=64, validation_data=(X_test, y_test))
model.plot_loss()
```

## Examples

See [`examples/`](./examples) for notebooks and scripts covering:
- Linear regression 
- Autoencoder
- Classification (Mnist & synthtic)

## Project Structure

```
ML-from-scratch/
├── ml/                     # Core package
│   ├── base.py             # Abstract class for trainable objects
│   ├── classification/     # Classification models, knn, naive bayes
│   ├── datasets/           # Functions for generating synthetic datasets
│   ├── regression/         # Linear regression models
│   ├── nn/                 # Layers, accuracies and model definitions
│   ├── losses/             # Loss function implementations
│   └── optimizers/         # Optimization algorithms
├── examples/               # Example scripts
├── setup.py          
├── requirements.txt  
├── README.md         
└── LICENSE           
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
