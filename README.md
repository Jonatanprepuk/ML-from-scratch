# ML-from-scratch

**ML-from-scratch** är ett Python-paket som implementerar grundläggande maskininlärningsalgoritmer och neurala nätverkskomponenter från grunden med NumPy.

## Översikt

* **Regression** – Linjär och polynomregression.
* **Neurala nätverk** – Modulärt ramverk med lager som `Dense`, `Conv2D`, `MaxPooling2d`, `Flatten` och `Dropout`.
* **Förlustfunktioner** – inklusive `MeanSquareError`, `MeanAbsoluteError`, `CategoricalCrossentropy` och `BinaryCrossentropy`.
* **Optimerare** – `SGD`, `Adagrad`, `RMSprop` och `Adam`.
* **Exempel** – Kodexempel för klassificering på MNIST, regressionsproblem med syntetiska data med mera.
* **Datasets** – Stöd för att ladda och hantera vanliga dataset (t.ex. MNIST).

> Detta projekt bygger vidare på idéer och kodexempel från boken *Neural Networks from Scratch* av Harrison Kinsley & Daniel Kukieła.

## Installation

1. Klona repo\:t:

   ```bash
   git clone https://github.com/Jonatanprepuk/ML-from-scratch.git
   cd ML-from-scratch
   ```
2. Installera paketet:

   ```bash
   pip install .
   ```
3. (Alternativt) Installera direkt via GitHub:

   ```bash
   pip install git+https://github.com/Jonatanprepuk/ML-from-scratch.git
   ```

## Kom igång (Snabbstart)

```python
from ml import Model, LayerInput, Dense, ActivationLayer, ReLU, Softmax
from ml.losses import CategoricalCrossentropy
from ml.optimizers import Adam
from ml.metrics import AccuracyCategorical

# Skapa och kompilera modell
model = Model([
    LayerInput(784),
    Dense(128), ActivationLayer(ReLU()),
    Dense(10), ActivationLayer(Softmax()),
])
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.001),
    metrics=[AccuracyCategorical()]
)

# Träna och utvärdera
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss, acc = model.evaluate(x_test, y_test)
print(f"Testförlust: {loss:.4f}, noggrannhet: {acc:.2%}")
```

## Projektstruktur

```
ML-from-scratch/
├── ml/               # Paketkod
│   ├── base.py       # Träningsbar abstraktion
│   ├── nn.py         # Lager och modell
│   ├── losses.py     # Förlustfunktioner
│   ├── optimizers.py # Optimerare
│   ├── metrics.py    # Mått
│   └── utils.py      # Hjälpklasser och datasetshantering
├── examples/         # Exempelskript
├── tests/            # Enhetstester
├── requirements.txt  # Beroenden
└── LICENSE           # MIT License
```

## Contributing

Förslag, felrapporter och pull requests är välkomna! Se [CONTRIBUTING.md](CONTRIBUTING.md) för riktlinjer.

## License

Detta projekt är licensierat under MIT License. Se [LICENSE](LICENSE) för detaljer.
