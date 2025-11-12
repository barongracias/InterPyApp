# interpy_bg

`interpy_bg` is a feedforward neural network library designed for 5D → 1D interpolation.  
It provides modular classes for defining, training, and testing neural networks, with built-in normalization, RMSE tracking, and plotting utilities.

## Features

- Feedforward neural networks with customizable hidden layers
- L2 regularization
- Training with RMSE tracking and validation split
- Normalization of input data
- Save/load trained weights and normalization values
- Simple plotting of training/validation loss and predictions

## Installation

Install via pip:

```bash
pip install interpy-bg
```

or directly via GitHub:

```bash
pip install git+https://github.com/barongracias/InterPyApp.git#egg=interpy-bg&subdirectory=interpy_bg
```

## Quick Start

### Training a model

```python
import numpy as np
from interpy_bg.trainer import Trainer

# Dummy dataset
X = np.random.rand(50, 5)
y = np.random.rand(50, 1)

# Initialize trainer
trainer = Trainer(
    hidden_sizes=[16, 8],
    Lambda=0.01,
    epochs=500,
    learning_rate=0.01,
    train_val_split=0.8
)

# Train model
train_loss, val_loss = trainer.train(X, y)
```

### Training a model

```python
from interpy_bg.tester import Tester

tester = Tester(hidden_sizes=[16, 8], Lambda=0.01)
predictions = tester.predict(X)  # Can also pass a .pkl file with test data
```

### Plotting results

```python
from interpy_bg.plotter import plot_loss, plot_predictions

plot_loss(train_loss, val_loss, filename="rmse_vs_epochs.png")
plot_predictions(y, predictions, filename="ytrue_vs_ypred.png")
```

## Documentation

Full API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io).
See details for every class, method and plotting utility.

## License

MIT License