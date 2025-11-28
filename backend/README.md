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

# assign output directory
output_dir = os.path.join("outputs")
os.makedirs(output_dir, exist_ok=True)

# Save training data to a pickle file as a dictionary with keys "X" and "y"
train_pkl = os.path.join(output_dir, "train_data.pkl")
with open(train_pkl, "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

# Initialize trainer
trainer = Trainer(
    directory=output_dir,
    hidden_sizes=[16, 8],
    Lambda=0.01,            # not required, default value set as 0.01
    epochs=1000,            # not required, default value set as 1000
    learning_rate=0.01,     # not required, default value set as 0.01
    train_val_split=0.8,    # not required, default value set as 0.8
    beta1=0.9,              # not required, default value set as 0.9
    beta2=0.999,            # not required, default value set as 0.999
    epsilon=1e-8            # not required, default value set as 1e-8
)

# Train model using the pickle file path
train_loss, val_loss = trainer.train(train_pkl)
```

### Testing a model

```python
from interpy_bg.tester import Tester

# Use the same output directory where the model was saved
output_dir = os.path.join("outputs")

tester = Tester(hidden_sizes=[16, 8], Lambda=0.01)
predictions = tester.predict(X)  # Can also pass a .pkl file with test data
```

### Plotting results

```python
from interpy_bg.plotter import plot_loss, plot_predictions

output_dir = os.path.join("outputs")

plot_loss(train_loss, val_loss, "rmse_vs_epochs.png", output_dir)
plot_predictions(y, predictions, "ytrue_vs_ypred.png", output_dir)
```

## Documentation

Full API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io).
See details for every class, method and plotting utility.

## License

MIT License
