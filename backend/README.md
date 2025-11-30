# interpy_bg

`interpy_bg` is a feedforward neural network library designed for 5D → 1D interpolation.  
It provides modular classes for defining, training, and testing neural networks, with built-in normalization, RMSE tracking, and plotting utilities.

## Features

- Feedforward neural networks with customizable hidden layers
- L2 regularization
- Training with RMSE tracking and validation split
- Normalization of input data
- Save/load trained weights, normalization values, and model metadata
- Simple plotting of training/validation loss and predictions
- Dataset validation/standardisation with train/val/test splits
- Synthetic 5D data generator utilities

## Installation

Install via pip:

```bash
pip install interpy_bg
```

or directly via GitHub:

```bash
pip install git+https://github.com/barongracias/InterPyApp.git#egg=interpy-bg&subdirectory=backend
```

For the TensorFlow implementation (`fivedreg`), install from its subdirectory:

```bash
pip install git+https://github.com/barongracias/InterPyApp.git#egg=fivedreg&subdirectory=backend/fivedreg
```

## Quick Start

### Training a model

```python
import numpy as np
from interpy_bg.trainer import Trainer
import os, pickle

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
    epochs=300,             # reduce for quicker runs
    learning_rate=0.01,     # not required, default value set as 0.01
    train_val_split=0.8,    # not required, default value set as 0.8
    beta1=0.9,              # not required, default value set as 0.9
    beta2=0.999,            # not required, default value set as 0.999
    epsilon=1e-8,           # not required, default value set as 1e-8
    activation="relu",      # optional: sigmoid/tanh/relu/leakyrelu
    weight_init="auto",     # optional: auto/he/xavier
    batch_size=32,          # optional: mini-batching
    grad_clip=5.0,          # optional: gradient clipping
    early_stop_patience=20, # optional: early stopping
    lr_decay=0.98,          # optional: LR decay per epoch
    seed=42,                # optional: reproducibility
)

# Train model using the pickle file path
train_loss, val_loss = trainer.train(train_pkl)
```

### Testing a model

```python
from interpy_bg.tester import Tester

# Use the same output directory where the model was saved
output_dir = os.path.join("outputs")

tester = Tester(
    hidden_sizes=[16, 8],
    Lambda=0.01,
    directory=output_dir,
    activation="relu",
    weight_init="auto",
)
predictions = tester.predict(X)  # Can also pass a .pkl file with test data
```

### Plotting results

```python
from interpy_bg.plotter import plot_loss, plot_predictions

output_dir = os.path.join("outputs")

plot_loss(train_loss, val_loss, "rmse_vs_epochs.png", output_dir)
plot_predictions(y, predictions, "ytrue_vs_ypred.png", output_dir)
```

### Synthetic data

```python
from interpy_bg.synthetic import synthetic_5d, synthetic_5d_pickle

# Generate arrays
X, y = synthetic_5d(1000, seed=42)

# Persist with metadata
path = synthetic_5d_pickle("outputs/synth.pkl", n=1000, seed=42)
```

## Tests

- Core library tests: `tests/` (e.g., `tests/test_trainer.py`, `tests/test_pipeline.py`).
- API/synthetic/performance tests: `backend/tests/` (e.g., `test_api.py`, `test_synthetic.py`, `test_performance.py`).

## Notes

- NumPy training writes `model_weights.npz`, `normalisation_values.npz`, plots, and `model_metadata.json` (architecture, Lambda, activation/init, batch/clip/seed, best metrics incl. R²) into `backend/outputs/` (when running via the API).
- TensorFlow training (set `model_type=tf` on `/train`) writes `model_tf.keras`, `normalisation_values_tf.npz`, plots, and `tf_model_metadata.json` into the same `backend/outputs/`.
- Prediction (`/predict` or `Tester.predict`) uses the trained architecture/config in metadata; client-supplied hidden sizes or Lambda are ignored. `/predict` also accepts `model_type` to choose NumPy vs TF.
- API endpoints include `/health`, `/upload` (accepts .pkl dict with X/y and returns dataset stats), `/train`, `/predict`, `/plots/{filename}`, `/artifacts/{filename}` (serves NumPy or TF artifacts), and `/evaluate` (prefers NumPy artifacts, falls back to TF if present).
- `/reset` clears uploads plus both output folders (`backend/outputs/` and `backend/outputs_tf/`).
- Plotting uses the headless `Agg` backend in both packages for compatibility with servers/CI.

## Documentation

Full API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io).
See details for every class, method and plotting utility.

## License

MIT License
