# interpy_bg

`interpy_bg` is a feedforward neural network library designed for 5D → 1D interpolation.  
It provides modular classes for defining, training, and testing neural networks, with built-in normalization, RMSE tracking, and plotting utilities.

- Docs: https://interpyapp.readthedocs.io/en/latest/index.html#
- Source: https://github.com/barongracias/InterPyApp

## Features

- Feedforward neural networks with customizable hidden layers
- L2 regularization
- Training with RMSE tracking and validation split
- Normalization of input data
- Save/load trained weights, normalization values, and model metadata
- Simple plotting of training/validation loss and predictions
- Dataset validation/standardisation with train/val/test splits
- Synthetic 5D data generator utilities (via `interpy_synth` dependency)

## Installation

Local/dev (installs interpy_bg + interpy_synth + fivedreg_tf from PyPI):

```bash
cd backend
pip install -r requirements.lock
```

PyPI:

```bash
pip install interpy_bg         # NumPy backend (pulls interpy-synth)
pip install fivedreg_tf        # TF backend (pulls interpy-synth + tensorflow)
```

Docker:

```bash
cd ..
./scripts/docker_build.sh
./scripts/docker_up.sh   # backend on :8000
# ./scripts/docker_down.sh to stop
```

Environment:
- Configure CORS via `ALLOWED_ORIGINS` (comma-separated), e.g. copy `backend/.env.example`.
- CPU-only: no GPU required; TensorFlow uses the CPU build.
- For reproducibility, prefer `requirements.lock`.

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
output_dir = os.path.join("outputs_numpy")
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
output_dir = os.path.join("outputs_numpy")

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

output_dir = os.path.join("outputs_numpy")

plot_loss(train_loss, val_loss, "rmse_vs_epochs.png", output_dir)
plot_predictions(y, predictions, "ytrue_vs_ypred.png", output_dir)
```

### Synthetic data

```python
from interpy_synth import synthetic_5d, synthetic_5d_pickle

# Generate arrays
X, y = synthetic_5d(1000, seed=42)

# Persist with metadata
path = synthetic_5d_pickle("outputs_numpy/synth.pkl", n=1000, seed=42)
```

## Hyperparameter guide (UI/API)

- `hidden_sizes`: Layer widths per hidden layer. More/larger layers increase capacity and training time and can overfit small datasets.
- `Lambda`: L2 regularization strength; higher shrinks weights harder to reduce overfitting but can underfit.
- `activation`: ReLU default; LeakyReLU avoids dead units; tanh/sigmoid bound outputs but can slow training.
- `weight_init`: Auto picks He for ReLU/LeakyReLU and Xavier for tanh/sigmoid; override to experiment.
- `epochs`: Full passes over the data. More epochs can fit better but take longer and may overfit.
- `learning_rate`: Step size for gradient updates. Higher learns faster but risks divergence; lower is steadier.
- `train_val_split`: Fraction for training vs validation/early stopping. Smaller training splits can reduce fit quality.
- `batch_size`: Samples per gradient step. Larger batches smooth updates but use more memory; blank/full-batch is allowed.
- `grad_clip`: Upper bound on gradient norm to prevent exploding gradients. Lower means more aggressive clipping.
- `lr_decay`: Multiplier (<1) applied per epoch to the learning rate. Leave unset to keep LR constant.
- `early_stop_patience`: Stop after this many epochs without validation improvement; lower stops sooner to avoid overfitting.
- `beta1` / `beta2`: Adam momentum terms for first/second moments. Higher values smooth updates but react slower.
- `epsilon`: Small constant for numerical stability in Adam; keep default unless debugging NaNs.
- `seed`: Set for deterministic initialisation/shuffling; leave unset for nondeterministic runs.

## Tests

- All backend tests live in `backend/tests/` (API, NumPy, TensorFlow, synthetic, performance).
- Run with `python -m pytest backend/tests`.

## Notes

- NumPy training writes `model_weights.npz`, `normalisation_values.npz`, plots, and `model_metadata.json` (architecture, Lambda, activation/init, batch/clip/seed, best metrics incl. R²) into `backend/outputs_numpy/` (when running via the API).
- TensorFlow training (set `model_type=tf` on `/train`) writes `model_tf.keras`, `normalisation_values_tf.npz`, plots, and `tf_model_metadata.json` into `backend/outputs_tf/` (served alongside NumPy artifacts).
- Prediction (`/predict` or `Tester.predict`) uses the trained architecture/config in metadata; client-supplied hidden sizes or Lambda are ignored. `/predict` also accepts `model_type` to choose NumPy vs TF.
- API endpoints include `/health`, `/upload` (accepts .pkl dict with X/y and returns dataset stats; uploads are content-type checked and stored with UUID-prefixed filenames), `/train`, `/jobs/{id}`, `/predict`, `/plots/{filename}`, `/artifacts/{filename}`, and `/evaluate` (returns RMSE on a supplied X/y pickle; prefers NumPy artifacts, falls back to TF if present). When `REDIS_URL` is set, `/train` pings Redis and enqueues an RQ job (`/jobs/{id}` reports status/results); if Redis is unavailable or enqueue fails, training logs a warning and runs synchronously.
- `/reset` clears uploads plus both output folders (`backend/outputs_numpy/` and `backend/outputs_tf/`).
- Plotting uses the headless `Agg` backend in both packages for compatibility with servers/CI.

## Documentation

Full API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io).
See details for every class, method and plotting utility.

## License

MIT License
