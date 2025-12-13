# Backend Overview

This backend hosts the FastAPI service (`main.py`) plus three installable packages:
- `interpy_bg`: NumPy implementation of the 5D→1D regressor (pip: `interpy_bg`)
- `fivedreg_tf`: TensorFlow implementation mirroring the NumPy API (pip: `fivedreg_tf`)
- `interpy_synth`: Synthetic data generator shared by both backends (pip: `interpy-synth`)

`main.py` exposes `/upload`, `/train`, `/predict`, `/artifacts`, `/plots`, `/evaluate`, and `/reset`, writing NumPy artifacts to `backend/outputs_numpy/` and TF artifacts to `backend/outputs_tf/`. Set `model_type` to `numpy` or `tf` on train/predict to choose the backend. `/evaluate` returns RMSE on a supplied X/y pickle and prefers NumPy artifacts, falling back to TF if present. Uploads are content-type checked and stored with UUID-prefixed filenames; use `stored_filename` from `/upload` when calling `/train`.

Below are package-specific notes; the examples remain focused on `interpy_bg`, with `fivedreg_tf` usage analogous (using `outputs_tf` and TF classes), and `interpy_synth` providing synthetic data.

## Features

- Feedforward neural networks with customizable hidden layers (NumPy and TF)
- L2 regularization; Adam optimisation with optional early stopping, LR decay, batch size, grad clipping
- Training with RMSE tracking and validation split
- Normalization of input data; save/load trained weights, normalization values, and model metadata
- Plotting of training/validation loss and predictions (headless Agg backend)
- Dataset validation/standardisation with train/val/test splits
- Synthetic 5D data generator utilities via `interpy_synth`
- Structured logging for API events (upload/train/predict/evaluate) including durations, backend type, and RMSE summaries

## Installation

Local/dev (installs interpy_bg + interpy_synth + fivedreg_tf from PyPI):

```bash
cd backend
pip install -r requirements.lock
```

PyPI:

```bash
pip install interpy_bg          # NumPy backend (pulls interpy-synth)
pip install fivedreg_tf         # TF backend (pulls interpy-synth + tensorflow-cpu)
pip install interpy-synth       # Synthetic data helpers
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
- CPU-only: TensorFlow backend is required and uses the CPU build.
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
- Applicability: NumPy backend uses all fields; TensorFlow backend honours activation, weight_init, batch_size, grad_clip, lr_decay, early_stop_patience, beta1/beta2/epsilon, learning_rate, hidden_sizes, Lambda, train_val_split, seed.

## Tests

- All backend tests live in `backend/tests/` (API, NumPy, TensorFlow, synthetic, performance). API tests now cover both backends including TensorFlow artifact serving and evaluate flows.
- Run with `python -m pytest backend/tests`.

## Notes

- NumPy training writes `model_weights.npz`, `normalisation_values.npz`, plots, and `model_metadata.json` (architecture, Lambda, activation/init, batch/clip/seed, best metrics incl. R²) into `backend/outputs_numpy/` (when running via the API).
- TensorFlow training (set `model_type=tf` on `/train`) writes `model_tf.keras`, `normalisation_values_tf.npz`, plots, and `tf_model_metadata.json` (includes activation/init, Adam betas/epsilon, batch/clip, learning rate, metrics) into `backend/outputs_tf/` (served alongside NumPy artifacts).
- Prediction (`/predict` or `Tester.predict`) uses the trained architecture/config in metadata; client-supplied hidden sizes or Lambda are ignored. `/predict` also accepts `model_type` to choose NumPy vs TF.
- API endpoints include `/health`, `/upload` (accepts .pkl dict with X/y and returns dataset stats), `/train`, `/predict`, `/plots/{filename}`, `/artifacts/{filename}` (serves NumPy or TF artifacts from their respective output folders), and `/evaluate` (returns RMSE on a supplied X/y pickle; prefers NumPy artifacts, falls back to TF if present).
- `/reset` clears uploads plus both output folders (`backend/outputs_numpy/` and `backend/outputs_tf/`).
- Plotting uses the headless `Agg` backend in both packages for compatibility with servers/CI.
- TensorFlow optimiser: uses `tf.keras.optimizers.legacy.Adam` when available (faster on Apple Silicon per TF warning) and falls back to `tf.keras.optimizers.Adam`.

## Performances and profiling

Two performance scripts exercise the NumPy and TensorFlow backends over multiple dataset sizes, logging throughput, memory, and accuracy metrics:

- NumPy: `python backend/tests/test_performance_numpy.py` (uses `outputs_numpy/size_<n>/`)
- TensorFlow: `python backend/tests/test_performance_tensorflow.py` (uses `outputs_tf/size_<n>/`)

Each run prints a summary table with train/predict wall time, peak memory (via `tracemalloc`), end-of-training RMSE, and evaluation MSE/R² on a fresh synthetic test set. Artifacts (plots, weights, metadata) are saved per size under the corresponding outputs folder for visual inspection. Review the printed tables and the saved `rmse_vs_epochs.png` / `ytrue_vs_ypred.png` to spot underfitting/overfitting or memory regressions when tuning hyperparameters or changing code.

Latest run (CPU, sizes 1k/5k/10k, 200 epochs, hidden [64,32,16]):

NumPy (`interpy_bg`)

| n    | train_s | pred_s | train_mb | pred_mb | train_rmse | val_rmse | mse      | r2    |
|------|---------|--------|----------|---------|------------|----------|----------|-------|
| 1000 | 2.847   | 0.0089 | 3.60     | 1.38    | 0.1624     | 0.1835   | 0.021614 | 0.9234 |
| 5000 | 6.511   | 0.0070 | 16.33    | 1.33    | 0.1254     | 0.1322   | 0.015862 | 0.9438 |
| 10000| 11.261  | 0.0073 | 32.52    | 1.33    | 0.1291     | 0.1287   | 0.016535 | 0.9414 |

TensorFlow (`fivedreg_tf`)

| n    | train_s | pred_s | train_mb | pred_mb | train_rmse | val_rmse | mse      | r2    |
|------|---------|--------|----------|---------|------------|----------|----------|-------|
| 1000 | 4.454   | 0.0904 | 5.32     | 0.31    | 0.1602     | 0.1685   | 0.024084 | 0.9078 |
| 5000 | 6.699   | 0.0895 | 5.27     | 0.31    | 0.1236     | 0.1308   | 0.013365 | 0.9488 |
| 10000| 9.177   | 0.0939 | 5.80     | 0.31    | 0.1106     | 0.1195   | 0.009920 | 0.9620 |

## Documentation

Full API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io).
See details for every class, method and plotting utility.

## Packaging (PyPI)

Build from each package directory (sdist + wheel) in a clean environment, preferably on Linux/CI for universal wheels:

```bash
python -m build --sdist --wheel --no-isolation
twine check dist/*
twine upload dist/*              # when ready to publish
```

Packages: `backend/interpy_synth`, `backend/interpy_bg`, `backend/fivedreg_tf`.

## License

MIT License
