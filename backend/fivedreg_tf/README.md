# fivedreg_tf (TensorFlow)

TensorFlow/Keras implementation of the 5D → 1D regressor with a simple training/testing API.

## Modules
- `tf_model.py`: `build_tf_model(hidden_sizes, Lambda)` builds a Sequential model with L2 regularisation and He/Xavier init.
- `trainer_tf.py`: `TrainerTF` loads/validates data, trains a TF model, and saves `model_tf.keras` plus normalisation values. Optional early stopping, LR decay, batch size, and grad clipping.
- `tester_tf.py`: `TesterTF` loads the saved TF model and normalisation stats to make predictions on NumPy arrays or .pkl files.
- `logger.py`, `utils.py`: lightweight logging and decorators.
- Synthetic data examples use the separate `interpy_synth` package (installed automatically).

## Installation (package)
From this `backend/fivedreg_tf` directory:
```bash
pip install -r requirements.lock  # pinned CPU-only deps
pip install .
```

Headless environments: plotting is configured with the `Agg` backend, so no display is required.
GPU is not required or supported; the package depends on `tensorflow-cpu`.
For reproducibility, install via the pinned `requirements.lock` in `backend/`.

Docker (whole app):

```bash
cd ../..
./scripts/docker_build.sh
./scripts/docker_up.sh   # backend on :8000 (includes TF if built with fivedreg_tf)
```

## Usage
```python
from fivedreg_tf.trainer_tf import TrainerTF
from fivedreg_tf.tester_tf import TesterTF
from interpy_synth import synthetic_5d_pickle
import os

out_dir = "outputs_tf"
os.makedirs(out_dir, exist_ok=True)
data_path = synthetic_5d_pickle(os.path.join(out_dir, "train.pkl"), n=1000, seed=42)

trainer = TrainerTF(
    directory=out_dir,
    hidden_sizes=[64, 32, 16],
    epochs=100,
    learning_rate=0.01,
    early_stop_patience=10,
    lr_decay=0.95,
    seed=42,
)
train_rmse, val_rmse = trainer.train(data_path)

tester = TesterTF(directory=out_dir)
y_pred = tester.predict([0.1, 0.2, 0.3, 0.4, 0.5])
```

Note: Ensure TensorFlow is installed in your environment to use this package. Training also saves plots (`rmse_vs_epochs.png`, `ytrue_vs_ypred.png`) to the `directory`.
Metadata (`tf_model_metadata.json`) includes hidden sizes, Lambda, epochs run, best epoch, best train/val RMSE, baseline RMSE, and final train/val R².

Performance/ops tips:
- CPU-only build; choose modest hidden sizes/batch sizes for constrained CPUs.
- Batch size and grad clipping can help stabilise small datasets (see tests for small-batch config).
- Use `requirements.lock` for reproducibility; mount outputs_tf via Docker volumes in production.

### FastAPI usage
- `/train` supports `model_type=tf` to train and save TF artifacts into `backend/outputs_tf/` (including TF plots) when running the API.
- `/predict` accepts `model_type=tf` to run predictions using the TF model.
- `/artifacts/{filename}` serves TF artifacts (`model_tf.keras`, `normalisation_values_tf.npz`, `tf_model_metadata.json`) as well as NumPy ones.
