# interpy_bg (NumPy)

NumPy implementation of the 5D → 1D regressor with a simple training/testing API.

## Modules
- `neural_network.py`: core feedforward net (5 inputs → hidden → 1 output).
- `trainer.py`: data validation/standardisation, training loop (Adam, L2, early stop/lr decay), artifacts/plots.
- `tester.py`: loads saved weights/norm stats to run predictions.
- `plotter.py`: loss/prediction plots.
- `logger.py`, `utils.py`: logging and helpers.
- Synthetic data generator is provided by the `interpy_synth` package (dependency).

## Usage
```python
import os, pickle
from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester
from interpy_synth import synthetic_5d_pickle

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

# create a training pickle (dict with X, y)
train_pkl = synthetic_5d_pickle(os.path.join(out_dir, "train.pkl"), n=1000, seed=42)

# train
trainer = Trainer(
    directory=out_dir,
    hidden_sizes=[16, 8],
    epochs=300,
    learning_rate=0.01,
    Lambda=0.01,
    train_val_split=0.8,
    activation="relu",
    weight_init="auto",   # auto/he/xavier
    batch_size=64,
    grad_clip=5.0,
    early_stop_patience=20,
    lr_decay=0.98,
    seed=42,
)
train_loss, val_loss = trainer.train(train_pkl)

# test
tester = Tester(hidden_sizes=[16, 8], Lambda=0.01, directory=out_dir)
with open(train_pkl, "rb") as f:
    data = pickle.load(f)
y_pred = tester.predict(data["X"])
```

Artifacts: `model_weights.npz`, `normalisation_values.npz`, `model_metadata.json`, `rmse_vs_epochs.png`, `ytrue_vs_ypred.png`. Metadata includes architecture, regularisation, activation/init, batch/clip/seed, best metrics, and R². You can serve artifacts via the FastAPI `/artifacts` and `/plots` endpoints.

Docker (whole app):

```bash
cd ../..
./scripts/docker_build.sh
./scripts/docker_up.sh   # backend on :8000
# ./scripts/docker_down.sh to stop
```

Notes:
- Plotting uses the headless `Agg` backend for compatibility with servers/CI.
- `Trainer.load_raw_data` loads X/y as `float32` by default (suitable for NumPy and TF interop).
- ML backend runs on CPU only; no GPU is required.
- For reproducibility, use `pip install -r requirements.lock` from the repo root/backend.
- Consider batch sizes/epochs appropriate to your hardware; outputs/ and uploads/ are mountable via Docker volumes.
