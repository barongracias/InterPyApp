# fivedreg_tf (TensorFlow)

TensorFlow/Keras implementation of the 5D → 1D regressor with a simple training/testing API.

- Docs: https://interpyapp.readthedocs.io/en/latest/index.html (package reference lives on RTD)
- Source: https://github.com/barongracias/InterPyApp

## Modules
- `tf_model.py`: `build_tf_model(hidden_sizes, Lambda)` builds a Sequential model with L2 regularisation and He/Xavier init.
- `trainer_tf.py`: `TrainerTF` loads/validates data, trains a TF model, and saves `model_tf.keras` plus normalisation values. Optional early stopping, LR decay, batch size, and grad clipping.
- `tester_tf.py`: `TesterTF` loads the saved TF model and normalisation stats to make predictions on NumPy arrays or .pkl files.
- `logger.py`, `utils.py`: lightweight logging and decorators.
- Synthetic data examples use the separate `interpy_synth` package (installed automatically).

## Installation (package)

PyPI:
```bash
pip install fivedreg_tf        # pulls interpy-synth + tensorflow-cpu/macos
```

From source (editable):
```bash
pip install -e .
```

Headless environments: plotting is configured with the `Agg` backend, so no display is required.
GPU is not required or supported; on macOS use `tensorflow-macos` (installed automatically via platform marker), and on Linux/Windows use `tensorflow-cpu`.

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
    activation="relu",
    weight_init="auto",
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    early_stop_patience=10,
    lr_decay=0.95,
    seed=42,
)
train_rmse, val_rmse = trainer.train(data_path)

tester = TesterTF(directory=out_dir)
y_pred = tester.predict([0.1, 0.2, 0.3, 0.4, 0.5])
```

Note: Ensure TensorFlow is installed in your environment to use this package. Training also saves plots (`rmse_vs_epochs.png`, `ytrue_vs_ypred.png`) to the `directory`.
Metadata (`tf_model_metadata.json`) includes hidden sizes, Lambda, activation/init, learning rate, Adam betas/epsilon, batch/clip, epochs run, best epoch, best train/val RMSE, baseline RMSE, and final train/val R².

Performance/ops tips:
- CPU-only build; choose modest hidden sizes/batch sizes for constrained CPUs.
- Batch size and grad clipping can help stabilise small datasets (see tests for small-batch config).
- Optimiser: defaults to `tf.keras.optimizers.legacy.Adam` when available (avoids the slower Apple Silicon path) and falls back to `tf.keras.optimizers.Adam` otherwise.

Hyperparameter guide (UI/API)
-----------------------------
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

### Documentation

Full package API documentation is hosted on [ReadTheDocs](https://interpyapp.readthedocs.io). Use RTD for full reference and examples.

## License

MIT License