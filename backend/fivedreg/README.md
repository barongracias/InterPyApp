# fivedreg (TensorFlow)

TensorFlow/Keras implementation of the 5D → 1D regressor, mirroring the numpy-based `interpy_bg` API.

## Modules
- `tf_model.py`: `build_tf_model(hidden_sizes, Lambda)` builds a Sequential model with L2 regularisation and He/Xavier init.
- `trainer_tf.py`: `TrainerTF` loads/validates data (via `interpy_bg.Trainer.load_dataset`), trains a TF model, and saves `model_tf.keras` plus normalisation values. Optional early stopping and LR decay.
- `tester_tf.py`: `TesterTF` loads the saved TF model and normalisation stats to make predictions on NumPy arrays or .pkl files.
- `logger.py`, `utils.py`: lightweight logging and decorators (independent from `interpy_bg`).

## Usage
```python
from fivedreg.trainer_tf import TrainerTF
from fivedreg.tester_tf import TesterTF
from interpy_bg.synthetic import synthetic_5d_pickle
import os

out_dir = "outputs"
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

Note: Ensure TensorFlow is installed in your environment to use this package. Training also saves plots (`rmse_vs_epochs_tf.png`, `ytrue_vs_ypred_tf.png`) to the `directory`.

### FastAPI usage
- `/train` supports `model_type=tf` to train and save TF artifacts into `backend/outputs/` (including TF plots).
- `/predict` accepts `model_type=tf` to run predictions using the TF model.
- `/artifacts/{filename}` serves TF artifacts (`model_tf.keras`, `normalisation_values_tf.npz`, `tf_model_metadata.json`) as well as NumPy ones.
