# fivedreg (TensorFlow)

TensorFlow/Keras implementation of the 5D → 1D regressor, mirroring the numpy-based `interpy_bg` API.

## Modules
- `tf_model.py`: `build_tf_model(hidden_sizes, Lambda)` builds a Sequential model with L2 regularisation.
- `trainer_tf.py`: `TrainerTF` loads/validates data (via `interpy_bg.Trainer.load_dataset`), trains a TF model, and saves `model_tf.keras` plus normalisation values.
- `tester_tf.py`: `TesterTF` loads the saved TF model and normalisation stats to make predictions on NumPy arrays or .pkl files.

## Usage
```python
from fivedreg.trainer_tf import TrainerTF
from fivedreg.tester_tf import TesterTF
from interpy_bg.synthetic import synthetic_5d_pickle
import os

out_dir = "outputs_tf"
os.makedirs(out_dir, exist_ok=True)
data_path = synthetic_5d_pickle(os.path.join(out_dir, "train.pkl"), n=1000, seed=42)

trainer = TrainerTF(directory=out_dir, hidden_sizes=[64, 32, 16], epochs=100, learning_rate=0.01)
train_rmse, val_rmse = trainer.train(data_path)

tester = TesterTF(directory=out_dir)
y_pred = tester.predict([0.1, 0.2, 0.3, 0.4, 0.5])
```

Note: Ensure TensorFlow is installed in your environment to use this package.