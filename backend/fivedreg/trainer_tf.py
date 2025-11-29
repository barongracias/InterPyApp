# TensorFlow trainer mirroring the numpy-based Trainer API
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import matplotlib

# ensure headless plotting
matplotlib.use("Agg")

from .logger import get_console_logger
from .utils import log_call, timer
from .tf_model import build_tf_model
from interpy_bg.plotter import plot_loss, plot_predictions


class TrainerTF:
    """
    TensorFlow/Keras trainer for the 5D→1D regressor.
    Uses its own data loading/normalisation pipeline mirroring the NumPy trainer.
    """

    def __init__(
        self,
        directory: str,
        hidden_sizes: Sequence[int],
        Lambda: float = 0.01,
        epochs: int = 200,
        learning_rate: float = 0.01,
        train_val_split: float = 0.8,
        seed: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
        lr_decay: Optional[float] = None,
        lr_decay_patience: int = 10,
        batch_size: Optional[int] = 64,
        grad_clip: Optional[float] = 5.0,
    ) -> None:
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError("Hidden sizes must be positive.")
        if Lambda <= 0:
            raise ValueError("Lambda must be positive.")
        if epochs <= 0:
            raise ValueError("Epochs must be positive.")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0 < train_val_split < 1:
            raise ValueError("train_val_split must be in (0, 1).")
        if early_stop_patience is not None and early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive if provided.")
        if lr_decay is not None and not (0 < lr_decay < 1):
            raise ValueError("lr_decay must be in (0, 1) if provided.")
        if lr_decay_patience <= 0:
            raise ValueError("lr_decay_patience must be positive.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive if provided.")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive if provided.")

        self.directory = directory
        self.hidden_sizes = list(hidden_sizes)
        self.Lambda = Lambda
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_val_split = train_val_split
        self.seed = seed
        self.early_stop_patience = early_stop_patience
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        os.makedirs(self.directory, exist_ok=True)
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info(
            f"TrainerTF initialised: hidden_sizes={self.hidden_sizes}, Lambda={Lambda}, epochs={epochs}, "
            f"lr={learning_rate}, train_val_split={train_val_split}, seed={seed}, early_stop={early_stop_patience}, "
            f"lr_decay={lr_decay}, batch_size={batch_size}, grad_clip={grad_clip}"
        )
        tf.random.set_seed(seed if seed is not None else 0)
        np.random.seed(seed if seed is not None else 0)

        self.model = build_tf_model(self.hidden_sizes, self.Lambda)
        adam_kwargs = {}
        if self.grad_clip is not None:
            adam_kwargs["clipnorm"] = self.grad_clip
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                **adam_kwargs,
            ),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

    @staticmethod
    def load_raw_data(pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate raw X, y arrays from pickle. Imputes NaNs with column means.
        """
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Training pickle file not found: {pkl_path}")
        if not pkl_path.endswith(".pkl"):
            raise ValueError("Training data must be provided as a .pkl file")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if "X" not in data or "y" not in data:
                raise ValueError("Pickle dictionary must contain 'X' and 'y' keys")
            X = np.array(data["X"], dtype=float)
            y = np.array(data["y"], dtype=float)
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            X = np.array(data[0], dtype=float)
            y = np.array(data[1], dtype=float)
        else:
            raise ValueError("Pickle file must contain a dict with 'X' and 'y', or a tuple/list (X, y)")

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
        if X.shape[1] != 5:
            raise ValueError(f"Expected input features to have 5 columns, got {X.shape[1]}")

        def _impute_nan(arr: np.ndarray) -> np.ndarray:
            col_mean = np.nanmean(arr, axis=0, keepdims=True)
            if np.isnan(col_mean).any():
                raise ValueError("Cannot impute missing values: column contains only NaNs")
            return np.where(np.isnan(arr), col_mean, arr)

        X = _impute_nan(X)
        y = _impute_nan(y)

        return X, y

    @staticmethod
    def load_dataset(
        pkl_path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        random_state: int | None = None
    ) -> dict:
        """
        Load and prepare dataset from a pickle file.

        - Accepts dict with keys "X" and "y" (metadata ignored) or legacy tuple/list (X, y)
        - Validates shapes (X: (N, 5), y: (N, 1))
        - Handles missing values by imputing column means (raises if an entire column is NaN)
        - Splits into train/val/test
        - Standardises features using train-set mean/std and returns splits
        """

        if train_frac < 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Split fractions must be non-negative")
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0")

        X, y = TrainerTF.load_raw_data(pkl_path)

        # shuffle
        N = X.shape[0]
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(N)
        X, y = X[idx], y[idx]

        # compute split indices
        train_end = int(N * train_frac)
        val_end = train_end + int(N * val_frac)
        if train_end == 0 or val_end == train_end:
            raise ValueError("Empty train or validation split; adjust split fractions")
        if val_end > N:
            val_end = N

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # standardise using train mean/std
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8  # avoid division by zero

        def _standardise(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            return (arr - mean) / std

        X_train_norm = _standardise(X_train)
        X_val_norm = _standardise(X_val)
        X_test_norm = _standardise(X_test)

        return {
            "X_train": X_train_norm,
            "y_train": y_train,
            "X_val": X_val_norm,
            "y_val": y_val,
            "X_test": X_test_norm,
            "y_test": y_test,
            "mean": mean,
            "std": std,
        }

    @timer
    @log_call
    def train(self, pkl_path: str) -> Tuple[list[float], list[float]]:
        """
        Train the TensorFlow model. Returns train/val RMSE history.
        """
        splits = TrainerTF.load_dataset(
            pkl_path,
            train_frac=self.train_val_split,
            val_frac=1 - self.train_val_split,
            test_frac=0.0,
            random_state=self.seed,
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_val = splits["X_val"]
        y_val = splits["y_val"]
        mean = splits["mean"]
        std = splits["std"]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            verbose=0,
            callbacks=self._callbacks(),
            batch_size=self.batch_size,
        )

        # save artifacts
        model_path = os.path.join(self.directory, "model_tf.keras")
        self.model.save(model_path)
        self._save_norm_vals(mean, std)

        train_rmse = [float(v) for v in history.history.get("rmse", [])]
        val_rmse = [float(v) for v in history.history.get("val_rmse", [])]

        # plots (use best weights if early stopping restored)
        y_pred_train = self.model.predict(X_train, verbose=0)
        y_pred_val = self.model.predict(X_val, verbose=0)
        plot_loss(train_rmse, val_rmse, "rmse_vs_epochs_tf.png", self.directory)
        plot_predictions(y_val, y_pred_val, "ytrue_vs_ypred_tf.png", self.directory)

        return train_rmse, val_rmse

    def _callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Build optional callbacks for early stopping and learning-rate decay.
        """
        callbacks: List[tf.keras.callbacks.Callback] = []
        if self.early_stop_patience:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_rmse",
                    mode="min",
                    patience=self.early_stop_patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        if self.lr_decay:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_rmse",
                    mode="min",
                    factor=self.lr_decay,
                    patience=self.lr_decay_patience,
                    min_lr=1e-6,
                    verbose=0,
                )
            )
        return callbacks
