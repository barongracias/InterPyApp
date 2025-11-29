# TensorFlow trainer mirroring the numpy-based Trainer API
import os
import numpy as np
import tensorflow as tf
import pickle
from typing import Tuple

from .tf_model import build_tf_model


class TrainerTF:
    """
    TensorFlow/Keras trainer for the 5D→1D regressor.
    Uses the same data loading/normalisation pipeline as the numpy trainer.
    """

    def __init__(
        self,
        directory: str,
        hidden_sizes: list[int],
        Lambda: float = 0.01,
        epochs: int = 200,
        learning_rate: float = 0.01,
        train_val_split: float = 0.8,
        seed: int | None = None,
    ):
        self.directory = directory
        self.hidden_sizes = hidden_sizes
        self.Lambda = Lambda
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_val_split = train_val_split
        self.seed = seed

        os.makedirs(self.directory, exist_ok=True)
        tf.random.set_seed(seed if seed is not None else 0)

        self.model = build_tf_model(hidden_sizes, Lambda)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

    def _save_norm_vals(self, mean: np.ndarray, std: np.ndarray, filename: str = "normalisation_values_tf.npz"):
        path = os.path.join(self.directory, filename)
        np.savez(path, mean=mean, std=std)
        return path

    def train(self, pkl_path: str) -> Tuple[list[float], list[float]]:
        """
        Train the TensorFlow model. Returns train/val RMSE history.
        """
        # load and validate data (dict with X/y or tuple)
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

        # impute NaNs
        def _impute_nan(arr: np.ndarray) -> np.ndarray:
            col_mean = np.nanmean(arr, axis=0, keepdims=True)
            if np.isnan(col_mean).any():
                raise ValueError("Cannot impute missing values: column contains only NaNs")
            return np.where(np.isnan(arr), col_mean, arr)

        X = _impute_nan(X)
        y = _impute_nan(y)

        # shuffle and split
        N = X.shape[0]
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(N)
        X, y = X[idx], y[idx]

        train_end = int(N * self.train_val_split)
        val_end = N
        if train_end == 0 or train_end == val_end:
            raise ValueError("Empty train or validation split; adjust train_val_split")

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        # standardise using train stats
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            verbose=0,
        )

        # save artifacts
        model_path = os.path.join(self.directory, "model_tf.keras")
        self.model.save(model_path)
        self._save_norm_vals(mean, std)

        train_rmse = history.history.get("rmse", [])
        val_rmse = history.history.get("val_rmse", [])
        return train_rmse, val_rmse
