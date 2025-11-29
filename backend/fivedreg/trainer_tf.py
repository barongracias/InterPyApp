# TensorFlow trainer mirroring the numpy-based Trainer API
from __future__ import annotations

import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from interpy_bg.trainer import Trainer as NumpyTrainer
from .tf_model import build_tf_model


class TrainerTF:
    """
    TensorFlow/Keras trainer for the 5D→1D regressor.
    Uses the same data loading/normalisation pipeline as the numpy trainer.
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
        lr_decay_patience: int = 10
        ) -> None:
        """
        Initialise the TensorFlow trainer.

        Args:
            directory: Directory where artifacts will be saved.
            hidden_sizes: Sizes of hidden layers.
            Lambda: L2 regularisation strength.
            epochs: Number of training epochs.
            learning_rate: Optimiser learning rate.
            train_val_split: Fraction of data for training (rest for validation).
            seed: Optional RNG seed.
            early_stop_patience: Optional epochs without improvement before stopping.
            lr_decay: Optional factor (<1) for ReduceLROnPlateau.
            lr_decay_patience: Patience for LR decay when lr_decay is set.
        """
        
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

        os.makedirs(self.directory, exist_ok=True)
        tf.random.set_seed(seed if seed is not None else 0)
        np.random.seed(seed if seed is not None else 0)

        self.model = build_tf_model(self.hidden_sizes, self.Lambda)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

    def _save_norm_vals(self, mean: np.ndarray, std: np.ndarray, filename: str = "normalisation_values_tf.npz") -> str:
        """
        Persist normalisation statistics for later inference.
        """
        
        path = os.path.join(self.directory, filename)
        np.savez(path, mean=mean, std=std)
        return path

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

    def train(self, pkl_path: str) -> Tuple[list[float], list[float]]:
        """
        Train the TensorFlow model. Returns train/val RMSE history.
        """
        
        # reuse numpy Trainer loader for validation/imputation/standardisation + splits
        splits = NumpyTrainer.load_dataset(
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
        )

        # save artifacts
        model_path = os.path.join(self.directory, "model_tf.keras")
        self.model.save(model_path)
        self._save_norm_vals(mean, std)

        train_rmse = [float(v) for v in history.history.get("rmse", [])]
        val_rmse = [float(v) for v in history.history.get("val_rmse", [])]
        return train_rmse, val_rmse
