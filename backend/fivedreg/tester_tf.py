# TensorFlow tester mirroring the numpy-based Tester API
from __future__ import annotations

import os
import pickle
from typing import Union

import numpy as np
import tensorflow as tf


class TesterTF:
    """
    TensorFlow tester for the 5D→1D regressor. Loads saved Keras model and normalisation stats.
    """

    def __init__(self, directory: str) -> None:
        """
        Initialise the tester with saved model artifacts.

        Args:
            directory: Directory containing `model_tf.keras` and `normalisation_values_tf.npz`.
        """
        
        self.directory = directory
        self.model_path = os.path.join(self.directory, "model_tf.keras")
        self.norm_path = os.path.join(self.directory, "normalisation_values_tf.npz")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorFlow model not found at {self.model_path}")
        if not os.path.exists(self.norm_path):
            raise FileNotFoundError(f"Normalisation values not found at {self.norm_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        norm = np.load(self.norm_path)
        self.mean = norm["mean"]
        self.std = norm["std"]

    def _load_input(self, X_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Load and validate input data from array or pickle path.
        """
        
        if isinstance(X_data, str):
            if not X_data.endswith(".pkl"):
                raise ValueError("String path must end with .pkl")
            with open(X_data, "rb") as f:
                X = pickle.load(f)
        else:
            X = np.asarray(X_data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 5:
            raise ValueError(f"Expected input shape (N, 5), got {X.shape}")
        return np.asarray(X, dtype=np.float32)

    def predict(self, X_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Run inference on normalised input data.

        Args:
            X_data: NumPy array of shape (N,5) or path to .pkl file containing such an array.

        Returns:
            Predicted targets of shape (N, 1).
        """
        
        X = self._load_input(X_data)
        X_norm = (X - self.mean) / self.std
        y_pred = self.model.predict(X_norm, verbose=0)
        return y_pred