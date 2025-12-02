"""TensorFlow tester mirroring the NumPy-based Tester API for the 5D→1D regressor."""

# TensorFlow tester mirroring the numpy-based Tester API
from __future__ import annotations

import os
import pickle
from typing import Union

import numpy as np
import tensorflow as tf

from .logger import get_console_logger
from .utils import log_call, timer


class TesterTF:
    __test__ = False  # prevent pytest from collecting this as a test class
    """
    TensorFlow tester for the 5D→1D regressor.

    The tester loads the persisted Keras model and associated normalisation statistics
    produced by :class:`TrainerTF`, then provides a thin wrapper to run normalised
    inference on new data.
    """

    def __init__(self, directory: str) -> None:
        """
        Initialise the tester with saved model artifacts.

        Args:
            directory: Directory containing ``model_tf.keras`` and
                ``normalisation_values_tf.npz`` produced by training.

        Raises:
            FileNotFoundError: If either the model or normalisation archive cannot be found.
        """
        
        self.directory = directory
        self.model_path = os.path.join(self.directory, "model_tf.keras")
        self.norm_path = os.path.join(self.directory, "normalisation_values_tf.npz")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorFlow model not found at {self.model_path}")
        if not os.path.exists(self.norm_path):
            raise FileNotFoundError(f"Normalisation values not found at {self.norm_path}")

        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info("TesterTF initialised")

        self.model = tf.keras.models.load_model(self.model_path)
        norm = np.load(self.norm_path)
        self.mean = norm["mean"]
        self.std = norm["std"]

    @log_call
    def _load_input(self, X_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Load and validate input data from a NumPy array or pickle file path.

        Args:
            X_data: Either an array-like object of shape ``(N, 5)`` or a path to a
                ``.pkl`` file containing such an array.

        Returns:
            ``np.ndarray`` with shape ``(N, 5)`` and dtype ``float32``.

        Raises:
            ValueError: If the provided path is not a ``.pkl`` file or the array does
                not have five feature columns.
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

    @timer
    @log_call
    def predict(self, X_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Run inference on normalised input data using the saved Keras model.

        The method loads/validates input data, applies the stored mean/std scaling, and
        returns model predictions.

        Args:
            X_data: NumPy array of shape ``(N, 5)`` or path to a ``.pkl`` file containing
                such an array.

        Returns:
            Array of predicted targets with shape ``(N, 1)``.

        Raises:
            ValueError: Propagated from ``_load_input`` when the supplied data is invalid.
        """
        
        X = self._load_input(X_data)
        X_norm = (X - self.mean) / self.std
        y_pred = self.model.predict(X_norm, verbose=0)
        self.logger.info(f"Generated predictions for {len(X)} samples (TF).")
        return y_pred
