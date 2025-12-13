"""TensorFlow trainer mirroring the NumPy-based Trainer API for the 5D→1D regressor."""
from __future__ import annotations

import os
import pickle
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from .logger import get_console_logger
from .utils import log_call, timer
from .tf_model import build_tf_model
from .plotter import plot_loss, plot_predictions

class TrainerTF:
    """
    TensorFlow/Keras trainer for the 5D→1D regressor.

    The trainer mirrors the NumPy-based implementation but builds and optimises a
    Keras model. It handles data loading, standardisation, train/validation splitting,
    training with optional callbacks, metric tracking, and artifact persistence
    (model weights, normalisation stats, plots, and metadata).
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
        activation: str = "relu",
        weight_init: str = "auto",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        early_stop_patience: Optional[int] = None,
        lr_decay: Optional[float] = None,
        lr_decay_patience: int = 10,
        batch_size: Optional[int] = 64,
        grad_clip: Optional[float] = 5.0,
    ) -> None:
        """
        Configure a TensorFlow training run and initialise the model.

        Args:
            directory: Output directory for saved artifacts (model, plots, logs, metadata).
            hidden_sizes: Width of each hidden Dense layer, ordered from input to output.
            Lambda: L2 regularisation strength applied to Dense layer kernels.
            epochs: Maximum number of training epochs.
            learning_rate: Initial learning rate for Adam.
            train_val_split: Fraction of data used for training; remainder is validation.
            seed: Random seed for deterministic weight initialisation and shuffling.
            activation: Hidden-layer activation (relu, leakyrelu, tanh, sigmoid).
            weight_init: Weight initializer ("auto", "he", "xavier").
            beta1: Adam first-moment decay.
            beta2: Adam second-moment decay.
            epsilon: Adam numerical stability term.
            early_stop_patience: Number of epochs to wait for validation improvement before
                stopping; ``None`` disables early stopping.
            lr_decay: Multiplicative factor applied when validation loss plateaus; ``None``
                disables learning-rate decay.
            lr_decay_patience: Number of epochs with no improvement before applying decay.
            batch_size: Mini-batch size; ``None`` uses full-batch training.
            grad_clip: Gradient clipping norm applied in the optimizer; ``None`` disables clipping.

        Raises:
            ValueError: If any numeric hyperparameters are non-positive or invalid, or if
                split parameters fall outside expected ranges.
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
        if activation.lower() not in {"sigmoid", "tanh", "relu", "leakyrelu"}:
            raise ValueError("activation must be sigmoid, tanh, relu, or leakyrelu.")
        if weight_init.lower() not in {"auto", "he", "xavier"}:
            raise ValueError("weight_init must be auto, he, or xavier.")
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
        if not 0 < beta1 < 1:
            raise ValueError("beta1 must be between 0 and 1.")
        if not 0 < beta2 < 1:
            raise ValueError("beta2 must be between 0 and 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.directory = directory
        self.hidden_sizes = list(hidden_sizes)
        self.Lambda = Lambda
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_val_split = train_val_split
        self.seed = seed
        self.activation = activation
        self.weight_init = weight_init
        self.early_stop_patience = early_stop_patience
        self.lr_decay = lr_decay
        self.lr_decay_patience = lr_decay_patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.best_epoch: Optional[int] = None
        self.best_val_rmse: Optional[float] = None
        self.best_train_rmse: Optional[float] = None
        self.baseline_rmse: Optional[float] = None
        self.final_train_r2: Optional[float] = None
        self.final_val_r2: Optional[float] = None

        os.makedirs(self.directory, exist_ok=True)
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info(
            f"TrainerTF initialised: hidden_sizes={self.hidden_sizes}, Lambda={Lambda}, epochs={epochs}, "
            f"lr={learning_rate}, train_val_split={train_val_split}, seed={seed}, early_stop={early_stop_patience}, "
            f"lr_decay={lr_decay}, batch_size={batch_size}, grad_clip={grad_clip}"
        )
        tf.random.set_seed(seed if seed is not None else 0)
        np.random.seed(seed if seed is not None else 0)

        self.model = build_tf_model(self.hidden_sizes, self.Lambda, activation=self.activation, weight_init=self.weight_init)
        adam_kwargs = {}
        if self.grad_clip is not None:
            adam_kwargs["clipnorm"] = self.grad_clip
        optimizer_cls = (
            tf.keras.optimizers.legacy.Adam
            if hasattr(tf.keras.optimizers, "legacy")
            else tf.keras.optimizers.Adam
        )
        self.model.compile(
            optimizer=optimizer_cls(
                learning_rate=self.learning_rate,
                beta_1=beta1,
                beta_2=beta2,
                epsilon=epsilon,
                **adam_kwargs,
            ),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )

    @log_call
    def _save_norm_vals(self, mean: np.ndarray, std: np.ndarray, filename: str = "normalisation_values_tf.npz") -> str:
        """
        Persist feature normalisation statistics for reuse during inference or evaluation.

        Args:
            mean: Per-feature mean computed on the training split.
            std: Per-feature standard deviation computed on the training split.
            filename: Name of the NumPy archive to create in ``self.directory``.

        Returns:
            Absolute path to the saved ``.npz`` file.
        """
        path = os.path.join(self.directory, filename)
        np.savez(path, mean=mean, std=std)
        return path

    @staticmethod
    def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute root mean squared error between predicted and true targets.

        Args:
            y_true: Ground-truth targets of shape ``(N, 1)``.
            y_pred: Predicted targets of shape ``(N, 1)``.

        Returns:
            RMSE as a Python ``float``.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the coefficient of determination (R²) between predictions and targets.

        Args:
            y_true: Ground-truth targets of shape ``(N, 1)``.
            y_pred: Predicted targets of shape ``(N, 1)``.

        Returns:
            R² score as a Python ``float``. Returns ``nan`` when variance is zero.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    def _save_metadata(self, metadata: dict, filename: str = "tf_model_metadata.json") -> None:
        """
        Save TensorFlow training metadata (architecture, metrics) to JSON.

        Args:
            metadata: Serializable metadata dictionary describing the training run.
            filename: JSON filename to create inside ``self.directory``.

        Returns:
            None. Writes a JSON file with the supplied metadata.
        """
        path = os.path.join(self.directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    @staticmethod
    def load_raw_data(pkl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and validate raw feature/target arrays from a pickle file.

        The pickle may contain either a dictionary with ``"X"`` and ``"y"`` keys or a
        two-element tuple/list ``(X, y)``. Missing values are imputed column-wise using
        the mean; if an entire column is NaN an error is raised.

        Args:
            pkl_path: Path to a ``.pkl`` file containing training data.

        Returns:
            Tuple ``(X, y)`` where ``X`` has shape ``(N, 5)`` and ``y`` has shape ``(N, 1)``.

        Raises:
            FileNotFoundError: If ``pkl_path`` does not exist.
            ValueError: When the file extension is not ``.pkl``, the payload does not
                contain the expected data structure, shapes are inconsistent, feature
                dimensionality is not 5, or missing values cannot be imputed.
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
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.float32)
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            X = np.array(data[0], dtype=np.float32)
            y = np.array(data[1], dtype=np.float32)
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
        Load a dataset from pickle, validate shapes, split, and standardise features.

        Args:
            pkl_path: Path to pickle containing ``X`` and ``y`` data in either a dict or
                ``(X, y)`` tuple/list form.
            train_frac: Fraction of samples assigned to the training split.
            val_frac: Fraction assigned to validation.
            test_frac: Fraction assigned to testing.
            random_state: Seed for shuffling prior to splitting; ``None`` yields
                non-deterministic ordering.

        Returns:
            Dictionary with normalised splits and statistics:
            ``{\"X_train\", \"y_train\", \"X_val\", \"y_val\", \"X_test\", \"y_test\", \"mean\", \"std\"}``.

        Raises:
            ValueError: If split fractions are negative or do not sum to 1.0, if a split
                would be empty, or if the data payload fails validation in
                ``load_raw_data``.
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
        Fit the TensorFlow model on the provided dataset and persist artifacts.

        The method loads and standardises data, trains with optional early stopping and
        learning-rate decay, saves the trained model, normalisation statistics, loss and
        prediction plots, and writes a metadata JSON describing the run.

        Args:
            pkl_path: Path to a training pickle consumed by ``load_dataset``.

        Returns:
            Tuple ``(train_rmse, val_rmse)`` where each element is a list of RMSE values
            per epoch as reported by Keras.
        """
        splits = TrainerTF.load_dataset(
            pkl_path,
            train_frac=self.train_val_split,
            val_frac=1-self.train_val_split,
            test_frac=0.0,
            random_state=self.seed,
        )

        X_train = splits["X_train"]
        y_train = splits["y_train"].astype(np.float32)
        X_val = splits["X_val"]
        y_val = splits["y_val"].astype(np.float32)
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
        y_pred_val = self.model.predict(X_val, verbose=0)
        plot_loss(train_rmse, val_rmse, "rmse_vs_epochs.png", self.directory)
        plot_predictions(y_val, y_pred_val, "ytrue_vs_ypred.png", self.directory)

        # metrics and metadata
        self.best_epoch = int(np.argmin(val_rmse) + 1) if val_rmse else None
        self.best_val_rmse = float(np.min(val_rmse)) if val_rmse else None
        self.best_train_rmse = (
            float(train_rmse[self.best_epoch - 1]) if self.best_epoch and train_rmse else (float(train_rmse[-1]) if train_rmse else None)
        )
        baseline_pred = np.full_like(y_train, np.mean(y_train))
        self.baseline_rmse = self.calc_rmse(y_train, baseline_pred)

        y_pred_train = self.model.predict(X_train, verbose=0)
        self.final_train_r2 = self.calc_r2(y_train, y_pred_train)
        self.final_val_r2 = self.calc_r2(y_val, y_pred_val)

        tf_metadata = {
            "hidden_sizes": self.hidden_sizes,
            "Lambda": self.Lambda,
            "activation": self.activation,
            "weight_init": self.weight_init,
            "epochs_configured": self.epochs,
            "epochs_run": len(train_rmse),
            "best_epoch": self.best_epoch,
            "best_val_rmse": self.best_val_rmse,
            "best_train_rmse": self.best_train_rmse,
            "baseline_rmse": self.baseline_rmse,
            "final_train_r2": self.final_train_r2,
            "final_val_r2": self.final_val_r2,
            "learning_rate": self.learning_rate,
            "beta1": float(self.model.optimizer.beta_1.numpy()) if hasattr(self.model.optimizer, "beta_1") else None,
            "beta2": float(self.model.optimizer.beta_2.numpy()) if hasattr(self.model.optimizer, "beta_2") else None,
            "epsilon": self.model.optimizer.epsilon if hasattr(self.model.optimizer, "epsilon") else None,
            "batch_size": self.batch_size,
            "grad_clip": self.grad_clip,
            "early_stop_patience": self.early_stop_patience,
            "lr_decay": self.lr_decay,
            "seed": self.seed,
            "model_type": "tf",
        }
        self._save_metadata(tf_metadata)

        return train_rmse, val_rmse

    def _callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Build optional callbacks for early stopping and learning-rate decay.

        Returns:
            List of configured Keras callbacks (may be empty) respecting the trainer's
            early stopping and learning-rate decay settings.
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
