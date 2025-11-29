# imports
import json
import numpy as np
import os
import pickle
from typing import Tuple

# local imports
from .neural_network import NeuralNetwork
from .plotter import plot_loss, plot_predictions
from .logger import get_console_logger
from .utils import timer, log_call

class Trainer(NeuralNetwork):
    """
    Trainer class for feedforward neural network.

    Inherits from NeuralNetwork and adds functionality for training, calculating RMSE and saving trained models.

    Attributes:
        epochs (int): Number of training iterations.
        learning_rate (float): Step size for gradient descent updates.
        train_val_split (float): Fraction of data used for training.
        train_loss_history (list[float]): RMSE per epoch for training set.
        val_loss_history (list[float]): RMSE per epoch for validation set.
        directory (str): Directory path to save output files.
        beta1 (float): Exponential decay rate for first-moment estimates in Adam.
        beta2 (float): Exponential decay rate for second-moment estimates in Adam.
        epsilon (float): Small constant for numerical stability in Adam.
        mean (np.ndarray | None): Mean of the input training data
        std (np.ndarray | None): Standard deviation of the input training data
    """
    
    def __init__(self,
                 directory: str,
                 hidden_sizes: list[int],
                 Lambda: float = 0.01,
                 epochs: int = 1000,
                 learning_rate: float = 0.01,
                 train_val_split: float = 0.8,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 early_stop_patience: int | None = None,
                 lr_decay: float | None = None
                 ):
        """
        Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

        Args:
            directory (str): Directory path to save output files.
            hidden_sizes (list[int]): Number of neurons in each hidden layer.
            Lambda (float): L2 regularization parameter.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for gradient descent.
            train_val_split (float): Fraction of dataset used for training.
            beta1 (float): Adam first-moment decay rate (default 0.9).
            beta2 (float): Adam second-moment decay rate (default 0.999).
            epsilon (float): Small constant to avoid divide-by-zero in Adam.
        """
        
        super().__init__(hidden_sizes, Lambda, directory)
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.train_val_split: float = train_val_split
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.early_stop_patience: int | None = early_stop_patience
        self.lr_decay: float | None = lr_decay
        self.best_val_rmse: float | None = None
        self.best_train_rmse: float | None = None
        self.best_epoch: int | None = None
        self.baseline_rmse: float | None = None
        
        # adam hyperparams
        self.beta1: float = float(beta1)
        self.beta2: float = float(beta2)
        self.epsilon: float = float(epsilon)
        
        # check adam hyperparams
        if not 0 < self.beta1 < 1:
            raise ValueError("beta1 must be between 0 and 1")
        if not 0 < self.beta2 < 1:
            raise ValueError("beta2 must be between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.early_stop_patience is not None and self.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive or None")
        if self.lr_decay is not None and not (0 < self.lr_decay < 1):
            raise ValueError("lr_decay must be between 0 and 1 if provided")
        self.logger.info(f"Initialised Adam hyperparams: beta1: {self.beta1}, beta2: {self.beta2}, epsilon: {self.epsilon}")
        
        # initialise bias moments for Adam (weights' m and v are initialised in NeuralNetwork)
        self.mb: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        self.vb: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        self.logger.debug(f"Initialised Adam moments: mb: {self.mb}, vb: {self.vb}")
        
        # set for later
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        
        # logger
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info(f"Trainer initialised: epochs={epochs}, lr={learning_rate}, train_val_split={train_val_split}")
    
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
            # legacy support for tuple/list format (X, y)
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
    def dataset_stats(pkl_path: str) -> dict:
        """
        Return minimal dataset stats for preview: rows, features, min/max for X and y.
        """
        X, y = Trainer.load_raw_data(pkl_path)
        stats = {
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "x_min": [float(v) for v in np.min(X, axis=0)],
            "x_max": [float(v) for v in np.max(X, axis=0)],
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
        }
        return stats

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

        Args:
            pkl_path (str): Path to the pickle file containing training data.
            train_frac (float): Fraction of data for training.
            val_frac (float): Fraction of data for validation.
            test_frac (float): Fraction of data for testing.
            random_state (int | None): Optional seed for reproducible shuffling.

        Returns:
            dict: {
                "X_train": np.ndarray,
                "y_train": np.ndarray,
                "X_val": np.ndarray,
                "y_val": np.ndarray,
                "X_test": np.ndarray,
                "y_test": np.ndarray,
                "mean": np.ndarray,
                "std": np.ndarray,
            }
        """
        
        if train_frac < 0 or val_frac < 0 or test_frac < 0:
            raise ValueError("Split fractions must be non-negative")
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0")

        X, y = Trainer.load_raw_data(pkl_path)

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
    
    @log_call
    def norm_vals(self, X_train: np.ndarray) -> None:
        """
        Calculate and store mean and standard deviation into instance for normalisation.
        
        Args:
            X_train (np.ndarray): Input training data.
        """
        
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0) + 1e-8   # avoid zero div for zero std
    
    @log_call
    def normalise(self, X: np.ndarray) -> np.ndarray:
        """
        Normalise input using mean and standard deviation.
        
        Args:
            X (np.ndarray): Input data, shape (N, 5).
        
        Returns:
            np.ndarray: Normalised input data, shape (N, 5).
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalisation values not set")
        return (X - self.mean) / self.std
    
    @log_call
    def save_norm_vals(self, filename: str = "normalisation_values.npz", directory: str = None) -> None:
        """
        Save normalisation values to outputs.

        Args:
            filename (str): Name of the file.
            directory (str): Directory path to save file.
        """
        
        # verify path
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        np.savez(path, mean=self.mean, std=self.std)
        self.logger.info(f"Normalisation values saved to {path}")

    @log_call
    def save_metadata(self, filename: str = "model_metadata.json", directory: str = None) -> None:
        """
        Save model metadata needed for inference (architecture and regularisation).

        Args:
            filename (str): Name of the metadata file.
            directory (str): Directory path to save file.
        """
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        metadata = {
            "hidden_sizes": self.hidden_sizes,
            "Lambda": self.Lambda,
            "best_val_rmse": self.best_val_rmse,
            "best_train_rmse": self.best_train_rmse,
            "best_epoch": self.best_epoch,
            "baseline_rmse": self.baseline_rmse,
            "epochs_configured": self.epochs,
            "lr_decay": self.lr_decay,
            "early_stop_patience": self.early_stop_patience,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        self.logger.info(f"Model metadata saved to {path}")
    
    @staticmethod
    def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate root mean squared error between predictions and true values.

        Args:
            y_true (np.ndarray): True target values, shape (N, 1).
            y_pred (np.ndarray): Predicted values, shape (N, 1).

        Returns:
            float: RMSE value.
        """
        
        return np.sqrt(np.mean((y_true - y_pred)**2))

    @timer
    def train(self, pkl_path: str) -> tuple[list[float], list[float]]:
        """
        Train the neural network using gradient descent with Adam optimiser and track RMSE. Saves RMSE vs epochs plot.

        Args:
            pkl_path (str): Path to .pkl file containing training data (X, y).

        Returns:
            tuple[list[float], list[float]]: Training and validation RMSE histories.
        """
        
        splits = self.load_dataset(
            pkl_path,
            train_frac=self.train_val_split,
            val_frac=1 - self.train_val_split,
            test_frac=0.0,
        )
        X_train_norm = splits["X_train"]
        y_train = splits["y_train"]
        X_val_norm = splits["X_val"]
        y_val = splits["y_val"]

        # cache norm stats for saving/predict
        self.mean = splits["mean"]
        self.std = splits["std"]
        # baseline: predict train mean
        baseline_pred = np.full_like(y_train, y_train.mean())
        self.baseline_rmse = float(self.calc_rmse(y_train, baseline_pred))
        self.logger.info(f"Baseline (mean) train RMSE: {self.baseline_rmse:.4f}")

        self.logger.info(f"Loaded training data: {y_train.shape[0] + y_val.shape[0]} samples")
        self.logger.info(f"Training samples: {len(X_train_norm)}, Validation samples: {len(X_val_norm)}")
        
        # Adam cache
        mb = self.mb
        vb = self.vb
        
        # hyperparams cache
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.epsilon
        base_lr = self.learning_rate
        
        # initialise history arrays
        train_loss_hist = []
        val_loss_hist = []
        best_weights = None
        best_biases = None
        best_epoch = None
        best_val = np.inf
        best_train = None
        patience_counter = 0
        
        # logging checkpoints -> print every 20 times
        log_every = max(1, self.epochs//20)
        
        # training loop
        for epoch in range(self.epochs):
            self.logger.debug(f"Epoch {epoch+1}/{self.epochs} starting")
            lr = base_lr * (self.lr_decay ** epoch) if self.lr_decay else base_lr

            # timestep for Adam
            self.t += 1
            t = self.t

            # apply forward pass
            y_pred_train = self.forward(X_train_norm)

            # apply backprop to return gradients
            dW, db = self.backprop(X_train_norm, y_train, y_pred_train)

            # Adam update
            for i in range(len(self.weights)):
                # update first moment estimates
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * dW[i]
                mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]

                # update second moment estimates
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (dW[i] ** 2)
                vb[i] = beta2 * vb[i] + (1 - beta2) * (db[i] ** 2)

                # bias-corrected estimates
                m_hat = self.m[i] / (1 - beta1 ** t)
                v_hat = self.v[i] / (1 - beta2 ** t)
                mb_hat = mb[i] / (1 - beta1 ** t)
                vb_hat = vb[i] / (1 - beta2 ** t)

                # update parameters
                self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

            # validation predictions
            y_pred_val = self.forward(X_val_norm)

            # RMSE
            train_rmse = self.calc_rmse(y_train, y_pred_train)
            val_rmse = self.calc_rmse(y_val, y_pred_val)

            train_loss_hist.append(train_rmse)
            val_loss_hist.append(val_rmse)
            if val_rmse < best_val:
                best_val = val_rmse
                best_train = train_rmse
                best_epoch = epoch + 1
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0
            else:
                patience_counter += 1

            # periodic logging
            if (epoch + 1) % log_every == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}"
                )
            if self.early_stop_patience and patience_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1} (no val improvement in {self.early_stop_patience} epochs)")
                break
        
        # restore best weights if captured
        if best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases
            self.logger.info(f"Restored best weights from epoch {best_epoch} with Val RMSE={best_val:.4f}")
            final_epoch = best_epoch
        else:
            final_epoch = len(train_loss_hist)
            best_val = val_loss_hist[-1] if val_loss_hist else None
            best_train = train_loss_hist[-1] if train_loss_hist else None

        # save loss histories
        self.train_loss_history = train_loss_hist
        self.val_loss_history = val_loss_hist
        self.best_val_rmse = float(best_val) if best_val is not None else None
        self.best_train_rmse = float(best_train) if best_train is not None else None
        self.best_epoch = int(best_epoch) if best_epoch is not None else final_epoch
        
        # save norm vals and weights
        self.save_norm_vals("normalisation_values.npz", self.directory)
        self.save_weights("model_weights.npz", self.directory)
        self.save_metadata("model_metadata.json", self.directory)
        
        # save RMSE vs Epoch plot and y_true vs y_preds plot for final epoch
        plot_loss(train_loss_hist, val_loss_hist, "rmse_vs_epochs.png", self.directory)
        plot_predictions(y_train, y_pred_train, "ytrue_vs_ypred.png", self.directory)
        
        self.logger.debug("Training complete")
        return train_loss_hist, val_loss_hist
