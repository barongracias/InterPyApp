# imports
import numpy as np
import os
import pickle

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
                 epsilon: float = 1e-8
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
    def load_train_data(pkl_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load training data from a pickle file.

        The pickle must contain a tuple (X, y).

        Args:
            pkl_path (str): Path to the pickle file containing training data.

        Returns:
            tuple: (X, y) as NumPy arrays
        """
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Training pickle file not found: {pkl_path}")
        if not pkl_path.endswith(".pkl"):
            raise ValueError("Training data must be provided as a .pkl file")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError("Pickle file must contain a tuple (X, y)")

        X = np.array(data[0], dtype=float)
        y = np.array(data[1], dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
        if X.shape[1] != 5:
            raise ValueError(f"Expected input features to have 5 columns, got {X.shape[1]}")

        return X, y
    
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

    @log_call
    @timer
    def train(self, pkl_path: str) -> tuple[list[float], list[float]]:
        """
        Train the neural network using gradient descent with Adam optimiser and track RMSE. Saves RMSE vs epochs plot.

        Args:
            pkl_path (str): Path to .pkl file containing training data (X, y).

        Returns:
            tuple[list[float], list[float]]: Training and validation RMSE histories.
        """
        
        X, y = self.load_train_data(pkl_path)
        self.logger.info(f"Loaded training data: {X.shape[0]} samples")

        # shuffle
        idx = np.random.permutation(X.shape[0])
        X, y = X[idx], y[idx]
        
        # split into train/val
        N = X.shape[0]
        split_index = int(N * self.train_val_split)
        if split_index == 0 or split_index == N:
            raise ValueError("Empty train or val set, adjust train_val_split")
        
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # normalise data
        self.norm_vals(X_train)
        X_train_norm = self.normalise(X_train)
        X_val_norm = self.normalise(X_val)
        
        # Adam cache
        mb = self.mb
        vb = self.vb
        
        # hyperparams cache
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.epsilon
        lr = self.learning_rate
        
        # initialise history arrays
        train_loss_hist = []
        val_loss_hist = []
        
        # logging checkpoints -> print every 20 times
        log_every = max(1, self.epochs//20)
        
        # training loop
        for epoch in range(self.epochs):
            self.logger.debug(f"Epoch {epoch+1}/{self.epochs} starting")

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

            # periodic logging
            if (epoch + 1) % log_every == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}"
                )
                
        # save loss histories
        self.train_loss_history = train_loss_hist
        self.val_loss_history = val_loss_hist
        
        # save norm vals and weights
        self.save_norm_vals("normalisation_values.npz", self.directory)
        self.save_weights("model_weights.npz", self.directory)
        
        # save RMSE vs Epoch plot and y_true vs y_preds plot for final epoch
        plot_loss(train_loss_hist, val_loss_hist, "rmse_vs_epochs.png", self.directory)
        plot_predictions(y_train, y_pred_train, "ytrue_vs_ypred.png", self.directory)
        
        self.logger.debug("Training complete")
        return train_loss_hist, val_loss_hist