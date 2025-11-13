# imports
import numpy as np
import os

# local imports
from .neural_network import NeuralNetwork
from .plotter import plot_loss, plot_predictions
from .logger import get_console_logger

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
        mean (np.ndarray | None): Mean of the input training data
        std (np.ndarray | None): Standard deviation of the input training data
    """
    
    def __init__(self,
                 hidden_sizes: list[int],
                 Lambda: float,
                 epochs: int,
                 learning_rate: float,
                 train_val_split: float,
                 directory: str
                 ):
        """
        Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

        Args:
            hidden_sizes (list[int]): Number of neurons in each hidden layer.
            Lambda (float): L2 regularization parameter.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for gradient descent.
            train_val_split (float): Fraction of dataset used for training.
            directory (str): Directory path to save output files.
        """
        
        super().__init__(hidden_sizes, Lambda, directory)
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.train_val_split: float = train_val_split
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        
        # set for later
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        
        # logger
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.setLevel("INFO")
        self.logger.info(f"Trainer initialised: epochs={epochs}, lr={learning_rate}, train_val_split={train_val_split}")
    
    def norm_vals(self, X_train: np.ndarray) -> None:
        """
        Calculate and store mean and standard deviation into instance for normalisation.
        
        Args:
            X_train (np.ndarray): Input training data.
        """
        
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0) + 1e-8   # avoid zero div for zero std
        
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
    
    @staticmethod   # doesn't require instance of Trainer, so no self
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
    
    def train(self, X: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
        """
        Train the neural network using gradient descent and track RMSE. Saves RMSE vs epochs plot.

        Args:
            X (np.ndarray): Input data, shape (N, 5).
            y (np.ndarray): Target data, shape (N, 1).

        Returns:
            tuple: Two lists of floats:
                - train_loss_history: RMSE for training set per epoch
                - val_loss_history: RMSE for validation set per epoch
        """
        
        # shuffle data
        shuffler = np.random.permutation(X.shape[0])
        X, y = X[shuffler], y[shuffler]
        self.logger.debug(f"Input data shuffled")
        
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
        
        # initialise history arrays
        train_loss_hist = []
        val_loss_hist = []
        
        # logging checkpoints -> print every 20 times
        epoch_interim = max(1, self.epochs//20)
        
        # iterate over epochs
        for epoch in range(self.epochs):
            self.logger.debug(f"Epoch {epoch+1}/{self.epochs} starting")
            
            # apply forward pass
            y_pred_train = self.forward(X_train_norm)
            
            # apply backprop (forward pass is computed in backprop method)
            dW, db = self.backprop(X_train_norm, y_train, y_pred_train)
            
            # update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
                
            # apply forward pass after updates
            y_pred_train = self.forward(X_train_norm)
            y_pred_val = self.forward(X_val_norm)
            
            # calc and append rmse
            train_rmse = self.calc_rmse(y_train, y_pred_train)
            val_rmse = self.calc_rmse(y_val, y_pred_val)
            
            train_loss_hist.append(train_rmse)
            val_loss_hist.append(val_rmse)
            
            # log
            if (epoch + 1) % epoch_interim == 0 or epoch == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # save loss histories
        self.train_loss_history = train_loss_hist
        self.val_loss_history = val_loss_hist
        
        # save norm vals
        self.save_norm_vals("normalisation_values.npz", self.directory)
        
        # save weights
        self.save_weights("model_weights.npz", self.directory)
        
        # save RMSE vs Epoch plot
        plot_loss(train_loss_hist, val_loss_hist, "rmse_vs_epochs.png", self.directory)
        
        # save y_true vs y_preds plot for final epoch
        plot_predictions(y_train, y_pred_train, "ytrue_vs_ypred.png", self.directory)
        
        self.logger.debug("Training complete")
        return train_loss_hist, val_loss_hist