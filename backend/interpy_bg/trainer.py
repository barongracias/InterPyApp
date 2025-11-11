# imports
import numpy as np
import os

# local imports
from .neural_network import NeuralNetwork
from .plotter import plot_loss
from .logger import get_console_logger
logger = get_console_logger(__name__)

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
    """
    def __init__(self,
                 hidden_sizes: list[int],
                 Lambda: float,
                 epochs: int,
                 learning_rate: float,
                 train_val_split: float
                 ):
        """
        Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

        Args:
            hidden_sizes (list[int]): Number of neurons in each hidden layer.
            Lambda (float): L2 regularization parameter.
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate for gradient descent.
            train_val_split (float): Fraction of dataset used for training.
        """
        super().__init__(hidden_sizes, Lambda)
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.train_val_split: float = train_val_split
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        logger.info(f"Trainer initialised: epochs={epochs}, lr={learning_rate}, train_val_split={train_val_split}")
    
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
        Train the neural network using gradient descent and track RMSE.

        Args:
            X (np.ndarray): Input data, shape (N, 5).
            y (np.ndarray): Target data, shape (N, 1).

        Returns:
            tuple: Two lists of floats:
                - train_loss_history: RMSE for training set per epoch
                - val_loss_history: RMSE for validation set per epoch
        """
        pass