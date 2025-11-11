# imports
import numpy as np
import os

# local imports
from .neural_network import NeuralNetwork
from .plotter import plot_loss, plot_predictions
from .logger import get_console_logger
logger = get_console_logger(__name__)
logger.setLevel("INFO")

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
        logger.debug(f"Input data shuffled")
        
        # split into train/val
        N = X.shape[0]
        split_index = int(N * self.train_val_split)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        logger.debug(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
        
        train_loss_hist = []
        val_loss_hist = []
        
        # iterate over epochs
        for epoch in range(self.epochs):
            logger.debug(f"Epoch {epoch+1}/{self.epochs} starting")
            
            # apply forward pass
            y_pred_train = self.forward(X_train)
            y_pred_val = self.forward(X_val)
            
            # apply backprop
            dW, db = self.backprop(X_train, y_train)
            
            # update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
            
            # calc and append rmse
            train_rmse = self.calc_rmse(y_train, y_pred_train)
            val_rmse = self.calc_rmse(y_val, y_pred_val)
            train_loss_hist.append(train_rmse)
            val_loss_hist.append(val_rmse)
            
            # log
            if (epoch+1)%50 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # save loss histories
        self.train_loss_history = train_loss_hist
        self.val_loss_history = val_loss_hist
        
        # make output folder
        output_dir = os.path.join("backend", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # save RMSE vs Epoch plot
        loss_filename = os.path.join("backend", "outputs", "rmse_vs_epochs.png")
        plot_loss(train_loss_hist, val_loss_hist, filename=loss_filename)
        logger.info(f"RMSE vs Epoch plot saved to {loss_filename}")
        
        # save y_true vs y_preds plot for final epoch
        preds_filename = os.path.join("backend", "outputs", "ytrue_vs_ypred.png")
        plot_predictions(y_train, y_pred_train, filename=preds_filename)
        logger.info(f"Predicted vs True value plot saved to {preds_filename}")
        
        logger.debug("Training complete")
        return train_loss_hist, val_loss_hist