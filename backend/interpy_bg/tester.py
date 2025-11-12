# imports
import numpy as np
import os
import pickle

# local imports
from .neural_network import NeuralNetwork
from .logger import get_console_logger
logger = get_console_logger(__name__)
logger.setLevel("INFO")

class Tester(NeuralNetwork):
    """
    Tester class for trained feedforward neural network. Loads model weights and normalisation values, applies normalisation,
    and calculates predicted outputs for given test inputs.

    Inherits from NeuralNetwork.

    Attributes:
        mean (np.ndarray | None): Mean of the input training data
        std (np.ndarray | None): Standard deviation of the input training data
    """
    
    def __init__(self, hidden_sizes: list[int], Lambda: float):
        """
        Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

        Args:
            hidden_sizes (list[int]): Number of neurons in each hidden layer.
            Lambda (float): L2 regularization parameter.
        """
        
        super().__init__(hidden_sizes, Lambda)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        
        logger.info("Tester initialised")
        
    def load_norm_vals(self, filename: str = "normalisation_values.npz") -> None:
        """
        Load stored normalisation values (mean and standard deviation)
        
        Args:
            filename (str): Name of the file.
        """
        
        path = os.path.join("backend", "outputs", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalisation file not found: {path}")
        
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        logger.debug(f"Loaded normalisation values from {path}")
        
    def normalise(self, X: np.ndarray) -> np.ndarray:
        """
        Normalise input using mean and standard deviation.
        
        Args:
            X (np.ndarray): Input data, shape (N, 5).
        
        Returns:
            np.ndarray: Normalised input data, shape (N, 5).
        """
        
        if self.mean is None or self.std is None:
            raise ValueError("Normalisation values not loaded")
        return (X - self.mean) / self.std
    
    @staticmethod   # doesn't require instance of Tester, so no self
    def load_test_data(X_data: np.ndarray | str) -> np.ndarray:
        """
        Load test data from numpy array or pickle (.pkl) file.

        Args:
            X_data (np.ndarray | str): Input data (N, 5) or path to .pkl file.

        Returns:
            np.ndarray: Input data of shape (N, 5).
        """
        
        # if path string passed
        if isinstance(X_data, str):
            # check pickle file
            if not X_data.endswith(".pkl"):
                raise ValueError("String path must end with .pkl")
            # open pickle file
            with open(X_data, "rb") as f:
                X_test = pickle.load(f)
            logger.info(f"Loaded test data from {X_data}")
        # if array passed
        else:
            X_test = np.array(X_data, dtype=float)
        
        # check 2D shape (N, 5)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.shape[1] != 5:
            raise ValueError(f"Expected input shape of (N, 5), got {X_test.shape}")
        
        return X_test
    
    def predict(self, X_data: np.ndarray | str) -> np.ndarray:
        """
        Perform interpolation using the trained model.

        Args:
            X_data (np.ndarray | str): Input data (N, 5) or path to .pkl file.

        Returns:
            np.ndarray: Predicted outputs of shape (N, 1).
        """
        
        # load model weights and norm vals
        self.load_weights("model_weights.npz")
        self.load_norm_vals("normalisation_values.npz")
        
        # load and normalise input testing data
        X_test = self.load_test_data(X_data)
        X_test_norm = self.normalise(X_test)
        
        # apply forward pass
        y_pred = self.forward(X_test_norm)
        logger.info(f"Generated predictions for {len(X_test)} samples.")
        
        return y_pred