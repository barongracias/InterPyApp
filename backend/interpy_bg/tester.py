# imports
import numpy as np
import os
import pickle
import json

# local imports
from .neural_network import NeuralNetwork
from .logger import get_console_logger
from .utils import timer, log_call

class Tester(NeuralNetwork):
    __test__ = False  # prevent pytest from collecting this as a test class
    """
    Tester class for trained feedforward neural network. Loads model weights and normalisation values, applies normalisation,
    and calculates predicted outputs for given test inputs.

    Inherits from NeuralNetwork.

    Attributes:
        directory (str): Directory path to save output files.
        mean (np.ndarray | None): Mean of the input training data
        std (np.ndarray | None): Standard deviation of the input training data
    """
    
    def __init__(self, hidden_sizes: list[int], Lambda: float, directory: str, activation: str = "sigmoid", weight_init: str = "auto", seed: int | None = None):
        """
        Initialise Tester with hyperparameters and call NeuralNetwork constructor.

        Args:
            hidden_sizes (list[int]): Number of neurons in each hidden layer.
            Lambda (float): L2 regularization parameter.
            directory (str): Directory path to save output files.
            activation (str): Activation for hidden layers (should match training).
            weight_init (str): Weight init strategy; used only for completeness when constructing.
            seed (int | None): Optional seed (not required for inference).
        """
        
        super().__init__(hidden_sizes, Lambda, directory, activation=activation, weight_init=weight_init, seed=seed)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        
        # logger
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info("Tester initialised")

    @staticmethod
    def load_metadata(filename: str = "model_metadata.json", directory: str = None) -> dict:
        """
        Load stored model metadata (architecture and regularisation).

        Args:
            filename (str): Metadata filename.
            directory (str): Directory path to load from.
        """
        
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    @log_call
    def load_norm_vals(self, filename: str = "normalisation_values.npz", directory: str = None) -> None:
        """
        Load stored normalisation values (mean and standard deviation)
        
        Args:
            filename (str): Name of the file.
            directory (str): Directory path to save file.
        """
        
        # verify path
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalisation file not found: {path}")
        
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        self.logger.debug(f"Loaded normalisation values from {path}")
    
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
            raise ValueError("Normalisation values not loaded")
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Input has {X.shape[1]} features but expected {self.mean.shape[0]}")
        
        return (X - self.mean) / self.std
    
    @staticmethod
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
        # if array passed
        else:
            X_test = np.asarray(X_data, dtype=float)
        
        # check 2D shape (N, 5)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.shape[1] != 5:
            raise ValueError(f"Expected input shape of (N, 5), got {X_test.shape}")
        
        return X_test
    
    @timer
    @log_call
    def predict(self, X_data: np.ndarray | str) -> np.ndarray:
        """
        Perform interpolation using the trained model.

        Args:
            X_data (np.ndarray | str): Input data (N, 5) or path to .pkl file.

        Returns:
            np.ndarray: Predicted outputs of shape (N, 1).
        """
        
        # load model weights and norm vals (always prefer saved artifacts if present)
        weights_path = os.path.join(self.directory, "model_weights.npz")
        norm_path = os.path.join(self.directory, "normalisation_values.npz")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Trained weights not found at {weights_path}. Run training first.")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalisation values not found at {norm_path}. Run training first.")

        self.load_weights("model_weights.npz", self.directory)
        self.load_norm_vals("normalisation_values.npz", self.directory)
        
        # load and normalise input testing data
        X_test = self.load_test_data(X_data)
        X_test_norm = self.normalise(X_test)
        
        # apply forward pass
        y_pred = self.forward(X_test_norm)
        self.logger.info(f"Generated predictions for {len(X_test)} samples.")
        
        return y_pred
