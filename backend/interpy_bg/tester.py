"""NumPy-based tester for running inference with saved interpy_bg models."""

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
    Inference helper for trained feedforward networks.

    Loads saved weights and normalisation statistics produced during training, applies
    the same scaling to incoming data, and returns predictions for provided samples.
    """
    
    def __init__(self, hidden_sizes: list[int], Lambda: float, directory: str, activation: str = "sigmoid", weight_init: str = "auto", seed: int | None = None):
        """
        Construct a tester aligned with a previously trained model.

        Args:
            hidden_sizes (list[int]): Number of neurons in each hidden layer (should match training).
            Lambda (float): L2 regularisation parameter (should match training).
            directory (str): Directory path containing saved artifacts.
            activation (str): Activation for hidden layers (should match training).
            weight_init (str): Weight init strategy; used only for completeness when constructing.
            seed (int | None): Optional seed; rarely needed for inference.
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

        Returns:
            dict: Parsed metadata contents.

        Raises:
            FileNotFoundError: If the metadata file cannot be found.
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

        Returns:
            None. Populates ``self.mean`` and ``self.std``.

        Raises:
            FileNotFoundError: If the normalisation archive cannot be located.
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

        Raises:
            ValueError: If normalisation statistics are missing or the input feature
                dimension does not match the stored statistics.
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

        Raises:
            ValueError: If a provided path is not ``.pkl`` or the loaded array has an
                unexpected feature dimension.
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
        Perform inference using the saved model and stored normalisation statistics.

        Args:
            X_data (np.ndarray | str): Input data (N, 5) or path to .pkl file.

        Returns:
            np.ndarray: Predicted outputs of shape (N, 1).

        Raises:
            FileNotFoundError: If weights or normalisation files are missing in
                ``self.directory``.
            ValueError: If the supplied input data is invalid.
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
