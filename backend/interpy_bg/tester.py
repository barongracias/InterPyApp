# imports
import numpy as np
import os
import pickle
from typing import Union

# local imports
from .neural_network import NeuralNetwork
from .logger import get_console_logger
logger = get_console_logger(__name__)
logger.setLevel("INFO")

class Tester(NeuralNetwork):
    """
    Tester class for trained feedforward neural network.

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
        self.mean = np.ndarray | None = None
        self.std = np.ndarray | None = None
    
    def load_trained_weights(self, weights_path: str, norm_oath: str) -> None:
        pass