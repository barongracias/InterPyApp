# imports
import numpy as np
import os

# logger
from .logger import get_console_logger
logger = get_console_logger(__name__)

class NeuralNetwork():
    """
    Feedforward neural network for 5D to 1 interpolation.
    
    Attributes:
        input_size (int): Number of input features (fixed at 5).
        hidden_sizes (list[int]): Number of neurons in each hidden layer.
        output_size (int): Number of output neurons (fixed at 1).
        Lambda (float): L2 regularization parameter.
        layer_sizes (list[int]): Complete list of layer sizes including input, hidden, output.
        weights (list[np.ndarray]): Weight matrices for each layer connection.
        biases (list[np.ndarray]): Bias vectors for each layer (excluding input layer).
    """
    
    def __init__(self, hidden_sizes: list[int], Lambda: float):
        """
        Initialize the neural network with random weights and zero biases.

        Args:
            hidden_sizes (list[int]): List specifying number of neurons in each hidden layer.
            Lambda (float): L2 regularisation parameter.
        """
        
        # fixed input/output size for 5D interpolation
        self.input_size = 5
        self.hidden_sizes = hidden_sizes    # e.g., [16, 32, 16]
        self.output_size = 1
        self.Lambda = Lambda
        logger.info(f"NeuralNetwork initialized: 5 inputs, hidden layers {hidden_sizes}, 1 output")
        
        # initialise layer sizes
        self.layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]    # e.g., [5, 16, 32, 16, 1]
        
        # initialise weights with small random numbers (Gaussian)
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01 for i in range(len(self.layer_sizes)-1)]
        logger.debug(f"Initialised weights: {[w.shape for w in self.weights]}")
        
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(len(self.layer_sizes)-1)]
        logger.debug(f"Initialised biases: {[b.shape for b in self.biases]}")

    # sigmoid activation function
    def activation(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function element-wise to the input.
        Numerically stable version to avoid overflow for large negative values.

        Args:
            z (np.ndarray): Input array of any shape.

        Returns:
            np.ndarray: Sigmoid activation applied element-wise to keep the same shape as z.
        """
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z))
                        )
    
    # forward pass method
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            X (np.ndarray): Input data of shape (N, 5) where N is the number of data points.

        Returns:
            np.ndarray: Network output of shape (N, 1).
        """
        
        logger.debug(f"Forward pass input shape: {X.shape}")
        # initialise lists for pre/post-activation function steps, z and z_hat
        self.z_list = []
        self.z_hat_list = [X]   # z_hat[0] is the input layer
        
        # loop over layers
        for i in range(len(self.weights)):
            z = self.z_hat_list[i] @ self.weights[i] + self.biases[i]
            self.z_list.append(z)

            if i == len(self.weights) - 1:
                # output layer: linear act
                z_hat = z
            else:
                # hidden layer/s: sigmoid act
                z_hat = self.activation(z)
            
            self.z_hat_list.append(z_hat)
            logger.debug(f"Layer {i} | z shape: {z.shape}, z_hat shape: {z_hat.shape}")
        
        logger.debug(f"Forward pass output shape: {self.z_hat_list[-1].shape}")
        return self.z_hat_list[-1]
    
    # sigmoid derivative for backpropagation
    def activation_deriv(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function applied element-wise.
        The derivative is set as sigmoid(z) * (1 - sigmoid(z)).

        Args:
            z (np.ndarray): Input array of any shape (pre-activation values).

        Returns:
            np.ndarray: Element-wise derivative of the sigmoid, same shape as z.
        """
        sigmoid = self.activation(z)
        return sigmoid * (1 - sigmoid)
    
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost (loss) of the neural network for given inputs and targets.
    
        The cost equals mean squared error (MSE) plus L2 regularisation on the weights.
    
        Args:
            X (np.ndarray): Input data of shape (N, 5), where N is the number of data points.
            y (np.ndarray): True target values of shape (N, 1).
    
        Returns:
            float: Scalar cost value.
        """
        # y predicted values from forward pass
        y_hat = self.forward(X)
        logger.debug(f"Forward pass completed in cost_function, predictions shape: {y_hat.shape}")
        
        # mean squared error
        mse = 0.5 * np.mean((y - y_hat)**2)
        
        # L2 regularisation term
        reg_term = 0.5 * self.Lambda * sum(np.sum(W**2) for W in self.weights)
        
        # total cost
        J = mse + reg_term
        logger.debug(f"Cost computed: MSE={mse:.6f}, Reg={reg_term:.6f}, Total={J:.6f}")
        
        return J
    
    def backprop(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Calculate the gradients of the cost function w.r.t. to weights and biases using backpropagation.

        Args:
            X (np.ndarray): Input data of shape (N, 5), where N is the number of data points.
            y (np.ndarray): True target values of shape (N, 1).

        Returns:
            tuple: Two lists:
                - dW: list of np.ndarray, gradients of weights for each layer
                - db: list of np.ndarray, gradients of biases for each layer
        """
        
        # forward pass to update z_hat_list and z_list
        y_hat = self.forward(X)
        N = X.shape[0]
        
        # initialise gradient lists
        dW = [np.zeros_like(W) for W in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # output layer error
        delta = y_hat - y   # shape (N, 1)
        logger.debug(f"Output layer delta shape: {delta.shape}")
        
        # backprop through layers in reverse
        for i in reversed(range(len(self.weights))):
            # grad for weights with reg
            dW[i] = self.z_hat_list[i].T @ (delta/N) + self.Lambda * self.weights[i]
            
            # grad for biases
            db[i] = np.mean(delta, axis=0, keepdims=True)
            logger.debug(f"Layer {i} | dW shape: {dW[i].shape}, db shape: {db[i].shape}")

            # delta for prev layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_deriv(self.z_list[i-1])   # except input
                logger.debug(f"Backpropagated delta for layer {i-1} shape: {delta.shape}")

        return dW, db
    
    def get_params(self) -> np.ndarray:
        """
        Get all weights and biases unrolled into a single 1D vector.

        Returns:
            np.ndarray: Flattened vector containing all weights and biases.
        """
        params = np.concatenate([w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])
        logger.debug(f"Parameters flattened into vector of length {params.size}")
        
        return params

    
    def set_params(self, params: np.ndarray) -> None:
        """
        Set all weights and biases from a flattened parameter vector.

        Args:
            params (np.ndarray): 1D vector containing weights and biases to set.
        """
        start = 0
        for i in range(len(self.weights)):
            # calc no. elements for weight matrix
            w_size = self.weights[i].size
            self.weights[i] = np.reshape(params[start:start + w_size], self.weights[i].shape)
            start += w_size
            
        for i in range(len(self.biases)):
            # calc no. elements for bias matrix
            b_size = self.biases[i].size
            self.biases[i] = np.reshape(params[start:start + b_size], self.biases[i].shape)
            start += b_size
        
        logger.debug(f"Parameters set from vector of length {params.size}")

    def save_weights(self, filename="model_weights.npz") -> None:
        """
        Save the weights and biases to backend/outputs.

        Args:
            filename (str): Name of the file (default: 'model_weights.npz').
        """
        path = os.path.join("backend", "outputs", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, *self.weights, *self.biases)
        logger.info(f"Saved weights and biases to {path}")
    
    def load_weights(self, filename="model_weights.npz") -> None:
        """
        Load the weights and biases from backend/outputs.

        Args:
            filename (str): Name of the file to load.
        """
        path = os.path.join("backend", "outputs", filename)
        data = np.load(path)
        total_layers = len(self.weights)

        # first total_layers arrays are weights, the rest are biases
        for i in range(total_layers):
            self.weights[i] = data[f"arr_{i}"]
        for i in range(total_layers):
            self.biases[i] = data[f"arr_{i + total_layers}"]

        logger.info(f"Loaded weights and biases from {path}")