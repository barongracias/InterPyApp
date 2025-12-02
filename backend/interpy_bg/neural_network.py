"""Core NumPy-based feedforward network used by interpy_bg training and testing."""

# imports
import numpy as np
import os

# local imports
from .logger import get_console_logger
from .utils import timer, log_call

class NeuralNetwork():
    """
    Lightweight feedforward neural network for 5D->1 interpolation.

    The network supports configurable hidden layer sizes, activation functions, and
    weight initialisation strategies. It also tracks Adam optimiser state required by
    the training loop and exposes forward/backprop routines used by ``Trainer`` and
    ``Tester``.
    """
    
    def __init__(
        self,
        hidden_sizes: list[int],
        Lambda: float,
        directory: str,
        activation: str = "sigmoid",
        weight_init: str = "auto",
        seed: int | None = None
    ):
        """
        Initialise network topology, parameters, and optimiser state.

        Args:
            hidden_sizes (list[int]): Width of each hidden layer in order.
            Lambda (float): L2 regularisation strength applied during backprop.
            directory (str): Output directory for artifacts (logs, weights).
            activation (str): Hidden-layer activation ('sigmoid', 'tanh', 'relu', 'leakyrelu').
            weight_init (str): Weight initialisation ('auto', 'he', 'xavier'); ``auto`` selects
                He for ReLU variants, otherwise Xavier.
            seed (int | None): Optional RNG seed for reproducible initialisation.

        Raises:
            ValueError: If an unsupported activation or weight initialisation strategy is provided.
        """
        
        self.hidden_sizes: list[int] = hidden_sizes    # e.g., [16, 32, 16]
        self.Lambda: float = Lambda
        self.directory: str = directory
        self.activation_name: str = activation.lower()
        self.weight_init: str = weight_init.lower()
        self.seed: int | None = seed
        if self.activation_name not in {"sigmoid", "tanh", "relu", "leakyrelu"}:
            raise ValueError("activation must be one of: sigmoid, tanh, relu, leakyrelu")
        if self.weight_init not in {"auto", "he", "xavier"}:
            raise ValueError("weight_init must be 'auto', 'he', or 'xavier'")

        # fixed input/output size for 5D->1D interpolation
        self.input_size: int = 5
        self.output_size: int = 1
        
        # logger (Trainer/Tester override this to their own module logger after init)
        self.logger = get_console_logger(__name__, os.path.join(self.directory, "logs"))
        self.logger.info(f"NeuralNetwork initialized: 5 inputs, hidden layers {hidden_sizes}, 1 output")
        
        # initialise layer sizes
        self.layer_sizes: list[int] = [self.input_size] + self.hidden_sizes + [self.output_size]    # e.g., [5, 16, 32, 16, 1]

        # rng for reproducibility
        self.rng = np.random.default_rng(self.seed)
        
        # initialise weights and biases with suitable scaling
        self.weights: list[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            init_type = self._resolve_init(fan_in, fan_out)
            if init_type == "he":
                std = np.sqrt(2.0 / fan_in)
            else:  # xavier
                std = np.sqrt(2.0 / (fan_in + fan_out))
            W = self.rng.normal(0.0, std, size=(fan_in, fan_out))
            self.weights.append(W)
        self.logger.debug(f"Initialised weights: {[w.shape for w in self.weights]}")
        
        self.biases: list[np.ndarray] = [np.zeros((1, self.layer_sizes[i+1])) for i in range(len(self.layer_sizes)-1)]
        self.logger.debug(f"Initialised biases: {[b.shape for b in self.biases]}")
        
        # adam optimiser state (weights)
        self.m: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]   # first moment for weights
        self.v: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]   # second moment for weights

        # adam optimiser state (biases)
        self.mb: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]   # first moment for biases
        self.vb: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]   # second moment for biases

        # timestep for bias correction
        self.t: int = 0
        self.logger.debug("Initialised Adam states for weights and biases.")

    def _resolve_init(self, fan_in: int, fan_out: int) -> str:
        """
        Decide which weight initialisation scheme to use for a layer.

        Args:
            fan_in (int): Number of input units to the layer.
            fan_out (int): Number of output units from the layer.

        Returns:
            str: Either ``"he"`` or ``"xavier"`` depending on configuration.

        Raises:
            ValueError: If ``weight_init`` is not recognised.
        """
        if self.weight_init == "auto":
            return "he" if self.activation_name in {"relu", "leakyrelu"} else "xavier"
        if self.weight_init not in {"he", "xavier"}:
            raise ValueError("weight_init must be 'auto', 'he', or 'xavier'")
        return self.weight_init

    def activation(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the configured activation function element-wise to a pre-activation array.

        Args:
            z (np.ndarray): Pre-activation values for a layer.

        Returns:
            np.ndarray: Activated values with the same shape as ``z``.

        Raises:
            ValueError: If the configured activation is unsupported.
        """
        if self.activation_name == "sigmoid":
            return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        if self.activation_name == "tanh":
            return np.tanh(z)
        if self.activation_name == "relu":
            return np.maximum(0, z)
        if self.activation_name == "leakyrelu":
            return np.where(z > 0, z, 0.01 * z)
        raise ValueError(f"Unsupported activation: {self.activation_name}")
    
    def activation_deriv(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the configured activation element-wise.

        Args:
            z (np.ndarray): Pre-activation values for a layer.

        Returns:
            np.ndarray: Derivative values matching the shape of ``z``.

        Raises:
            ValueError: If the configured activation is unsupported.
        """
        if self.activation_name == "sigmoid":
            sig = self.activation(z)
            return sig * (1 - sig)
        if self.activation_name == "tanh":
            return 1 - np.tanh(z) ** 2
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        if self.activation_name == "leakyrelu":
            return np.where(z > 0, 1.0, 0.01)
        raise ValueError(f"Unsupported activation: {self.activation_name}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network and cache activations.

        Args:
            X (np.ndarray): Input data of shape ``(N, 5)`` where N is the number of rows.

        Returns:
            np.ndarray: Network output of shape ``(N, 1)``.
        """
        
        self.logger.debug(f"Forward pass input shape: {X.shape}")
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
            self.logger.debug(f"Layer {i} | z shape: {z.shape}, z_hat shape: {z_hat.shape}")
        
        self.logger.debug(f"Forward pass output shape: {self.z_hat_list[-1].shape}")
        return self.z_hat_list[-1]
    
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mean squared error with L2 regularisation for given inputs/targets.

        Args:
            X (np.ndarray): Input data of shape ``(N, 5)``.
            y (np.ndarray): Target values of shape ``(N, 1)``.

        Returns:
            float: Scalar cost (MSE + L2 penalty).
        """
        
        # y predicted values from forward pass
        y_hat = self.forward(X)
        self.logger.debug(f"Forward pass completed in cost_function, predictions shape: {y_hat.shape}")
        
        # mean squared error
        mse = 0.5 * np.mean((y - y_hat)**2)
        
        # L2 regularisation term scaled by dataset size to match mean-based loss
        N = X.shape[0]
        reg_term = 0.5 * (self.Lambda / N) * sum(np.sum(W**2) for W in self.weights)
        
        # total cost
        J = mse + reg_term
        self.logger.debug(f"Cost computed: MSE={mse:.6f}, Reg={reg_term:.6f}, Total={J:.6f}")
        
        return J
    
    def backprop(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Calculate gradients of the loss with respect to weights and biases.

        Args:
            X (np.ndarray): Input data of shape ``(N, 5)``.
            y (np.ndarray): True targets of shape ``(N, 1)``.
            y_hat (np.ndarray): Predicted targets of shape ``(N, 1)`` from a forward pass.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: Gradients ``(dW, db)`` aligned with
            the ``weights`` and ``biases`` lists.
        """
        
        # initialise gradient lists
        dW = [np.zeros_like(W) for W in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # output layer error
        delta = y_hat - y   # shape (N, 1)
        self.logger.debug(f"Output layer delta shape: {delta.shape}")
        
        # backprop through layers in reverse
        N = X.shape[0]
        for i in reversed(range(len(self.weights))):
            # grad for weights with reg
            dW[i] = self.z_hat_list[i].T @ (delta/N) + (self.Lambda / N) * self.weights[i]
            
            # grad for biases
            db[i] = np.mean(delta, axis=0, keepdims=True)
            self.logger.debug(f"Layer {i} | dW shape: {dW[i].shape}, db shape: {db[i].shape}")

            # delta for prev layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_deriv(self.z_list[i-1])   # except input
                self.logger.debug(f"Backpropagated delta for layer {i-1} shape: {delta.shape}")

        return dW, db

    @log_call
    def save_weights(self, filename: str = "model_weights.npz", directory: str = None) -> None:
        """
        Save the weights and biases to backend/outputs_numpy.

        Args:
            filename (str): Name of the file.
            directory (str): Directory path to save file.
        """
        
        # verify path
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        
        # save weights
        np.savez(path, *self.weights, *self.biases)
        self.logger.info(f"Saved weights and biases to {path}")
    
    @log_call
    def load_weights(self, filename: str = "model_weights.npz", directory: str = None) -> None:
        """
        Load the weights and biases from backend/outputs_numpy.

        Args:
            filename (str): Name of the file to load.
            directory (str): Directory path to save file.
        """
        
        # verify path
        if directory is None:
            directory = os.getcwd()
        path = os.path.join(directory, filename)
        data = np.load(path)

        total_layers = len(self.weights)
        # first total_layers arrays are weights, the rest are biases
        for i in range(total_layers):
            self.weights[i] = data[f"arr_{i}"]
        for i in range(total_layers):
            self.biases[i] = data[f"arr_{i + total_layers}"]

        self.logger.info(f"Loaded weights and biases from {path}")
