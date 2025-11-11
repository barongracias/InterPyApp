import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.neural_network import NeuralNetwork
import numpy as np

def test_neural_network():
    # Create dummy input and target
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.random.rand(10, 1)  # 10 samples, 1 output

    # Initialize network with 2 hidden layers
    nn = NeuralNetwork(hidden_sizes=[8, 4], Lambda=0.01)

    # Forward pass
    output = nn.forward(X)

    # Compute cost
    cost = nn.cost_function(X, y)

    # Backprop
    dW, db = nn.backprop(X, y)

    # Test get_params and set_params
    params = nn.get_params()
    nn.set_params(params)

    # Test save/load weights
    nn.save_weights("test_weights.npz")
    nn.load_weights("test_weights.npz")

if __name__ == "__main__":
    test_neural_network()