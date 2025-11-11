import sys
import os
import numpy as np

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.neural_network import NeuralNetwork

def test_neural_network():
    # Set seed for reproducibility
    np.random.seed(42)

    # Create dummy input and target
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.random.rand(10, 1)  # 10 samples, 1 output

    # Initialize network with 2 hidden layers
    nn = NeuralNetwork(hidden_sizes=[8, 4], Lambda=0.01)

    # ---- Forward pass ----
    output = nn.forward(X)
    assert output.shape == (10, 1), f"Forward pass output shape mismatch: {output.shape}"
    
    # ---- Cost computation ----
    cost = nn.cost_function(X, y)
    assert cost >= 0, f"Cost should be non-negative, got {cost}"

    # ---- Backpropagation ----
    dW, db = nn.backprop(X, y)
    # Check gradients shapes match weights/biases
    for i, (dw, dbias) in enumerate(zip(dW, db)):
        assert dw.shape == nn.weights[i].shape, f"dW shape mismatch at layer {i}"
        assert dbias.shape == nn.biases[i].shape, f"db shape mismatch at layer {i}"

    # ---- Test get_params and set_params ----
    params = nn.get_params()
    assert params.ndim == 1, f"get_params should return 1D vector, got {params.ndim}D"
    nn.set_params(params)  # Should not change anything
    assert np.allclose(nn.get_params(), params), "Parameters changed after set_params"

    # ---- Test save/load weights ----
    weights_file = os.path.join("backend", "outputs", "test_weights.npz")
    nn.save_weights("test_weights.npz")
    assert os.path.exists(weights_file), f"Weights file not saved: {weights_file}"
    
    nn.load_weights("test_weights.npz")
    assert np.allclose(nn.get_params(), params), "Parameters changed after load_weights"

    print("All neural_network.py tests passed successfully.")

if __name__ == "__main__":
    test_neural_network()