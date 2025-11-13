import sys
import os
import numpy as np

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.neural_network import NeuralNetwork

def test_neural_network():
    # Reproducibility
    np.random.seed(42)

    # Dummy input and target
    X = np.random.rand(10, 5)  # 10 samples, 5 features
    y = np.random.rand(10, 1)  # 10 samples, 1 output
    
    # output directory
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize network
    nn = NeuralNetwork([8, 4], 0.01, output_dir)
    print("NeuralNetwork initialized successfully")

    # Forward pass
    output = nn.forward(X)
    assert output.shape == (10, 1), f"Forward output shape mismatch: {output.shape}"
    print("Forward pass output shape correct")

    # Cost computation
    cost = nn.cost_function(X, y)
    assert cost >= 0, f"Cost should be non-negative, got {cost}"
    print(f"Cost computed successfully: {cost:.6f}")

    # Backpropagation
    dW, db = nn.backprop(X, y, nn.forward(X))
    for i, (dw, dbias) in enumerate(zip(dW, db)):
        assert dw.shape == nn.weights[i].shape, f"dW shape mismatch at layer {i}"
        assert dbias.shape == nn.biases[i].shape, f"db shape mismatch at layer {i}"
    print("Backpropagation gradients shapes correct")

    # Test save/load weights
    weights_file = os.path.join(output_dir, "test_weights.npz")

    nn.save_weights("test_weights.npz", directory=output_dir)
    assert os.path.exists(weights_file), f"Weights file not found: {weights_file}"

    # Modify weights to ensure loading restores them
    original_weights = [w.copy() for w in nn.weights]
    nn.weights = [np.random.randn(*w.shape) for w in nn.weights]
    nn.load_weights("test_weights.npz", directory=output_dir)

    for w_loaded, w_original in zip(nn.weights, original_weights):
        assert np.allclose(w_loaded, w_original), "Loaded weights do not match original values"
    print("Save/load weights test passed")

    print("\n🎉 All neural_network.py tests passed successfully!")


if __name__ == "__main__":
    test_neural_network()