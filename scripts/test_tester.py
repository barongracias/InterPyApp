import sys
import os
import numpy as np
import pickle

# add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.tester import Tester

def test_tester():
    hidden_sizes = [8, 4]
    Lambda = 0.01
    tester = Tester(hidden_sizes=hidden_sizes, Lambda=Lambda)

    # check required files exist
    weights_path = os.path.join("backend", "outputs", "model_weights.npz")
    norm_path = os.path.join("backend", "outputs", "normalisation_values.npz")

    assert os.path.exists(weights_path), f"Weights not found at {weights_path}. Run test_trainer.py first."
    assert os.path.exists(norm_path), f"Normalisation file not found at {norm_path}. Run test_trainer.py first."

    # dummy input test data
    X_test = np.random.rand(5, 5)  # 5 samples, 5 features

    # run prediction using NumPy array
    y_pred = tester.predict(X_test)
    assert isinstance(y_pred, np.ndarray), "Prediction output is not a NumPy array"
    assert y_pred.shape == (5, 1), f"Expected output shape (5, 1), got {y_pred.shape}"
    assert np.all(np.isfinite(y_pred)), "Predicted values contain NaN or Inf"

    print("Tester successfully generated predictions using NumPy array.")

    # create a temporary pickle file with test data
    pkl_path = os.path.join(os.path.dirname(__file__), "X_test.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(X_test, f)

    try:
        # run prediction using pickle file
        y_pred_pkl = tester.predict(pkl_path)
        assert isinstance(y_pred_pkl, np.ndarray), "Prediction output from pickle is not a NumPy array"
        assert y_pred_pkl.shape == (5, 1), f"Expected output shape (5, 1) from pickle, got {y_pred_pkl.shape}"
        assert np.all(np.isfinite(y_pred_pkl)), "Predicted values from pickle contain NaN or Inf"

        print("Tester successfully generated predictions using .pkl file input.")
    finally:
        # clean up pickle file
        if os.path.exists(pkl_path):
            os.remove(pkl_path)


if __name__ == "__main__":
    test_tester()