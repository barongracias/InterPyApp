import sys
import os
import numpy as np
import pickle

# add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester

def test_pipeline():
    """
    End-to-end test for the full pipeline:
    - Train a small network using Trainer (from a .pkl file)
    - Save weights, normalisation values, and plots
    - Load model using Tester
    - Predict on test data (NumPy array and .pkl)
    """

    np.random.seed(42)

    # synthetic dataset
    X_train = np.random.rand(50, 5)   # 50 samples, 5 features
    y_train = np.random.rand(50, 1)   # 50 samples, 1 target

    # output directory
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # create temporary pickle file for training data (dict with X and y)
    train_pkl_path = os.path.join(output_dir, "Xy_train.pkl")
    with open(train_pkl_path, "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)

    try:
        # train the network
        trainer = Trainer(
            directory=output_dir,
            hidden_sizes=[16, 8],
            Lambda=0.01,
            epochs=200,
            learning_rate=0.01,
            train_val_split=0.8,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            activation="relu",
            batch_size=16,
            seed=321,
        )

        train_loss, val_loss = trainer.train(train_pkl_path)

        # check outputs exist
        weights_path = os.path.join(output_dir, "model_weights.npz")
        norm_path = os.path.join(output_dir, "normalisation_values.npz")
        loss_plot_path = os.path.join(output_dir, "rmse_vs_epochs.png")
        pred_plot_path = os.path.join(output_dir, "ytrue_vs_ypred.png")
        metadata_path = os.path.join(output_dir, "model_metadata.json")

        assert os.path.exists(weights_path), "Trained weights file missing"
        assert os.path.exists(norm_path), "Normalisation values file missing"
        assert os.path.exists(loss_plot_path), "RMSE vs Epoch plot missing"
        assert os.path.exists(pred_plot_path), "Prediction plot missing"
        assert os.path.exists(metadata_path), "Model metadata file missing"

        # check that losses decrease
        assert train_loss[-1] <= train_loss[0], "Train RMSE did not decrease"
        assert val_loss[-1] <= val_loss[0], "Validation RMSE did not decrease"

        # create a tester instance
        tester = Tester(hidden_sizes=[16, 8], Lambda=0.01, directory=output_dir)

        # test with NumPy array
        X_test = np.random.rand(5, 5)
        y_pred = tester.predict(X_test)
        assert y_pred.shape == (5, 1), "Prediction shape mismatch for NumPy array"
        assert np.all(np.isfinite(y_pred)), "Predictions contain NaN or Inf"

        # test with pickle file
        pkl_path = os.path.join(output_dir, "X_test.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(X_test, f)

        y_pred_pkl = tester.predict(pkl_path)
        assert y_pred_pkl.shape == (5, 1), "Prediction shape mismatch for pickle input"
        assert np.all(np.isfinite(y_pred_pkl)), "Predictions from pickle contain NaN or Inf"

    finally:
        # clean up temporary pickle files
        if os.path.exists(train_pkl_path):
            os.remove(train_pkl_path)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

    print("🎉 Full pipeline test passed successfully.")


if __name__ == "__main__":
    test_pipeline()
