import sys
import os
import numpy as np
import pickle

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.trainer import Trainer


def test_trainer():
    # Create dummy data
    np.random.seed(42)
    X = np.random.rand(50, 5)   # 50 samples, 5 features
    y = np.random.rand(50, 1)   # 50 samples, 1 output

    # Define output directory
    output_dir = os.path.join("backend", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary pickle file for training data (dict with X and y)
    train_pkl_path = os.path.join(output_dir, "Xy_train.pkl")
    with open(train_pkl_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    try:
        # Initialise Trainer with test parameters
        trainer = Trainer(
            directory=output_dir,
            hidden_sizes=[16, 8],
            Lambda=0.01,
            epochs=1000,
            learning_rate=0.01,
            train_val_split=0.8,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8 
        )

        # Train the network using the pickle file
        train_loss, val_loss = trainer.train(train_pkl_path)

        # Basic checks
        assert len(train_loss) == trainer.epochs, "Train loss history length mismatch"
        assert len(val_loss) == trainer.epochs, "Validation loss history length mismatch"
        assert all(l >= 0 for l in train_loss), "Train loss contains negative values"
        assert all(l >= 0 for l in val_loss), "Validation loss contains negative values"

        # Check RMSE generally decreases
        assert train_loss[-1] <= train_loss[0] or np.isclose(train_loss[-1], train_loss[0]), "Train RMSE did not decrease"
        assert val_loss[-1] <= val_loss[0] or np.isclose(val_loss[-1], val_loss[0]), "Validation RMSE did not decrease"

        # Check saved output files
        expected_files = [
            "normalisation_values.npz",
            "model_weights.npz",
            "rmse_vs_epochs.png",
            "ytrue_vs_ypred.png",
            "model_metadata.json",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.exists(fpath), f"Expected output file missing: {fpath}"

        print("\n🎉 All Trainer.py tests passed successfully!")

    finally:
        # clean up temporary pickle file
        if os.path.exists(train_pkl_path):
            os.remove(train_pkl_path)


if __name__ == "__main__":
    test_trainer()
