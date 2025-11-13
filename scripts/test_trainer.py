import sys
import os
import numpy as np

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

    # Initialise Trainer with test parameters
    trainer = Trainer(hidden_sizes=[8, 4],
                      Lambda=0.01,
                      epochs=200,  # fewer epochs for faster test
                      learning_rate=0.1,
                      train_val_split=0.8,
                      directory=output_dir)

    # Train the network
    train_loss, val_loss = trainer.train(X, y)

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
    ]
    for fname in expected_files:
        fpath = os.path.join(output_dir, fname)
        assert os.path.exists(fpath), f"Expected output file missing: {fpath}"

    print("\n🎉 All Trainer.py tests passed successfully!")


if __name__ == "__main__":
    test_trainer()