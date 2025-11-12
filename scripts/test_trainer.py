import sys
import os
import numpy as np

# add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from interpy_bg.trainer import Trainer

def test_trainer():
    # create dummy data
    np.random.seed(42)
    X = np.random.rand(50, 5)   # 50 samples, 5 features
    y = np.random.rand(50, 1)   # 50 samples, 1 output

    # initialise Trainer with small network for testing
    trainer = Trainer(hidden_sizes=[8, 4],
                      Lambda=0.01,
                      epochs=1000,
                      learning_rate=0.1,
                      train_val_split=0.8)

    # train the network
    train_loss, val_loss = trainer.train(X, y)

    # basic checks
    assert len(train_loss) == trainer.epochs, "Train loss history length mismatch"
    assert len(val_loss) == trainer.epochs, "Validation loss history length mismatch"
    assert all(l >= 0 for l in train_loss), "Train loss contains negative values"
    assert all(l >= 0 for l in val_loss), "Validation loss contains negative values"

    # check that RMSE generally decreases (final < initial)
    assert train_loss[-1] <= train_loss[0], "Train RMSE did not decrease"
    assert val_loss[-1] <= val_loss[0], "Validation RMSE did not decrease"

    # check plots exist
    loss_plot_path = os.path.join("backend", "outputs", "rmse_vs_epochs.png")
    pred_plot_path = os.path.join("backend", "outputs", "ytrue_vs_ypred.png")
    assert os.path.exists(loss_plot_path), f"Loss plot not found: {loss_plot_path}"
    assert os.path.exists(pred_plot_path), f"Prediction plot not found: {pred_plot_path}"

    print("All Trainer.py tests passed successfully.")

if __name__ == "__main__":
    test_trainer()