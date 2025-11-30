# plots for fivedreg (TensorFlow backend)
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# headless-friendly backend
matplotlib.use("Agg")

from .logger import get_console_logger


def plot_loss(train_loss: list[float],
              val_loss: list[float],
              filename: str = "rmse_vs_epochs.png",
              directory: str | None = None) -> None:
    """
    Plot training and validation RMSE vs epochs and save the figure.

    Args:
        train_loss: Training RMSE per epoch.
        val_loss: Validation RMSE per epoch.
        filename: Name of the file.
        directory: Directory path to save file.
    """
    if directory is None:
        directory = os.getcwd()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(directory, ".mplcache"))
    logger = get_console_logger(__name__, os.path.join(directory, "logs"))
    logger.setLevel("INFO")

    try:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, color="blue", linewidth=2, label="Train RMSE")
        plt.plot(epochs, val_loss, color="red", linewidth=2, linestyle="--", label="Validation RMSE")

        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.title("Training vs Validation RMSE", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Loss plot saved to {path}")
    except Exception as e:
        logger.error(f"Error creating loss plot: {e}")


def plot_predictions(y_true: list[float],
                     y_pred: list[float],
                     filename: str = "ytrue_vs_ypred.png",
                     directory: str | None = None) -> None:
    """
    Plot predicted vs true values for the model and save as a figure.

    Args:
        y_true: True target values.
        y_pred: Predicted values from the neural network.
        filename: Name of the file.
        directory: Directory path to save file.
    """
    if directory is None:
        directory = os.getcwd()
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(directory, ".mplcache"))
    logger = get_console_logger(__name__, os.path.join(directory, "logs"))
    logger.setLevel("INFO")

    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color="navy", s=40, alpha=0.7, label="Predictions")

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title("Predicted vs True Values", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Prediction plot saved to {path}")
    except Exception as e:
        logger.error(f"Error creating prediction plot: {e}")
