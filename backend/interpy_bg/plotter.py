# imports
import numpy as np
import matplotlib.pyplot as plt
import os

# local imports
from .logger import get_console_logger
logger = get_console_logger(__name__)

def plot_loss(train_loss: list[float], val_loss: list[float], filename: str):
    """
    Plot training and validation loss vs epochs and save the figure.

    Args:
        train_loss (list[float]): Training loss per epoch.
        val_loss (list[float]): Validation loss per epoch.
        filename (str): Path to save the plot.
    """
    try:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, 'b-', label='Train RMSE')
        plt.plot(epochs, val_loss, 'r-', label='Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Training vs Validation RMSE')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # check output folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # save
        plt.savefig(filename)
        logger.info(f"Loss plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error creating loss plot: {e}")
        
def plot_predictions(y_true: list[float], y_pred: list[float], filename: str) -> None:
    """
    Plot predicted vs true values and save the figure.

    Args:
        y_true (list[float]): True target values.
        y_pred (list[float]): Predicted values from the neural network.
        filename (str): Path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, c='blue', marker='o', alpha=0.7, label='Predictions')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction')
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title('Predicted vs True Values', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # check output folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # save figure
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Prediction plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error creating prediction plot: {e}")