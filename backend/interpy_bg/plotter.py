# imports
import numpy as np
import matplotlib.pyplot as plt
import os

# local imports
from .logger import get_console_logger
logger = get_console_logger(__name__)

def plot_loss(train_loss: list[float], val_loss: list[float], filename: str) -> None:
    """
    Plot training and validation RMSE vs epochs and save the figure in high-quality format.

    Args:
        train_loss (list[float]): Training RMSE per epoch.
        val_loss (list[float]): Validation RMSE per epoch.
        filename (str): Path to save the plot.
    """
    try:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, color='blue', linewidth=2, label='Train RMSE')
        plt.plot(epochs, val_loss, color='red', linewidth=2, linestyle='--', label='Validation RMSE')
        
        # plot meta
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Training vs Validation RMSE', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()
        
        # check folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # save figure at high DPI
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Loss plot saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error creating loss plot: {e}")


def plot_predictions(y_true: list[float], y_pred: list[float], filename: str) -> None:
    """
    Plot predicted vs true values for the model and save as a figure.

    Args:
        y_true (list[float]): True target values.
        y_pred (list[float]): Predicted values from the neural network.
        filename (str): Path to save the plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color='navy', s=40, alpha=0.7, label='Predictions')
        
        # perfect prediction line
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # plot meta
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title('Predicted vs True Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()
        
        # check folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # save figure
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Prediction plot saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error creating prediction plot: {e}")