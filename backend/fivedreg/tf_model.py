# TensorFlow implementation of the 5D→1D regressor
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Sequential


def build_tf_model(hidden_sizes, Lambda: float):
    """
    Build a feedforward network using Keras Sequential API.

    Args:
        hidden_sizes (list[int]): Sizes of hidden layers.
        Lambda (float): L2 regularisation strength.

    Returns:
        tf.keras.Model
    """
    model = Sequential()
    model.add(layers.Input(shape=(5,)))
    for units in hidden_sizes:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(Lambda),
            )
        )
    model.add(layers.Dense(1, activation="linear"))
    return model
