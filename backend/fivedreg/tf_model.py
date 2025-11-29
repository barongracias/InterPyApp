# TensorFlow implementation of the 5D→1D regressor
from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers


def build_tf_model(
    hidden_sizes: Sequence[int],
    Lambda: float,
    input_dim: int = 5,
    ) -> tf.keras.Model:
    """
    Build a feedforward network using the Keras Sequential API.

    Args:
        hidden_sizes: Sizes of hidden layers.
        Lambda: L2 regularisation strength.
        input_dim: Number of input features (default: 5).

    Returns:
        A compiled Keras model with L2-regularised Dense layers.
    """
    
    model = Sequential(name="fived_regressor")
    model.add(layers.Input(shape=(input_dim,), dtype="float32"))
    for units in hidden_sizes:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(Lambda),
                kernel_initializer="he_normal",
            )
        )
    model.add(layers.Dense(1, activation="linear"))
    return model