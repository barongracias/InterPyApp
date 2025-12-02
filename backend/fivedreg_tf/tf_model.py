"""TensorFlow implementation of the 5D→1D regressor network architecture."""

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers


def build_tf_model(
    hidden_sizes: Sequence[int],
    Lambda: float,
    input_dim: int = 5,
    ) -> tf.keras.Model:
    """
    Build the feedforward network used for five-dimensional regression.

    Args:
        hidden_sizes: Sequence of hidden layer widths in the order they should appear.
        Lambda: L2 regularisation strength applied to each Dense layer kernel.
        input_dim: Dimensionality of the input feature vector. Defaults to 5 for the
            5D→1D setup.

    Returns:
        A Keras ``Model`` composed of ReLU-activated Dense layers followed by a single
        linear output neuron.
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
