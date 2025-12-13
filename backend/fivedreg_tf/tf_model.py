"""TensorFlow implementation of the 5D→1D regressor network architecture."""

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers, initializers


def build_tf_model(
    hidden_sizes: Sequence[int],
    Lambda: float,
    input_dim: int = 5,
    activation: str = "relu",
    weight_init: str = "auto",
) -> tf.keras.Model:
    """
    Build the feedforward network used for five-dimensional regression.

    Args:
        hidden_sizes: Sequence of hidden layer widths in the order they should appear.
        Lambda: L2 regularisation strength applied to each Dense layer kernel.
        input_dim: Dimensionality of the input feature vector. Defaults to 5 for the
            5D→1D setup.
        activation: Hidden activation function (relu, leakyrelu, tanh, sigmoid).
        weight_init: Weight initialiser ("auto", "he", "xavier").

    Returns:
        A Keras ``Model`` composed of activated Dense layers followed by a single
        linear output neuron.
    """

    act = activation.lower()
    init = weight_init.lower()

    def _resolve_initializer():
        if init == "he":
            return initializers.he_normal()
        if init == "xavier":
            return initializers.glorot_uniform()
        # auto: He for ReLU/LeakyReLU, Xavier otherwise
        if act in {"relu", "leakyrelu"}:
            return initializers.he_normal()
        return initializers.glorot_uniform()

    model = Sequential(name="fived_regressor")
    model.add(layers.InputLayer(input_shape=(input_dim,), dtype="float32", name="model_input"))
    for units in hidden_sizes:
        model.add(
          layers.Dense(
              units,
              activation="relu" if act == "leakyrelu" else act,
              kernel_regularizer=regularizers.l2(Lambda),
              kernel_initializer=_resolve_initializer(),
          )
        )
        if act == "leakyrelu":
            model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(1, activation="linear"))
    return model
