"""Mathematical operations for neural network layers."""

import numpy as np

from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix


def feedforward_through_layer(
    input_vals: Matrix, weights: Matrix, bias: Matrix, activation: type[ActivationFunction]
) -> Matrix:
    """Feedforward values through a layer.

    :param Matrix input_vals:
        Values to feedforward through layer.
    :param Matrix weights:
        Layer weights.
    :param Matrix bias:
        Layer bias.
    :param type[ActivationFunction] activation:
        Layer activation function.
    :return Matrix:
        Output values.
    """
    output_vals = weights @ input_vals
    output_vals = output_vals + bias
    return Matrix.map(output_vals, activation)


def calculate_gradient(layer_vals: Matrix, errors: Matrix, activation: type[ActivationFunction], lr: float) -> Matrix:
    """Calculate gradient for gradient descent.

    :param Matrix layer_vals:
        Layer values from feedforward.
    :param Matrix errors:
        Errors from feedforward.
    :param type[ActivationFunction] activation:
        Layer activation function.
    :param float lr:
        Learning rate.
    :return Matrix:
        Gradient values.
    """
    gradient = Matrix.from_array(np.vectorize(activation.derivative)(layer_vals.vals))
    return gradient * errors * lr


def calculate_delta(layer_vals: Matrix, gradients: Matrix) -> Matrix:
    """Calculate delta factor to adjust weights and biases.

    :param Matrix layer_vals:
        Layer values from feedforward.
    :param Matrix gradients:
        Values from gradient descent.
    :return Matrix:
        Delta factors.
    """
    return gradients @ Matrix.transpose(layer_vals)


def calculate_error_from_expected(expected_outputs: Matrix, actual_outputs: Matrix) -> Matrix:
    """Calculate error between expected and actual outputs.

    :param Matrix expected_outputs:
        Expected values.
    :param Matrix actual_outputs:
        Actual values.
    :return Matrix:
        Difference between expected and actual outputs.
    """
    return expected_outputs - actual_outputs


def calculate_next_errors(weights: Matrix, calculated_errors: Matrix) -> Matrix:
    """Calculate next set of errors during backpropagation.

    :param Matrix weights:
        Layer weights.
    :param Matrix calculated_errors:
        Errors from layer in front.
    :return Matrix:
        Next errors.
    """
    return Matrix.transpose(weights) @ calculated_errors
