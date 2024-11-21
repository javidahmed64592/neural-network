import numpy as np

from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix


def feedforward_through_layer(
    input_vals: Matrix, weights: Matrix, bias: Matrix, activation: ActivationFunction
) -> Matrix:
    """
    Feedforward layer to next layer.

    Parameters:
        input_vals (Matrix): Values to feedforward through layer
        weights (Matrix): Layer weights
        bias (Matrix): Layer bias
        activation (ActivationFunction): Layer activation function

    Returns:
        output_vals (Matrix): Output values
    """
    output_vals = weights @ input_vals
    output_vals = output_vals + bias
    return Matrix.map(output_vals, activation)


def calculate_gradient(layer_vals: Matrix, errors: Matrix, activation: ActivationFunction, lr: float) -> Matrix:
    """
    Calculate gradient for gradient descent.

    Parameters:
        layer_vals (Matrix): Layer values from feedforward
        errors (Matrix): Errors from feedforward
        activation (ActivationFunction): Layer activation function
        lr (float): Learning rate

    Returns:
        gradient (Matrix): Gradient values
    """
    gradient = Matrix.from_array(np.vectorize(activation.derivative)(layer_vals.vals))
    gradient = Matrix.multiply_element_wise(gradient, errors)
    return gradient * lr


def calculate_delta(layer_vals: Matrix, gradients: Matrix) -> Matrix:
    """
    Calculate delta factor to adjust weights and biases.

    Parameters:
        layer_vals (Matrix): Layer values from feedforward
        gradients (Matrix): Values from gradient descent

    Returns:
        delta (Matrix): Delta factors
    """
    return gradients @ Matrix.transpose(layer_vals)


def calculate_error_from_expected(expected_outputs: Matrix, actual_outputs: Matrix) -> Matrix:
    """
    Calculate error between expected and actual outputs.

    Parameters:
        expected_outputs (Matrix): Expected values
        actual_outputs (Matrix): Actual values

    Returns:
        errors (Matrix): Difference between expected and actual outputs
    """
    return expected_outputs - actual_outputs


def calculate_next_errors(weights: Matrix, calculated_errors: Matrix) -> Matrix:
    """
    Calculate next set of errors during backpropagation.

    Parameters:
        weights (Matrix): Layer weights
        calculated_errors (Matrix): Errors from layer in front

    Returns:
        errors (Matrix): Next errors
    """
    return Matrix.transpose(weights) @ calculated_errors
