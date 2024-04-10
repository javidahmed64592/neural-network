from typing import Callable

from src.math.matrix import Matrix


def feedforward_through_layer(input_vals: Matrix, weights: Matrix, bias: Matrix, activation: Callable) -> Matrix:
    """
    Feedforward layer to next layer.

    Parameters:
        input_vals (Matrix): Values to feedforward through layer
        weights (Matrix): Layer weights
        bias (Matrix): Layer bias
        activation (Callable): Layer activation function

    Returns:
        output_vals (Matrix): Output values
    """
    output_vals = Matrix.multiply(weights, input_vals)
    output_vals = Matrix.add(output_vals, bias)
    output_vals = Matrix.map(output_vals, activation)
    return output_vals


def calculate_gradient(layer_vals: Matrix, errors: Matrix, lr: float) -> Matrix:
    """
    Calculate gradient for gradient descent.

    Parameters:
        layer_vals (Matrix): Layer values from feedforward
        errors (Matrix): Errors from feedforward
        lr (float): Learning rate

    Returns:
        gradient (Matrix): Gradient values
    """
    gradient = Matrix.from_array(layer_vals.data * (1 - layer_vals.data))
    gradient = Matrix.multiply_element_wise(gradient, errors)
    gradient = Matrix.multiply(gradient, lr)
    return gradient


def calculate_delta(layer_vals: Matrix, gradients: Matrix) -> Matrix:
    """
    Calculate delta factor to adjust weights and biases.

    Parameters:
        layer_vals (Matrix): Layer values from feedforward
        gradients (Matrix): Values from gradient descent

    Returns:
        delta (Matrix): Delta factors
    """
    incoming_T = Matrix.transpose(layer_vals)
    delta = Matrix.multiply(gradients, incoming_T)
    return delta


def calculate_error_from_expected(expected_outputs: Matrix, actual_outputs: Matrix) -> Matrix:
    """
    Calculate error between expected and actual outputs.

    Parameters:
        expected_outputs (Matrix): Expected values
        actual_outputs (Matrix): Actual values

    Returns:
        errors (Matrix): Difference between expected and actual outputs
    """
    errors = Matrix.subtract(expected_outputs, actual_outputs)
    return errors


def calculate_next_errors(weights: Matrix, calculated_errors: Matrix) -> Matrix:
    """
    Calculate next set of errors during backpropagation.

    Parameters:
        weights (Matrix): Layer weights
        calculated_errors (Matrix): Errors from layer in front

    Returns:
        errors (Matrix): Next errors
    """
    weights_t = Matrix.transpose(weights)
    errors = Matrix.multiply(weights_t, calculated_errors)
    return errors
