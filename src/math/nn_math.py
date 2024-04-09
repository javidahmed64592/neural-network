from typing import Callable

from src.math.matrix import Matrix


def feedforward_through_layer(incoming_matrix: Matrix, weights: Matrix, bias: Matrix, activation: Callable) -> Matrix:
    vals = Matrix.multiply(weights, incoming_matrix)
    vals = Matrix.add(vals, bias)
    vals = Matrix.map(vals, activation)
    return vals


def calculate_gradient(incoming_matrix: Matrix, errors: Matrix, lr: float) -> Matrix:
    gradient = Matrix.from_matrix_array(incoming_matrix.data * (1 - incoming_matrix.data))
    gradient = Matrix.multiply_element_wise(gradient, errors)
    gradient = Matrix.multiply(gradient, lr)
    return gradient


def calculate_delta(incoming_matrix: Matrix, gradients: Matrix) -> Matrix:
    incoming_T = Matrix.transpose(incoming_matrix)
    delta = Matrix.multiply(gradients, incoming_T)
    return delta


def calculate_error_from_expected(expected_outputs: Matrix, actual_outputs: Matrix) -> Matrix:
    errors = Matrix.subtract(expected_outputs, actual_outputs)
    return errors


def calculate_error_from_errors(weights: Matrix, calculated_errors: Matrix) -> Matrix:
    weights_t = Matrix.transpose(weights)
    errors = Matrix.multiply(weights_t, calculated_errors)
    return errors
