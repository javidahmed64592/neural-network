from typing import List, cast

import numpy as np

from src.math.activation_functions import ActivationFunctions
from src.math.matrix import Matrix
from src.math.nn_math import (
    calculate_delta,
    calculate_error_from_errors,
    calculate_error_from_expected,
    calculate_gradient,
    feedforward_through_layer,
)


class NeuralNetwork:
    """
    This class can be used to create a NeuralNetwork with specified layer sizes. This neural network can feedforward a
    list of inputs, and be trained through backpropagation of errors.
    """

    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]
    LR = 0.1

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
        Initialise NeuralNetwork object with specified layer sizes.

        Parameters:
            input_nodes (int): Number of inputs
            hidden_nodes (int): Number of hidden nodes
            output_nodes (int): Number of outputs
        """
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes

        self._weights_ih = Matrix.random_matrix(
            rows=self._hidden_nodes, cols=self._input_nodes, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1]
        )
        self._weights_ho = Matrix.random_matrix(
            rows=self._output_nodes, cols=self._hidden_nodes, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1]
        )

        self._bias_h = Matrix.random_column(rows=self._hidden_nodes, low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])
        self._bias_o = Matrix.random_column(rows=self._output_nodes, low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])

    def feedforward(self, inputs: List[float]) -> List[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (List[float]): List of outputs
        """
        input_matrix = Matrix.from_matrix_array(np.array(inputs))
        hidden = feedforward_through_layer(input_matrix, self._weights_ih, self._bias_h, ActivationFunctions.sigmoid)
        output = feedforward_through_layer(hidden, self._weights_ho, self._bias_o, ActivationFunctions.sigmoid)
        output = Matrix.transpose(output)
        return cast(List[float], output.data[0])

    def train(self, inputs: List[float], expected_outputs: List[float]) -> None:
        """
        Train NeuralNetwork using a list of input values and expected output values and backpropagate errors.

        Parameters:
            inputs (List[float]): List of input values
            expected_outputs (List[float]): List of output values
        """
        # Feedforward
        input_matrix = Matrix.from_matrix_array(np.array(inputs))
        hidden = feedforward_through_layer(input_matrix, self._weights_ih, self._bias_h, ActivationFunctions.sigmoid)
        output = feedforward_through_layer(hidden, self._weights_ho, self._bias_o, ActivationFunctions.sigmoid)

        # Calculate errors
        expected_output_matrix = Matrix.from_matrix_array(expected_outputs)
        output_errors = calculate_error_from_expected(expected_output_matrix, output)

        # Calculate gradient
        gradient_ho = calculate_gradient(output, output_errors, self.LR)

        # Adjust weights and bias
        weights_ho_delta = calculate_delta(hidden, gradient_ho)
        self._weights_ho = Matrix.add(self._weights_ho, weights_ho_delta)
        self._bias_o = Matrix.add(self._bias_o, gradient_ho)

        # Calculate errors
        hidden_errors = calculate_error_from_errors(self._weights_ho, output_errors)

        # Calculate gradient
        gradient_ih = calculate_gradient(hidden, hidden_errors, self.LR)

        # Adjust weights and bias
        weights_ih_delta = calculate_delta(input_matrix, gradient_ih)
        self._weights_ih = Matrix.add(self._weights_ih, weights_ih_delta)
        self._bias_h = Matrix.add(self._bias_h, gradient_ih)
