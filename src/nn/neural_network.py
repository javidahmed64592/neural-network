from typing import List, cast

import numpy as np

from src.math.activation_functions import ActivationFunctions
from src.math.matrix import Matrix
from src.math.nn_math import (
    calculate_delta,
    calculate_error_from_expected,
    calculate_gradient,
    calculate_next_errors,
    feedforward_through_layer,
)
from src.nn.layer import Layer


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

        self._hidden_layer = Layer(
            num_nodes=hidden_nodes, num_inputs=input_nodes, activation=ActivationFunctions.sigmoid
        )
        self._output_layer = Layer(
            num_nodes=output_nodes,
            num_inputs=hidden_nodes,
            activation=ActivationFunctions.sigmoid,
            prev_layer=self._hidden_layer,
        )

    def feedforward(self, inputs: List[float]) -> List[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (List[float]): List of outputs
        """
        input_matrix = Matrix.from_matrix_array(np.array(inputs))
        hidden = self._hidden_layer.feedforward(input_matrix)
        output = self._output_layer.feedforward(hidden)
        output = Matrix.transpose(output)
        return cast(List[float], output.data[0])

    def train(self, inputs: List[float], expected_outputs: List[float]) -> List[float]:
        """
        Train NeuralNetwork using a list of input values and expected output values and backpropagate errors.

        Parameters:
            inputs (List[float]): List of input values
            expected_outputs (List[float]): List of output values

        Returns:
            output_errors (List[float]): List of output errors
        """
        # Feedforward
        input_matrix = Matrix.from_matrix_array(np.array(inputs))
        hidden = self._hidden_layer.feedforward(input_matrix)
        output = self._output_layer.feedforward(hidden)

        # Calculate errors
        expected_output_matrix = Matrix.from_matrix_array(expected_outputs)
        output_errors = calculate_error_from_expected(expected_output_matrix, output)

        # Calculate gradient and adjust weights and bias
        self._output_layer.backpropagate_error(layer_vals=output, input_vals=hidden, errors=output_errors)

        # Calculate errors
        hidden_errors = calculate_next_errors(self._output_layer.weights, output_errors)

        # Calculate gradient and adjust weights and bias
        self._hidden_layer.backpropagate_error(layer_vals=hidden, input_vals=input_matrix, errors=hidden_errors)

        output_errors = Matrix.transpose(output_errors)
        return cast(List[float], output_errors.data[0])
