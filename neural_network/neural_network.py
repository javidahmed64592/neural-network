from typing import List

import numpy as np
from numpy.typing import NDArray

from neural_network.math.activation_functions import ActivationFunctions
from neural_network.math.matrix import Matrix
from neural_network.math.nn_math import calculate_error_from_expected, calculate_next_errors
from neural_network.nn.layer import Layer


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

        self._hidden_layer = Layer(size=hidden_nodes, num_inputs=input_nodes, activation=ActivationFunctions.sigmoid)
        self._output_layer = Layer(
            size=output_nodes,
            num_inputs=hidden_nodes,
            activation=ActivationFunctions.sigmoid,
            prev_layer=self._hidden_layer,
        )

    def feedforward(self, inputs: NDArray | List[float]) -> List[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (NDArray | List[float]): List of input values

        Returns:
            output (List[float]): List of outputs
        """
        input_matrix = Matrix.from_array(np.array(inputs))
        hidden = self._hidden_layer.feedforward(input_matrix)
        output = self._output_layer.feedforward(hidden)
        output = Matrix.transpose(output)
        return output.to_array()

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
        input_matrix = Matrix.from_array(np.array(inputs))
        hidden = self._hidden_layer.feedforward(input_matrix)
        output = self._output_layer.feedforward(hidden)

        # Calculate errors
        expected_output_matrix = Matrix.from_array(expected_outputs)
        output_errors = calculate_error_from_expected(expected_output_matrix, output)

        # Calculate gradient and adjust weights and bias
        self._output_layer.backpropagate_error(layer_vals=output, input_vals=hidden, errors=output_errors)

        # Calculate errors
        hidden_errors = calculate_next_errors(self._output_layer.weights, output_errors)

        # Calculate gradient and adjust weights and bias
        self._hidden_layer.backpropagate_error(layer_vals=hidden, input_vals=input_matrix, errors=hidden_errors)

        output_errors = Matrix.transpose(output_errors)
        return output_errors.to_array()
