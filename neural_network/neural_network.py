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

    def __init__(self, num_inputs: int, num_outputs: int, hidden_layer_sizes: List[int]) -> None:
        """
        Initialise NeuralNetwork object with specified layer sizes.

        Parameters:
            num_inputs (int): Number of inputs
            num_outputs (int): Number of outputs
            hidden_layer_sizes (List[int]): List of hidden layer sizes
        """
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_sizes = hidden_layer_sizes
        self._create_layers()

    def _create_layers(self) -> None:
        """
        Create neural network layers using list of layer sizes.
        """
        _prev_layer = None
        self._hidden_layers: List[Layer] = []

        for index in range(1, len(self.layer_sizes) - 1):
            _layer = Layer(
                size=self.layer_sizes[index],
                num_inputs=self.layer_sizes[index - 1],
                activation=ActivationFunctions.sigmoid,
                prev_layer=_prev_layer,
            )
            self._hidden_layers.append(_layer)
            _prev_layer = _layer

        self._output_layer = Layer(
            size=self.layer_sizes[-1],
            num_inputs=self.layer_sizes[-2],
            activation=ActivationFunctions.sigmoid,
            prev_layer=_prev_layer,
        )

    @property
    def layer_sizes(self) -> List[int]:
        return [self._num_inputs] + self._hidden_layer_sizes + [self._num_outputs]

    def feedforward(self, inputs: NDArray | List[float]) -> List[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (NDArray | List[float]): List of input values

        Returns:
            output (List[float]): List of outputs
        """
        input_matrix = Matrix.from_array(np.array(inputs))
        prev = input_matrix
        for layer in self._hidden_layers:
            hidden = layer.feedforward(prev)
            prev = hidden
        output = self._output_layer.feedforward(prev)
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

        prev = input_matrix

        for layer in self._hidden_layers:
            hidden = layer.feedforward(prev)
            prev = hidden

        output = self._output_layer.feedforward(prev)

        # Calculate errors
        expected_output_matrix = Matrix.from_array(expected_outputs)
        output_errors = calculate_error_from_expected(expected_output_matrix, output)

        # Calculate gradient and adjust weights and bias
        self._output_layer.backpropagate_error(layer_vals=output, input_vals=hidden, errors=output_errors)

        # Calculate errors
        prev_layer = self._output_layer
        prev_errors = output_errors

        for layer in self._hidden_layers:
            hidden_errors = calculate_next_errors(prev_layer.weights, prev_errors)
            layer.backpropagate_error(layer_vals=hidden, input_vals=input_matrix, errors=hidden_errors)
            prev_layer = layer
            prev_errors = hidden_errors

        output_errors = Matrix.transpose(output_errors)
        return output_errors.to_array()
