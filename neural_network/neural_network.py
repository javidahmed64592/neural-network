import json
from typing import ClassVar

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

    WEIGHTS_RANGE: ClassVar = [-1, 1]
    BIAS_RANGE: ClassVar = [-1, 1]
    LR = 0.1

    def __init__(self, num_inputs: int, num_outputs: int, hidden_layer_sizes: list[int]) -> None:
        """
        Initialise NeuralNetwork object with specified layer sizes.

        Parameters:
            num_inputs (int): Number of inputs
            num_outputs (int): Number of outputs
            hidden_layer_sizes (list[int]): List of hidden layer sizes
        """
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_sizes = hidden_layer_sizes
        self._create_layers()

    def _create_layers(self) -> None:
        """
        Create neural network layers using list of layer sizes.
        """
        _layer = None
        self._hidden_layers: list[Layer] = []

        for index in range(1, len(self.layer_sizes) - 1):
            _layer = Layer(
                size=self.layer_sizes[index],
                num_inputs=self.layer_sizes[index - 1],
                activation=ActivationFunctions.sigmoid,
                prev_layer=_layer,
            )
            self._hidden_layers.append(_layer)

        self._output_layer = Layer(
            size=self.layer_sizes[-1],
            num_inputs=self.layer_sizes[-2],
            activation=ActivationFunctions.sigmoid,
            prev_layer=_layer,
        )

    @property
    def layer_sizes(self) -> list[int]:
        return [self._num_inputs, *self._hidden_layer_sizes, self._num_outputs]

    @property
    def layers(self) -> list[Layer]:
        return [*self._hidden_layers, self._output_layer]

    @property
    def weights(self) -> list[Matrix]:
        _weights = []
        for layer in self.layers:
            _weights.append(layer.weights)
        return _weights

    @weights.setter
    def weights(self, new_weights: list[Matrix]) -> None:
        for layer, weights in zip(self.layers, new_weights, strict=False):
            layer.weights = weights

    @property
    def bias(self) -> list[Matrix]:
        _bias = []
        for layer in self.layers:
            _bias.append(layer.bias)
        return _bias

    @bias.setter
    def bias(self, new_bias: list[Matrix]) -> None:
        for layer, bias in zip(self.layers, new_bias, strict=False):
            layer.bias = bias

    def feedforward(self, inputs: NDArray | list[float]) -> list[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (NDArray | list[float]): List of input values

        Returns:
            output (list[float]): List of outputs
        """
        input_matrix = Matrix.from_array(np.array(inputs))

        hidden = input_matrix
        for layer in self._hidden_layers:
            hidden = layer.feedforward(hidden)

        output = self._output_layer.feedforward(hidden)
        output = Matrix.transpose(output)
        return output.as_list

    def train(self, inputs: list[float], expected_outputs: list[float]) -> list[float]:
        """
        Train NeuralNetwork using a list of input values and expected output values and backpropagate errors.

        Parameters:
            inputs (list[float]): List of input values
            expected_outputs (list[float]): List of output values

        Returns:
            output_errors (list[float]): List of output errors
        """
        input_matrix = Matrix.from_array(np.array(inputs))

        hidden = input_matrix
        for layer in self._hidden_layers:
            hidden = layer.feedforward(hidden)

        output = self._output_layer.feedforward(hidden)

        expected_output_matrix = Matrix.from_array(expected_outputs)
        output_errors = calculate_error_from_expected(expected_output_matrix, output)
        self._output_layer.backpropagate_error(layer_vals=output, input_vals=hidden, errors=output_errors)

        prev_layer = self._output_layer
        hidden_errors = output_errors

        for layer in self._hidden_layers:
            hidden_errors = calculate_next_errors(prev_layer.weights, hidden_errors)
            layer.backpropagate_error(layer_vals=hidden, input_vals=input_matrix, errors=hidden_errors)
            prev_layer = layer

        output_errors = Matrix.transpose(output_errors)
        return output_errors.as_list

    def save(self, filepath: str) -> None:
        """
        Save neural network layer weights and biases to JSON file.

        Parameters:
            filepath (str): Path to file with weights and biases
        """
        _data = {
            "weights": [weights.data.tolist() for weights in self.weights],
            "bias": [bias.data.tolist() for bias in self.bias],
        }
        with open(filepath, "w") as file:
            json.dump(_data, file)

    def load(self, filepath: str) -> None:
        """
        Load neural network layer weights and biases from JSON file.

        Parameters:
            filepath (str): Path to file with weights and biases
        """
        with open(filepath) as file:
            _data = json.load(file)

        self.weights = [Matrix.from_array(weights) for weights in _data["weights"]]
        self.bias = [Matrix.from_array(bias) for bias in _data["bias"]]
