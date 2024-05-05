from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

from neural_network.math.activation_functions import ActivationFunctions
from neural_network.math.matrix import Matrix
from neural_network.math.nn_math import calculate_error_from_expected, calculate_next_errors
from neural_network.nn.layer import HiddenLayer, InputLayer, Layer, OutputLayer


class NeuralNetwork:
    """
    This class can be used to create a NeuralNetwork with specified layer sizes. This neural network can feedforward a
    list of inputs, and be trained through backpropagation of errors.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float] | None = None,
        bias_range: tuple[float, float] | None = None,
        lr: float | None = None,
    ) -> None:
        """
        Initialise NeuralNetwork object with specified layer sizes.

        Parameters:
            num_inputs (int): Number of inputs
            num_outputs (int): Number of outputs
            hidden_layer_sizes (list[int]): List of hidden layer sizes
            weights_range (tuple[float, float]): Range for random weights, defaults to [-1, 1]
            bias_range (tuple[float, float]): Range for random biases, defaults to [-1, 1]
            lr (float): Learning rate for training, defaults to 0.1
        """
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_sizes = hidden_layer_sizes
        self._weights_range = weights_range or [-1.0, 1.0]
        self._bias_range = bias_range or [-1.0, 1.0]
        self._lr = lr or 0.1
        self._create_layers()

    @classmethod
    def from_file(cls, filepath: str) -> NeuralNetwork:
        """
        Load neural network layer with weights and biases from JSON file.

        Parameters:
            filepath (str): Path to file with neural network data
        """
        with open(filepath) as file:
            _data = json.load(file)
        nn = cls(_data["num_inputs"], _data["num_outputs"], _data["hidden_layer_sizes"])
        nn.weights = [Matrix.from_array(weights) for weights in _data["weights"]]
        nn.bias = [Matrix.from_array(bias) for bias in _data["bias"]]
        return nn

    @property
    def layers(self) -> list[Layer]:
        return [self._input_layer, *self._hidden_layers, self._output_layer]

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

    def _create_layers(self) -> None:
        """
        Create neural network layers using list of layer sizes.
        """
        _layer_sizes = [self._num_inputs, *self._hidden_layer_sizes, self._num_outputs]

        self._input_layer = InputLayer(
            size=_layer_sizes[0],
            activation=ActivationFunctions.sigmoid,
        )

        _layer = self._input_layer
        self._hidden_layers: list[Layer] = []

        for index in range(1, len(_layer_sizes) - 1):
            _layer = HiddenLayer(
                size=_layer_sizes[index],
                num_inputs=_layer_sizes[index - 1],
                activation=ActivationFunctions.sigmoid,
                weights_range=self._weights_range,
                bias_range=self._bias_range,
                prev_layer=_layer,
            )
            self._hidden_layers.append(_layer)

        self._output_layer = OutputLayer(
            size=_layer_sizes[-1],
            activation=ActivationFunctions.sigmoid,
            weights_range=self._weights_range,
            bias_range=self._bias_range,
            prev_layer=self._hidden_layers[-1],
        )

    def save(self, filepath: str) -> None:
        """
        Save neural network layer weights and biases to JSON file.

        Parameters:
            filepath (str): Path to file with weights and biases
        """
        _data = {
            "num_inputs": self._num_inputs,
            "num_outputs": self._num_outputs,
            "hidden_layer_sizes": self._hidden_layer_sizes,
            "weights": [weights.vals.tolist() for weights in self.weights],
            "bias": [bias.vals.tolist() for bias in self.bias],
        }
        with open(filepath, "w") as file:
            json.dump(_data, file)

    def mutate(self, shift_vals: float, prob_new_node: float, prob_remove_node: float) -> None:
        """
        Mutate NeuralNetwork Layers by adjusting weights and biases, and potentially adding new Nodes.

        Parameters:
            shift_vals (float): Factor to adjust Layer weights and biases by
            prob_new_node (float): Probability for a new Node, range [0, 1]
            prob_remove_node(float): Probability to remove a Node, range[0, 1]
        """
        for layer in self._hidden_layers:
            layer.mutate(shift_vals, prob_new_node, prob_remove_node)
        self._output_layer.mutate(shift_vals)

    def feedforward(self, inputs: NDArray | list[float]) -> list[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (NDArray | list[float]): List of input values

        Returns:
            output (list[float]): List of outputs
        """
        input_matrix = Matrix.from_array(np.array(inputs))

        vals = self._input_layer.feedforward(input_matrix)
        for layer in self._hidden_layers:
            vals = layer.feedforward(vals)

        output = self._output_layer.feedforward(vals)
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
        layer_input_matrix = Matrix.from_array(np.array(inputs))
        expected_output_matrix = Matrix.from_array(expected_outputs)

        vals = self._input_layer.feedforward(layer_input_matrix)

        for layer in self._hidden_layers:
            vals = layer.feedforward(vals)

        output = self._output_layer.feedforward(vals)

        errors = calculate_error_from_expected(expected_output_matrix, output)
        self._output_layer.backpropagate_error(errors, self._lr)
        output_errors = Matrix.transpose(errors)

        prev_layer = self._output_layer

        for layer in self._hidden_layers[::-1]:
            errors = calculate_next_errors(prev_layer.weights, errors)
            layer.backpropagate_error(errors, self._lr)
            prev_layer = layer

        return output_errors.as_list

    def crossover(
        self, nn: NeuralNetwork, other_nn: NeuralNetwork, mutation_rate: float
    ) -> tuple[list[Matrix], list[Matrix]]:
        """
        Crossover two Neural Networks by mixing their weights and biases, matching the topology of the instance of this
        class.

        Parameters:
            nn (NeuralNetwork): Neural Network to use for average weights and biases
            other_nn (NeuralNetwork): Other Neural Network to use for average weights and biases
            mutation_rate (float): Percentage of weights and biases to be randomised

        Returns:
            new_weights, new_biases (tuple[list[Matrix], list[Matrix]]): New Layer weights and biases
        """
        new_weights = []
        new_biases = []

        for index in range(len(self.layers)):
            new_weight = Matrix.mix_matrices(nn.weights[index], other_nn.weights[index], self.weights[index])
            new_weight = Matrix.mutated_matrix(new_weight, mutation_rate, self._weights_range)
            new_bias = Matrix.mix_matrices(nn.bias[index], other_nn.bias[index], self.bias[index])
            new_bias = Matrix.mutated_matrix(new_bias, mutation_rate, self._bias_range)

            new_weights.append(new_weight)
            new_biases.append(new_bias)

        return [new_weights, new_biases]
