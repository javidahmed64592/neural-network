from __future__ import annotations

import json

import numpy as np

from neural_network.math.activation_functions import SigmoidActivation
from neural_network.math.matrix import Matrix
from neural_network.math.nn_math import calculate_error_from_expected, calculate_next_errors
from neural_network.nn.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.nn.node import NodeConnection


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
        self._input_layer: InputLayer = None
        self._hidden_layers: list[HiddenLayer] = []
        self._output_layer: OutputLayer = None

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
    def layers_reversed(self) -> list[Layer]:
        return self.layers[::-1]

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

    @property
    def connections(self) -> list[NodeConnection]:
        _connections = [node._node_connections for layer in self.layers[1:] for node in layer._nodes]
        return np.array([nc for node in _connections for nc in node])

    def _create_layers(self) -> None:
        """
        Create neural network layers using list of layer sizes.
        """
        _layer_sizes = [self._num_inputs, *self._hidden_layer_sizes, self._num_outputs]

        self._input_layer = InputLayer(
            size=_layer_sizes[0],
            activation=SigmoidActivation,
        )

        for index in range(1, len(_layer_sizes) - 1):
            self._hidden_layers.append(
                HiddenLayer(
                    size=_layer_sizes[index],
                    activation=SigmoidActivation,
                    weights_range=self._weights_range,
                    bias_range=self._bias_range,
                    prev_layer=self.layers[index - 1],
                )
            )

        self._output_layer = OutputLayer(
            size=_layer_sizes[-1],
            activation=SigmoidActivation,
            weights_range=self._weights_range,
            bias_range=self._bias_range,
            prev_layer=self.layers[-2],
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

    def mutate(self, shift_vals: float, prob_new_node: float, prob_toggle_connection: float) -> None:
        """
        Mutate NeuralNetwork Layers by adjusting weights and biases, and potentially adding new Nodes.

        Parameters:
            shift_vals (float): Factor to adjust Layer weights and biases by
            prob_new_node (float): Probability per Layer for a new Node, range [0, 1]
            prob_toggle_connection (float): Probability per Layer to toggle a random Node, range[0, 1]
        """
        for layer in self.layers[1:]:
            layer.mutate(shift_vals)

        add_node_array = np.random.uniform(low=0, high=1, size=len(self._hidden_layers))
        masked_layers = np.array(self._hidden_layers)[add_node_array < prob_new_node]
        for layer in masked_layers:
            layer._add_node()

        connections = self.connections
        toggle_connections_array = np.random.uniform(low=0, high=1, size=len(connections))
        masked_connections = connections[toggle_connections_array < prob_toggle_connection]
        for connection in masked_connections:
            connection.toggle_active()

    def feedforward(self, inputs: list[float]) -> list[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (list[float]): List of input values

        Returns:
            output (list[float]): List of outputs
        """
        self._input_layer.feedforward(Matrix.from_array(inputs))
        output = Matrix.transpose(self._output_layer.output)
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
        vals = Matrix.from_array(inputs)

        for layer in self.layers:
            vals = layer.feedforward(vals)

        expected_output_matrix = Matrix.from_array(expected_outputs)
        errors = calculate_error_from_expected(expected_output_matrix, vals)
        self._output_layer.backpropagate_error(errors, self._lr)
        output_errors = Matrix.transpose(errors)

        for layer in self._hidden_layers[::-1]:
            errors = calculate_next_errors(layer._next_layer.weights, errors)
            layer.backpropagate_error(errors, self._lr)

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
