from __future__ import annotations

from collections.abc import Callable

from neural_network.math import nn_math
from neural_network.math.matrix import Matrix
from neural_network.nn.node import Node


class Layer:
    """
    This class creates a neural network Layer and has weights, biases, learning rate and activation function.
    """

    def __init__(
        self,
        size: int,
        num_inputs: int,
        activation: Callable,
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        prev_layer: Layer | None = None,
    ) -> None:
        """
        Initialise Layer object with number of nodes, inputs, activation function and previous layer if exists.

        Parameters:
            size (int): Size of Layer
            num_inputs (int): Number of inputs into Layer
            activation (Callable): Layer activation function
            weights_range (tuple[float, float]): Range for Layer weights
            bias_range (tuple[float, float]): Range for Layer bias
            prev_layer (Layer): Previous Layer to connect
        """
        self._size = size
        self._num_inputs = num_inputs
        self._activation = activation
        self._weights_range = weights_range
        self._bias_range = bias_range
        self._prev_layer = prev_layer

        self._nodes = [self.random_node for _ in range(size)]

    @property
    def random_node(self) -> Node:
        return Node.random_node(self._num_inputs, self._weights_range, self._bias_range, self._activation)

    @property
    def weights(self) -> Matrix:
        _weights = Matrix.from_array([node._weights for node in self._nodes])
        return _weights

    @weights.setter
    def weights(self, new_weights: Matrix) -> None:
        for index, node in enumerate(self._nodes):
            node._weights = new_weights.data[index]

    @property
    def bias(self) -> Matrix:
        _bias = Matrix.from_array([node._bias for node in self._nodes])
        return _bias

    @bias.setter
    def bias(self, new_bias: Matrix) -> None:
        for index, node in enumerate(self._nodes):
            node._bias = new_bias.data[index]

    def feedforward(self, vals: Matrix) -> Matrix:
        """
        Feedforward values through Layer.

        Parameters:
            vals (Matrix): Values to feedforward through Layer

        Returns:
            output (Matrix): Layer output from inputs
        """
        output = nn_math.feedforward_through_layer(
            input_vals=vals, weights=self.weights, bias=self.bias, activation=self._activation
        )
        self._layer_input = vals
        self._layer_output = output
        return output

    def backpropagate_error(self, errors: Matrix, learning_rate: float) -> None:
        """
        Backpropagate errors during training.

        Parameters:
            errors (Matrix): Errors from next Layer
            learning_rate (float): Learning rate
        """
        gradient = nn_math.calculate_gradient(layer_vals=self._layer_output, errors=errors, lr=learning_rate)
        delta = nn_math.calculate_delta(layer_vals=self._layer_input, gradients=gradient)
        self.weights = Matrix.add(self.weights, delta)
        self.bias = Matrix.add(self.bias, gradient)
