from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from neural_network.math import nn_math
from neural_network.math.matrix import Matrix
from neural_network.nn.node import Node


class Layer:
    """
    This class creates a neural network Layer and has weights, biases, learning rate and activation function.
    """

    WEIGHTS_RANGE: ClassVar = [-1.0, 1.0]
    BIAS_RANGE: ClassVar = [-1.0, 1.0]
    LR = 0.1

    def __init__(self, size: int, num_inputs: int, activation: Callable, prev_layer: Layer | None = None) -> None:
        """
        Initialise Layer object with number of nodes, inputs, activation function and previous layer if exists.

        Parameters:
            size (int): Size of Layer
            num_inputs (int): Number of inputs into Layer
            activation (Callable): Layer activation function
            prev_layer (Layer): Previous Layer to connect
        """
        self._size = size
        self._num_inputs = num_inputs
        self._activation = activation
        self._prev_layer = prev_layer

        self._nodes = [self.random_node for _ in range(size)]

    @property
    def random_node(self) -> Node:
        return Node.random_node(self._num_inputs, self.WEIGHTS_RANGE, self.BIAS_RANGE, self._activation)

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
        return output

    def backpropagate_error(self, layer_vals: Matrix, input_vals: Matrix, errors: Matrix) -> None:
        """
        Backpropagate errors during training.

        Parameters:
            layer_vals (Matrix): Values calculated by this Layer
            input_vals (Matrix): Values from previous Layer
            errors (Matrix): Errors from next Layer
        """
        gradient = nn_math.calculate_gradient(layer_vals=layer_vals, errors=errors, lr=self.LR)
        delta = nn_math.calculate_delta(layer_vals=input_vals, gradients=gradient)
        self.weights = Matrix.add(self.weights, delta)
        self.bias = Matrix.add(self.bias, gradient)
