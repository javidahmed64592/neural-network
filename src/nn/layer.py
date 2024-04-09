from __future__ import annotations

from typing import Callable, Optional

from src.math import nn_math
from src.math.matrix import Matrix


class Layer:
    """
    This class creates a neural network Layer and has weights, biases, learning rate and activation function.
    """

    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]
    LR = 0.1

    def __init__(
        self, num_nodes: int, num_inputs: int, activation: Callable, prev_layer: Optional[Layer] = None
    ) -> None:
        """
        Initialise Layer object with number of nodes, inputs, activation function and previous layer if exists.

        Parameters:
            num_nodes (int): Size of Layer
            num_inputs (int): Number of inputs into Layer
            activation (Callable): Layer activation function
            prev_layer (Layer): Previous Layer to connect
        """
        self._size = num_nodes
        self._num_inputs = num_inputs
        self._activation = activation
        self._prev_layer = prev_layer
        self._weights = Matrix.random_matrix(
            rows=self._size, cols=self._num_inputs, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1]
        )
        self._bias = Matrix.random_column(rows=self._size, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1])

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

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
        self._weights = Matrix.add(self._weights, delta)
        self._bias = Matrix.add(self._bias, gradient)
