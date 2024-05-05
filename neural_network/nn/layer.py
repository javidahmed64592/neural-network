from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
        prev_layer: Layer | None,
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
        self._prev_layer: Layer = None
        self._next_layer: Layer = None

        self._num_inputs = num_inputs
        self._activation = activation
        self._weights_range = weights_range
        self._bias_range = bias_range

        if prev_layer:
            self._prev_layer = prev_layer
            prev_layer._next_layer = self

        self._nodes = [self.random_node for _ in range(size)]

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def num_inputs(self) -> int:
        if self._prev_layer:
            self._num_inputs = self._prev_layer.size
        return self._num_inputs

    @property
    def random_node(self) -> Node:
        return Node.random_node(self.num_inputs, self._weights_range, self._bias_range, self._activation)

    @property
    def weights(self) -> Matrix:
        _weights = Matrix.from_array([node._weights for node in self._nodes])
        return _weights

    @weights.setter
    def weights(self, new_weights: Matrix) -> None:
        for index, node in enumerate(self._nodes):
            node._weights = new_weights.vals[index]

    @property
    def bias(self) -> Matrix:
        _bias = Matrix.from_array([node._bias for node in self._nodes])
        return _bias

    @bias.setter
    def bias(self, new_bias: Matrix) -> None:
        for index, node in enumerate(self._nodes):
            node._bias = new_bias.vals[index]

    def mutate(self, shift_vals: float) -> None:
        """
        Mutate Layer weights and biases.

        Parameters:
            shift_vals (float): Factor to adjust Layer weights and biases by
        """
        self.weights.shift_vals(shift_vals)
        self.bias.shift_vals(shift_vals)

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


class InputLayer(Layer):
    """
    An input Layer in the NeuralNetwork.
    """

    def __init__(
        self,
        size: int,
        activation: Callable,
    ) -> None:
        """
        Initialise InputLayer object with number of nodes and activation function.

        Parameters:
            size (int): Size of OutputLayer
            activation (Callable): OutputLayer activation function
        """
        super().__init__(size, 1, activation, [1, 1], [0, 0], None)

    def feedforward(self, vals: Matrix) -> Matrix:
        """
        Set InputLayer values.

        Parameters:
            vals (Matrix): Input values

        Returns:
            vals (Matrix): Input values
        """
        self._layer_input = vals
        self._layer_output = vals
        return vals


class HiddenLayer(Layer):
    """
    A hidden Layer in the NeuralNetwork.
    """

    def __init__(
        self,
        size: int,
        num_inputs: int,
        activation: Callable,
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        prev_layer: InputLayer | Layer,
    ) -> None:
        """
        Initialise HiddenLayer object with number of nodes, inputs, activation function and previous layer if exists.

        Parameters:
            size (int): Size of Layer
            num_inputs (int): Number of inputs into Layer
            activation (Callable): Layer activation function
            weights_range (tuple[float, float]): Range for Layer weights
            bias_range (tuple[float, float]): Range for Layer bias
            prev_layer (InputLayer | Layer): Previous Layer to connect
        """
        super().__init__(size, num_inputs, activation, weights_range, bias_range, prev_layer)

    def mutate(self, shift_vals: float, prob_new_node: float, prob_remove_node: float) -> None:
        """
        Mutate HiddenLayer weights and biases, and potentially add/remove Node to/from HiddenLayer.

        Parameters:
            shift_vals (float): Factor to adjust Layer weights and biases by
            prob_new_node (float): Probability for a new Node, range [0, 1]
            prob_remove_node(float): Probability to remove a Node
        """
        super().mutate(shift_vals)

        rng = np.random.uniform(low=0, high=1)
        if rng < prob_new_node:
            self._add_node()
        elif rng > 1 - prob_remove_node:
            self._remove_node()

    def _add_node(self) -> None:
        """
        Add a random Node to HiddenLayer.
        """
        self._nodes.append(self.random_node)

        for node in self._next_layer._nodes:
            node.add_weight(self._next_layer._weights_range)

    def _remove_node(self) -> None:
        """
        Remove random Node from HiddenLayer.
        """
        if self.size == 1:
            return

        index = np.random.randint(low=0, high=self.size)
        del self._nodes[index]

        for node in self._next_layer._nodes:
            node.remove_weight(index)


class OutputLayer(Layer):
    """
    An output Layer in the NeuralNetwork.
    """

    def __init__(
        self,
        size: int,
        activation: Callable,
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        prev_layer: HiddenLayer,
    ) -> None:
        """
        Initialise OutputLayer object with number of nodes, inputs, activation function and previous layer.

        Parameters:
            size (int): Size of OutputLayer
            activation (Callable): OutputLayer activation function
            weights_range (tuple[float, float]): Range for OutputLayer weights
            bias_range (tuple[float, float]): Range for OutputLayer bias
            prev_layer (HiddenLayer): Previous HiddenLayer to connect
        """
        super().__init__(size, prev_layer.size, activation, weights_range, bias_range, prev_layer)
