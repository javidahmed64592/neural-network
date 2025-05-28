from __future__ import annotations

import numpy as np

from neural_network.math import nn_math
from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix

rng = np.random.default_rng()


class Layer:
    """
    This class creates a NeuralNetwork Layer and has weights, biases, and activation function.
    """

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Initialise Layer object with number of nodes, activation function, weights range and bias range.

        Parameters:
            size (int): Size of Layer
            activation (type[ActivationFunction]): Layer activation function
            weights_range (tuple[float, float]): Range for Layer weights
            bias_range (tuple[float, float]): Range for Layer bias
        """
        self._prev_layer: Layer | None = None
        self._next_layer: Layer | None = None
        self._weights: Matrix | None = None
        self._bias: Matrix | None = None

        self._size = size
        self._activation = activation
        self._weights_range = weights_range
        self._bias_range = bias_range

    def __str__(self) -> str:
        return f"Size: {self.size} \t| Activation: {self._activation} \tWeights: {self.weights} | Bias: {self.bias}"

    @property
    def size(self) -> int:
        return self._size

    @property
    def num_inputs(self) -> int:
        if self._prev_layer is None:
            return 1
        return self._prev_layer.size

    @property
    def weights(self) -> Matrix:
        if not self._weights:
            self._weights = Matrix.random_matrix(
                self._size, self.num_inputs, self._weights_range[0], self._weights_range[1]
            )
        return self._weights

    @weights.setter
    def weights(self, new_weights: Matrix) -> None:
        self._weights = new_weights

    @property
    def bias(self) -> Matrix:
        if not self._bias:
            self._bias = Matrix.random_column(self._size, self._bias_range[0], self._bias_range[1])
        return self._bias

    @bias.setter
    def bias(self, new_bias: Matrix) -> None:
        self._bias = new_bias

    def set_prev_layer(self, prev_layer: Layer) -> None:
        """
        Connect Layer with previous Layer.

        Parameters:
            prev_layer (Layer): Layer preceding this one
        """
        self._prev_layer = prev_layer
        prev_layer._next_layer = self

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
        gradient = nn_math.calculate_gradient(
            activation=self._activation, layer_vals=self._layer_output, errors=errors, lr=learning_rate
        )
        delta = nn_math.calculate_delta(layer_vals=self._layer_input, gradients=gradient)
        self.weights = self.weights + delta
        self.bias = self.bias + gradient


class InputLayer(Layer):
    """
    An input Layer in the NeuralNetwork.
    """

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
    ) -> None:
        """
        Initialise InputLayer object with number of nodes and activation function.

        Parameters:
            size (int): Size of InputLayer
            activation (type[ActivationFunction]): InputLayer activation function
        """
        super().__init__(size, activation, (1.0, 1.0), (0.0, 0.0))

    def __str__(self) -> str:
        return f"Input \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"

    def feedforward(self, vals: Matrix) -> Matrix:
        """
        Set InputLayer values.

        Parameters:
            vals (Matrix): Input values

        Returns:
            output (Matrix): Layer output from inputs
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
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Initialise HiddenLayer object with number of nodes, activation function, weights range and bias range.

        Parameters:
            size (int): Size of HiddenLayer
            activation (type[ActivationFunction]): HiddenLayer activation function
            weights_range (tuple[float, float]): Range for HiddenLayer weights
            bias_range (tuple[float, float]): Range for HiddenLayer bias
        """
        super().__init__(size, activation, weights_range, bias_range)

    def __str__(self) -> str:
        return f"Hidden \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"


class OutputLayer(Layer):
    """
    An output Layer in the NeuralNetwork.
    """

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
    ) -> None:
        """
        Initialise OutputLayer object with number of nodes, activation function, weights range and bias range.

        Parameters:
            size (int): Size of OutputLayer
            activation (type[ActivationFunction]): OutputLayer activation function
            weights_range (tuple[float, float]): Range for OutputLayer weights
            bias_range (tuple[float, float]): Range for OutputLayer bias
        """
        super().__init__(size, activation, weights_range, bias_range)

    def __str__(self) -> str:
        return f"Output \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"
