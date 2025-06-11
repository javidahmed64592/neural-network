"""Layer classes for neural network architecture."""

from __future__ import annotations

from neural_network.math import nn_math
from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import Optimizer, SGDOptimizer


class Layer:
    """Class representing a neural network layer with weights, biases, and activation function."""

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        lr: float = 0.1,
        optimizer: type[Optimizer] | None = None,
    ) -> None:
        """Initialise Layer object with number of nodes, activation function, weights range and bias range.

        :param int size:
            Number of nodes in the layer.
        :param type[ActivationFunction] activation:
            Activation function class for the layer.
        :param tuple[float, float] weights_range:
            Range for initializing layer weights.
        :param tuple[float, float] bias_range:
            Range for initializing layer biases.
        :param float lr:
            Learning rate for the optimizer (default is 0.1).
        :param type[Optimizer] | None optimizer:
            Optimizer class for updating weights and biases (optional).
        """
        self._prev_layer: Layer | None = None
        self._next_layer: Layer | None = None
        self._weights: Matrix | None = None
        self._bias: Matrix | None = None
        optimizer_class: type[Optimizer] = optimizer or SGDOptimizer
        self._optimizer = optimizer_class(learning_rate=lr)

        self._size = size
        self._activation = activation
        self._weights_range = weights_range
        self._bias_range = bias_range

    def __str__(self) -> str:
        """Return a string representation of the Layer.

        :return str:
            String describing the layer.
        """
        return f"Size: {self.size} \t| Activation: {self._activation} \tWeights: {self.weights} | Bias: {self.bias}"

    @property
    def size(self) -> int:
        """Return the size of the layer.

        :return int:
            Number of nodes in the layer.
        """
        return self._size

    @property
    def num_inputs(self) -> int:
        """Return the number of inputs to the layer.

        :return int:
            Number of inputs (size of previous layer or 1 if none).
        """
        if self._prev_layer is None:
            return 1
        return self._prev_layer.size

    @property
    def weights(self) -> Matrix:
        """Return the weights matrix for the layer, initializing if necessary.

        :return Matrix:
            Weights matrix.
        """
        if not self._weights:
            self._weights = Matrix.random_matrix(
                self._size, self.num_inputs, self._weights_range[0], self._weights_range[1]
            )
        return self._weights

    @weights.setter
    def weights(self, new_weights: Matrix) -> None:
        """Set the weights matrix for the layer.

        :param Matrix new_weights:
            New weights matrix.
        """
        self._weights = new_weights

    @property
    def bias(self) -> Matrix:
        """Return the bias matrix for the layer, initializing if necessary.

        :return Matrix:
            Bias matrix.
        """
        if not self._bias:
            self._bias = Matrix.random_column(self._size, self._bias_range[0], self._bias_range[1])
        return self._bias

    @bias.setter
    def bias(self, new_bias: Matrix) -> None:
        """Set the bias matrix for the layer.

        :param Matrix new_bias:
            New bias matrix.
        """
        self._bias = new_bias

    def set_prev_layer(self, prev_layer: Layer) -> None:
        """Connect this layer to the previous layer.

        :param Layer prev_layer:
            Layer preceding this one.
        """
        self._prev_layer = prev_layer
        prev_layer._next_layer = self

    def feedforward(self, vals: Matrix) -> Matrix:
        """Feedforward values through the layer.

        :param Matrix vals:
            Input values to feedforward.
        :return Matrix:
            Output values from the layer.
        """
        output = nn_math.feedforward_through_layer(
            input_vals=vals, weights=self.weights, bias=self.bias, activation=self._activation
        )
        self._layer_input = vals
        self._layer_output = output
        return output

    def backpropagate_error(self, errors: Matrix) -> None:
        """Backpropagate errors during training.

        :param Matrix errors:
            Errors from the next layer.
        """
        gradient = nn_math.calculate_gradient(
            activation=self._activation, layer_vals=self._layer_output, errors=errors, lr=1.0
        )
        weight_gradients = nn_math.calculate_delta(layer_vals=self._layer_input, gradients=gradient)

        self.weights = self._optimizer.update_weights(self.weights, weight_gradients)
        self.bias = self._optimizer.update_bias(self.bias, gradient)
        self._optimizer.step()


class InputLayer(Layer):
    """Input layer for a neural network."""

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        lr: float = 0.1,
        optimizer: type[Optimizer] | None = None,
    ) -> None:
        """Initialise InputLayer object with number of nodes and activation function.

        :param int size:
            Number of input nodes.
        :param type[ActivationFunction] activation:
            Activation function class for the input layer.
        :param float lr:
            Learning rate for the optimizer (default is 0.1).
        :param type[Optimizer] | None optimizer:
            Optimizer class for updating weights and biases (optional).
        """
        super().__init__(size, activation, (1.0, 1.0), (0.0, 0.0), lr, optimizer)

    def __str__(self) -> str:
        """Return a string representation of the InputLayer.

        :return str:
            String describing the input layer.
        """
        return f"Input \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"

    def feedforward(self, vals: Matrix) -> Matrix:
        """Set input values for the InputLayer.

        :param Matrix vals:
            Input values.
        :return Matrix:
            Output values (same as input).
        """
        self._layer_input = vals
        self._layer_output = vals
        return vals


class HiddenLayer(Layer):
    """Hidden layer for a neural network."""

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        lr: float = 0.1,
        optimizer: type[Optimizer] | None = None,
    ) -> None:
        """Initialize HiddenLayer object.

        :param int size:
            Number of nodes in the hidden layer.
        :param type[ActivationFunction] activation:
            Activation function class for the hidden layer.
        :param tuple[float, float] weights_range:
            Range for initializing hidden layer weights.
        :param tuple[float, float] bias_range:
            Range for initializing hidden layer biases.
        :param float lr:
            Learning rate for the optimizer (default is 0.1).
        :param type[Optimizer] | None optimizer:
            Optimizer class for updating weights and biases (optional).
        """
        super().__init__(size, activation, weights_range, bias_range, lr, optimizer)

    def __str__(self) -> str:
        """Return a string representation of the HiddenLayer.

        :return str:
            String describing the hidden layer.
        """
        return f"Hidden \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"


class OutputLayer(Layer):
    """Output layer for a neural network."""

    def __init__(
        self,
        size: int,
        activation: type[ActivationFunction],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        optimizer: type[Optimizer] | None = None,
        lr: float = 0.1,
    ) -> None:
        """Initialise OutputLayer object with number of nodes, activation function, weights range and bias range.

        :param int size:
            Number of output nodes.
        :param type[ActivationFunction] activation:
            Activation function class for the output layer.
        :param tuple[float, float] weights_range:
            Range for initializing output layer weights.
        :param tuple[float, float] bias_range:
            Range for initializing output layer biases.
        :param float lr:
            Learning rate for the optimizer (default is 0.1).
        :param type[Optimizer] | None optimizer:
            Optimizer class for updating weights and biases (optional).
        """
        super().__init__(size, activation, weights_range, bias_range, lr, optimizer)

    def __str__(self) -> str:
        """Return a string representation of the OutputLayer.

        :return str:
            String describing the output layer.
        """
        return f"Output \t| Size: {self.size} \t| Activation: {self._activation().__str__()}"
