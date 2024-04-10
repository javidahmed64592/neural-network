from __future__ import annotations

from typing import Callable, List, cast

import numpy as np
from numpy.typing import NDArray


class Node:
    """
    This class can be used to create a Node object to be used within neural network layers.
    Each node has an array of random weights in the specified range. The node also has a random bias and a learning
    rate. These values affect the node's output and training.
    """

    LR = 0.00001

    def __init__(self, weights: NDArray, bias: float, activation: Callable) -> None:
        """
        Initialise Node object with number of weights, equal to number of inputs.

        Parameters:
            weights (NDArray): Node weights
            bias (float): Node bias
            activation (Callable): Activation function for node
        """
        self._weights = weights
        self._bias = bias
        self._activation = activation

    @classmethod
    def random_node(cls, size: int, weights_range: List[float], bias_range: List[float], activation: Callable) -> Node:
        """
        Create a Node with random weights and bias.

        Parameters:
            size (int): Number of Node weights
            weights_range (List[float]): Lower and upper limits for weights
            bias_range (List[float]): Lower and upper limits for bias
            activation (Callable): Node activation function

        Returns:
            node (Node): Node with random weights and bias
        """
        _weights = np.random.uniform(low=weights_range[0], high=weights_range[1], size=(size))
        _bias = np.random.uniform(low=bias_range[0], high=bias_range[1])
        node = cls(_weights, _bias, activation)
        return node

    def _calculate_output(self, inputs: List[float]) -> float:
        """
        Calculate node output from array of inputs.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (float): Inputs multiplied by weight
        """
        output = np.sum(self._weights * np.array(inputs)) + self._bias
        return cast(float, output)

    def _calculate_error(self, predicted_output: float, expected_output: float) -> float:
        """
        Calculate error between predicted output from feedforward and expected output.

        Parameters:
            predicted_output (float): Output from feedforward algorithm
            expected_output (float): Expected output from inputs

        Returns:
            error (float): Difference between predicted and expected output
        """
        error = expected_output - predicted_output
        return error

    def _calculate_delta_w(self, inputs: List[float], error: float) -> NDArray:
        """
        Calculate delta_w to modify weights through backpropagation.

        Paremeters:
            inputs (List[float]): List of input values
            error (float): Error from node output

        Returns:
            delta_w (NDArray): Array to add to weights
        """
        _delta_factor = error * self.LR
        delta_w = np.array(inputs) * _delta_factor
        return delta_w

    def _calculate_delta_b(self, error: float) -> float:
        """
        Calculate delta_b to modify bias through backpropagation.

        Paremeters:
            error (float): Error from node output

        Returns:
            delta_b (float): Number to add to bias
        """
        delta_b = error * self.LR
        return delta_b

    def _backpropagate(self, inputs: List[float], error: float) -> None:
        """
        Backpropagate error from inputs.

        Parameters:
            inputs (List[float]): List of input values
            error (float): Error from node output
        """
        self._weights += self._calculate_delta_w(inputs, error)
        self._bias += self._calculate_delta_b(error)

    def feedforward(self, inputs: List[float]) -> float:
        """
        Feedforward inputs and calculate output.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (float): Node output
        """
        _sum = self._calculate_output(inputs=inputs)
        output = self._activation(_sum)
        return cast(float, output)

    def train(self, inputs: List[float], target: float) -> None:
        """
        Train node with inputs and an expected output.

        Parameters:
            inputs (List[float]): Inputs to pass through feedforward
            target (float): Expected output from inputs
        """
        _output = self.feedforward(inputs)
        if _output != target:
            _error = self._calculate_error(_output, target)
            self._backpropagate(inputs, _error)
