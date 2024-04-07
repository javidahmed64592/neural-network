from typing import List, cast

import numpy as np
from numpy.typing import NDArray


class Node:
    """
    This class can be used to create a Node object to be used within neural network layers.
    Each node has an array of random weights in the specified range. The node also has a random bias and a learning
    rate. These values affect the node's output and training.
    """

    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]
    LR = 0.001

    def __init__(self, num_weights: int) -> None:
        """
        Initialise Node object with number of weights, equal to number of inputs.

        Parameters:
            num_weights (int): Number of weights for node
        """
        self._weights = np.random.uniform(low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1], size=(num_weights))
        self._bias = np.random.uniform(low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])

    def _activation(self, x: float) -> float:
        """
        Activation function for node output.

        Parameters:
            x (float): Output to pass through activation function

        Returns:
            output (float): Node output passed through activation function
        """
        output = np.sign(x)
        return cast(float, output)

    def _calculate_output(self, inputs: NDArray) -> float:
        """
        Calculate node output from array of inputs.

        Parameters:
            inputs (NDArray): Array of input values

        Returns:
            output (float): Inputs multiplied by weight, then adding bias
        """
        output = np.sum(self._weights * inputs) + self._bias
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
        error = predicted_output - expected_output
        return error

    def _calculate_delta_w(self, inputs: NDArray, error: float) -> NDArray:
        """
        Calculate delta_w to modify weights through backpropagation.

        Paremeters:
            inputs (NDArray): Array of inputs
            error (float): Error from node output

        Returns:
            delta_w (NDArray): Array to add to weights
        """
        delta_w = inputs * error * self.LR
        return delta_w

    def _backpropagate(self, inputs: List[float], error: float) -> None:
        """
        Backpropagate error from inputs.

        Parameters:
            inputs (List[float]): Array of inputs
            error (float): Error from node output
        """
        self._weights += self._calculate_delta_w(np.array(inputs), error)

    def feedforward(self, inputs: List[float]) -> float:
        """
        Feedforward inputs and calculate output.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (float): Node output
        """
        sum = self._calculate_output(inputs=np.array(inputs))
        output = self._activation(sum)
        return output

    def train(self, inputs: List[float], target: float) -> None:
        """
        Train node with inputs and an expected output.

        Parameters:
            inputs (List[float]): Inputs to pass through feedforward
            target (float): Expected output from inputs
        """
        guess = self.feedforward(inputs)
        if guess != target:
            error = self._calculate_error(guess, target)
            self._backpropagate(inputs, error)
