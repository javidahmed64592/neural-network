from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class Node:
    """
    This class can be used to create a Node object to be used within neural network layers.
    Each node has an array of random weights in the specified range. The node also has a random bias and a learning
    rate. These values affect the node's output and training.
    """

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
    def random_node(
        cls, size: int, weights_range: tuple[float, float], bias_range: tuple[float, float], activation: Callable
    ) -> Node:
        """
        Create a Node with random weights and bias.

        Parameters:
            size (int): Number of Node weights
            weights_range (tuple[float, float]): Lower and upper limits for weights
            bias_range (tuple[float, float]): Lower and upper limits for bias
            activation (Callable): Node activation function

        Returns:
            node (Node): Node with random weights and bias
        """
        _weights = np.random.uniform(low=weights_range[0], high=weights_range[1], size=(size))
        _bias = np.random.uniform(low=bias_range[0], high=bias_range[1])
        node = cls(_weights, _bias, activation)
        return node

    def add_weight(self, weights_range: tuple[float, float]) -> None:
        """
        Add a random weight to Node.

        Parameters:
            weights_range (tuple[float, float]): Range for random weight to add
        """
        _weight = np.random.uniform(low=weights_range[0], high=weights_range[1])
        self._weights = np.append(self._weights, _weight)

    def remove_weight(self, index: int) -> None:
        """
        Remove weight from Node at index.

        Parameters:
            index (int): Index to remove weight at
        """
        self._weights = np.delete(self._weights, index)
