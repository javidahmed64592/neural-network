from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class Node:
    """
    This class can be used to create a Node object to be used within neural network layers.
    Each node has a random bias and connections to other Nodes. If the Node is an input Node, it will have a weight of
    1, and the Node will have a bias of 0.
    """

    def __init__(self, index: int, bias: float) -> None:
        """
        Initialise Node object with index and bias.

        Parameters:
            index (int): Node position in Layer
            bias (float): Node bias
        """
        self._node_connections: list[NodeConnection] = []
        self._index = index
        self._bias = bias

    @property
    def weights(self) -> NDArray:
        return np.array([nc.weight for nc in self._node_connections])

    @weights.setter
    def weights(self, new_weights: list[float]) -> None:
        for index, nc in enumerate(self._node_connections):
            nc.connection_weight = new_weights[index]

    @classmethod
    def random_node(
        cls,
        index: int,
        bias_range: tuple[float, float],
    ) -> Node:
        """
        Create a Node with random weights and bias.

        Parameters:
            index (int): Node position in Layer
            bias_range (tuple[float, float]): Lower and upper limits for bias

        Returns:
            node (Node): Node with random weights and bias with NodeConnections to input Nodes
        """
        node = cls(index, np.random.uniform(low=bias_range[0], high=bias_range[1]))
        return node

    @classmethod
    def fully_connected(
        cls,
        index: int,
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        input_nodes: list[Node],
    ) -> Node:
        """
        Create a Node with random weights and bias.

        Parameters:
            index (int): Node position in Layer
            weights_range (tuple[float, float]): Lower and upper limits for weights
            bias_range (tuple[float, float]): Lower and upper limits for bias
            input_nodes (list[Node]): Input Nodes for new random Node

        Returns:
            node (Node): Node with random weights and bias with NodeConnections to input Nodes
        """
        node = cls.random_node(index, bias_range)

        new_weights = np.random.uniform(low=weights_range[0], high=weights_range[1], size=len(input_nodes))
        for input_node, new_weight in zip(input_nodes, new_weights, strict=False):
            node.add_node(input_node, new_weight)

        return node

    def add_node(self, node: Node, weight: float) -> None:
        """
        Add a NodeConnection to Node.

        Parameters:
            node (Node): Node to connect
            weight (float): Weight of NodeConnection
        """
        self._node_connections.append(NodeConnection(node, self, weight))

    def toggle_node_connection(self, index: int) -> None:
        """
        Toggle NodeConnection at index.

        Parameters:
            index (int): Node index in Layer to toggle connection
        """
        self._node_connections[index].toggle_active()


class InputNode(Node):
    """
    An InputNode with a weight of 1 and bias of 0.
    """

    def __init__(self, index: int) -> None:
        """
        Initialise InputNode object with index.

        Parameters:
            index (int): Node position in Layer
        """
        super().__init__(index, 0)

    @property
    def weights(self) -> NDArray:
        return np.ones(shape=1)

    @weights.setter
    def weights(self, new_weights: list[float]) -> None:
        return


@dataclass
class NodeConnection:
    """
    This class can be used to hold the weight of a connection between two Nodes and toggle that connection on/off.

    Parameters:
        input_node (Node): Input Node of connection
        output_node (Node): Output Node of connection
        connection_weight (float): Weight of connection
        is_active (bool): Inactive returns connection weight as 0
    """

    input_node: Node
    output_node: Node
    connection_weight: float
    is_active: bool = True

    @property
    def weight(self) -> float:
        return [0, self.connection_weight][self.is_active]

    @property
    def connection_index(self) -> tuple[int, int]:
        return [self.input_node._index, self.output_node._index]

    def toggle_active(self) -> None:
        """
        Toggle NodeConnection active status between on/off.
        """
        self.is_active = not self.is_active
