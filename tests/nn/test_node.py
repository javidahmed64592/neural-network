from collections.abc import Callable

import numpy as np

from neural_network.nn.node import Node


class TestNode:
    def test_given_size_when_creating_input_node_then_check_node_has_correct_weight(
        self,
        mock_activation: Callable,
    ) -> None:
        node = Node.input_node(0, mock_activation)
        assert node.weights == 1

    def test_given_size_when_creating_random_node_then_check_node_has_correct_no_of_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
        mock_activation: Callable,
    ) -> None:
        input_node = Node.input_node(0, mock_activation)
        node = Node.random_node(0, mock_weights_range, mock_bias_range, mock_activation, [input_node] * mock_len_inputs)
        assert len(node.weights) == mock_len_inputs
        assert np.all([mock_weights_range[0] <= weight <= mock_weights_range[1] for weight in node.weights])

    def test_given_new_weights_when_overwriting_node_weights_then_check_node_has_correct_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
        mock_activation: Callable,
    ) -> None:
        input_node = Node.input_node(0, mock_activation)
        node = Node.random_node(0, mock_weights_range, mock_bias_range, mock_activation, [input_node] * mock_len_inputs)

        new_weights = [0.1, 0.2, 0.3]
        node.weights = new_weights

        for node_weight, new_weight in zip(node.weights, new_weights, strict=False):
            assert node_weight == new_weight


class TestNodeConnection:
    def test_given_two_nodes_when_getting_active_connection_weight_then_check_correct_weight_returned(
        self, mock_activation: Callable
    ) -> None:
        node_1 = Node(0, 0.3, mock_activation)
        node_2 = Node(0, 0.4, mock_activation)
        connection_weight = 0.5

        node_1.add_node(node_2, connection_weight)

        assert node_1._node_connections[0].weight == connection_weight

    def test_given_two_nodes_when_getting_inactive_connection_weight_then_check_correct_weight_returned(
        self, mock_activation: Callable
    ) -> None:
        node_1 = Node(0, 0.3, mock_activation)
        node_2 = Node(0, 0.4, mock_activation)
        connection_weight = 0.5

        node_1.add_node(node_2, connection_weight)
        node_1.toggle_node_connection(0)

        assert node_1._node_connections[0].weight == 0

    def test_given_two_nodes_when_getting_connection_index_then_check_correct_values_returned(
        self, mock_activation: Callable
    ) -> None:
        index_1 = 0
        index_2 = 1
        node_1 = Node(index_1, 0.3, mock_activation)
        node_2 = Node(index_2, 0.4, mock_activation)
        connection_weight = 0.5

        node_1.add_node(node_2, connection_weight)

        assert np.all(node_1._node_connections[0].connection_index == [index_2, index_1])
