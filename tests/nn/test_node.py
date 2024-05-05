from collections.abc import Callable

import numpy as np

from neural_network.nn.node import Node


class TestNode:
    def test_given_size_when_creating_input_node_then_check_node_has_correct_weight(
        self,
        mock_activation: Callable,
    ) -> None:
        node = Node.input_node(mock_activation)
        assert node.weights == 1

    def test_given_size_when_creating_random_node_then_check_node_has_correct_no_of_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
        mock_activation: Callable,
    ) -> None:
        node = Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation)
        assert len(node.weights) == mock_len_inputs
        assert np.all([mock_weights_range[0] <= weight <= mock_weights_range[1] for weight in node.weights])

    def test_given_new_weights_when_overwriting_node_weights_then_check_node_has_correct_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
        mock_activation: Callable,
    ) -> None:
        node = Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation)

        new_weights = [0.1, 0.2, 0.3]
        node.weights = new_weights

        for node_weight, new_weight in zip(node.weights, new_weights, strict=False):
            assert node_weight == new_weight
