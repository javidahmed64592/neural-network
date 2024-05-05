from collections.abc import Callable

from neural_network.nn.node import Node


class TestNode:
    def test_given_size_when_creating_input_node_then_check_node_has_correct_no_of_weights(
        self,
        mock_len_inputs: int,
        mock_activation: Callable,
    ) -> None:
        node = Node.input_node(mock_len_inputs, mock_activation)
        assert len(node.weights) == mock_len_inputs

    def test_given_size_when_creating_random_node_then_check_node_has_correct_no_of_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
        mock_activation: Callable,
    ) -> None:
        node = Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation)
        assert len(node.weights) == mock_len_inputs
