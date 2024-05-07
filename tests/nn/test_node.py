import numpy as np

from neural_network.nn.node import InputNode, Node


class TestNode:
    def test_given_size_when_creating_input_node_then_check_node_has_correct_weight(self) -> None:
        node = InputNode(0)
        assert node.weights == 1

    def test_given_size_when_creating_random_node_then_check_node_has_correct_no_of_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
    ) -> None:
        input_node = InputNode(0)
        node = Node.fully_connected(0, mock_weights_range, mock_bias_range, [input_node] * mock_len_inputs)
        assert len(node.weights) == mock_len_inputs
        assert np.all([mock_weights_range[0] <= weight <= mock_weights_range[1] for weight in node.weights])

    def test_given_new_weights_when_overwriting_node_weights_then_check_node_has_correct_weights(
        self,
        mock_len_inputs: int,
        mock_weights_range: list[float],
        mock_bias_range: list[float],
    ) -> None:
        input_node = InputNode(0)
        node = Node.fully_connected(0, mock_weights_range, mock_bias_range, [input_node] * mock_len_inputs)

        new_weights = [0.1, 0.2, 0.3]
        node.weights = new_weights

        for node_weight, new_weight in zip(node.weights, new_weights, strict=False):
            assert node_weight == new_weight


class TestNodeConnection:
    def test_given_two_nodes_when_getting_active_connection_weight_then_check_correct_weight_returned(self) -> None:
        node_1 = InputNode(0)
        node_2 = Node(0, 0.4)
        connection_weight = 0.5

        node_2.add_node(node_1, connection_weight)

        assert node_2._node_connections[0].weight == connection_weight

    def test_given_two_nodes_when_getting_inactive_connection_weight_then_check_correct_weight_returned(self) -> None:
        node_1 = InputNode(0)
        node_2 = Node(0, 0.4)
        connection_weight = 0.5

        node_2.add_node(node_1, connection_weight)
        node_2.toggle_node_connection(0)

        assert node_2._node_connections[0].weight == 0

    def test_given_two_nodes_when_reactivating_then_check_correct_weight_returned(self) -> None:
        node_1 = InputNode(0)
        node_2 = Node(0, 0.4)
        connection_weight = 0.5

        node_2.add_node(node_1, connection_weight)
        node_2.toggle_node_connection(0)
        node_2.toggle_node_connection(0)

        assert node_2._node_connections[0].weight == connection_weight

    def test_given_two_nodes_when_getting_connection_index_then_check_correct_values_returned(self) -> None:
        index_1 = 0
        index_2 = 1
        node_1 = InputNode(index_1)
        node_2 = Node(index_2, 0.4)
        connection_weight = 0.5

        node_2.add_node(node_1, connection_weight)

        assert np.all(node_2._node_connections[0].connection_index == [index_1, index_2])

    def test_given_two_nodes_when_getting_output_then_check_correct_value_returned(self) -> None:
        node_1 = InputNode(0)
        node_2 = Node(0, 0.4)
        connection_weight = 0.5
        input_val = np.array([0.6])

        node_2.add_node(node_1, connection_weight)
        node_1.set_input(input_val)

        assert node_2.output == (input_val * connection_weight) + node_2._bias

    def test_given_two_nodes_when_getting_inactive_output_then_check_zero_returned(self) -> None:
        node_1 = InputNode(0)
        node_2 = Node(0, 0.4)
        connection_weight = 0.5
        input_val = np.array([0.6])

        node_2.add_node(node_1, connection_weight)
        node_2.toggle_node_connection(0)
        node_1.set_input(input_val)

        assert node_2.output == node_2._bias
