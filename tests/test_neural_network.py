from neural_network.neural_network import NeuralNetwork


class TestNeuralNetwork:
    def test_given_nn_when_creating_layers_then_check_nn_has_correct_layers(
        self, mock_nn: NeuralNetwork, mock_len_inputs: int, mock_len_hidden: list[int], mock_len_outputs: int
    ) -> None:
        expected_sizes = [mock_len_inputs, *mock_len_hidden, mock_len_outputs]
        for index, layer in enumerate(mock_nn.layers):
            assert layer.size == expected_sizes[index]

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_len_outputs: int
    ) -> None:
        output = mock_nn.feedforward(mock_inputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs

    def test_given_inputs_and_outputs_when_training_then_check_error_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_outputs: list[float], mock_len_outputs: int
    ) -> None:
        output = mock_nn.train(mock_inputs, mock_outputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs

    def test_given_nn_when_adding_random_node_then_check_output_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_len_outputs: int
    ) -> None:
        mock_nn.mutate(1, 1, 0)
        output = mock_nn.feedforward(mock_inputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs

    def test_given_two_nns_with_same_shape_when_performing_crossover_then_check_feedforward_maintains_correct_shape(
        self, mock_inputs: list[float], mock_len_inputs: int, mock_len_outputs: int
    ) -> None:
        mock_nn_1 = NeuralNetwork(mock_len_inputs, mock_len_outputs, [5, 4, 3])
        mock_nn_2 = NeuralNetwork(mock_len_inputs, mock_len_outputs, [5, 4, 3])

        mock_nn_1.weights, mock_nn_1.bias = mock_nn_1.crossover(mock_nn_2, mock_nn_1, 0.01)
        mock_nn_2.weights, mock_nn_2.bias = mock_nn_2.crossover(mock_nn_1, mock_nn_2, 0.01)

        output_1 = mock_nn_1.feedforward(mock_inputs)
        output_2 = mock_nn_2.feedforward(mock_inputs)

        assert len(output_1) == mock_len_outputs
        assert len(output_2) == mock_len_outputs

    def test_given_three_nns_when_performing_crossover_then_check_feedforward_maintains_correct_shape(
        self, mock_inputs: list[float], mock_len_inputs: int, mock_len_outputs: int
    ) -> None:
        mock_nn_1 = NeuralNetwork(mock_len_inputs, mock_len_outputs, [5, 4, 3, 2, 1])
        mock_nn_2 = NeuralNetwork(mock_len_inputs, mock_len_outputs, [6, 3, 2, 5, 3])
        mock_nn_3 = NeuralNetwork(mock_len_inputs, mock_len_outputs, [3, 5, 4, 2, 3])

        mock_nn_3.weights, mock_nn_3.bias = mock_nn_3.crossover(mock_nn_1, mock_nn_2, 0.01)
        mock_nn_1.weights, mock_nn_1.bias = mock_nn_1.crossover(mock_nn_2, mock_nn_3, 0.01)
        mock_nn_2.weights, mock_nn_2.bias = mock_nn_2.crossover(mock_nn_3, mock_nn_1, 0.01)

        output_1 = mock_nn_1.feedforward(mock_inputs)
        output_2 = mock_nn_2.feedforward(mock_inputs)
        output_3 = mock_nn_3.feedforward(mock_inputs)

        assert len(output_1) == mock_len_outputs
        assert len(output_2) == mock_len_outputs
        assert len(output_3) == mock_len_outputs

    def test_given_nn_when_getting_node_connections_then_check_correct_number_of_connections_returned(
        self,
        mock_nn: NeuralNetwork,
        mock_layer_sizes: list[int],
    ) -> None:
        expected_count = 0
        for index in range(1, len(mock_layer_sizes)):
            expected_count += mock_layer_sizes[index] * mock_layer_sizes[index - 1]

        actual_count = len(mock_nn.connections)

        assert actual_count == expected_count
