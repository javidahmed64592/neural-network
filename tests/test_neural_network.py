from neural_network.neural_network import NeuralNetwork


class TestNeuralNetwork:
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

    def test_given_nn_when_removing_random_node_then_check_output_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_len_outputs: int
    ) -> None:
        mock_nn.mutate(1, 0, 1)
        output = mock_nn.feedforward(mock_inputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs
