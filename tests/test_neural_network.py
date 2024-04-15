class TestNeuralNetwork:
    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_nn, mock_inputs, mock_len_outputs
    ):
        output = mock_nn.feedforward(mock_inputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs

    def test_given_inputs_and_outputs_when_training_then_check_error_has_correct_shape(
        self, mock_nn, mock_inputs, mock_outputs, mock_len_outputs
    ):
        output = mock_nn.train(mock_inputs, mock_outputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs