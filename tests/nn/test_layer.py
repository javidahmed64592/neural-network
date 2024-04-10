class TestLayer:
    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(
        self, mock_layer, mock_len_hidden, mock_len_inputs
    ):
        expected_weights_shape = (mock_len_hidden, mock_len_inputs)
        expected_bias_shape = (mock_len_hidden, 1)

        actual_weights_shape = mock_layer.weights.shape
        actual_bias_shape = mock_layer.bias.shape

        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_layer, mock_len_hidden, mock_input_matrix
    ):
        output = mock_layer.feedforward(mock_input_matrix)

        expected_output_shape = (mock_len_hidden, 1)
        actual_output_shape = output.shape

        assert actual_output_shape == expected_output_shape
