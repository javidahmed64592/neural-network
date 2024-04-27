from neural_network.math.matrix import Matrix
from neural_network.nn.layer import Layer


class TestLayer:
    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(
        self, mock_hidden_layer: Layer, mock_len_hidden: int, mock_len_inputs: int
    ) -> None:
        expected_weights_shape = (mock_len_hidden, mock_len_inputs)
        expected_bias_shape = (mock_len_hidden, 1)

        actual_weights_shape = mock_hidden_layer.weights.shape
        actual_bias_shape = mock_hidden_layer.bias.shape
        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_hidden_layer: Layer, mock_len_hidden: int, mock_input_matrix: Matrix
    ) -> None:
        output = mock_hidden_layer.feedforward(mock_input_matrix)

        expected_output_shape = (mock_len_hidden, 1)
        actual_output_shape = output.shape
        assert actual_output_shape == expected_output_shape

    def test_given_layers_when_adding_node_to_hidden_then_check_layers_have_correct_shape(
        self,
        mock_hidden_layer: Layer,
        mock_output_layer: Layer,
    ) -> None:
        initial_size = mock_hidden_layer.size
        initial_inputs = mock_output_layer.num_inputs

        mock_output_layer.mutate(0, 1, 0)

        expected_size = initial_size + 1
        expected_inputs = initial_inputs + 1
        actual_size = mock_hidden_layer.size
        actual_inputs = mock_output_layer.num_inputs

        assert actual_size == expected_size
        assert actual_inputs == expected_inputs

    def test_given_layers_when_removing_node_from_hidden_then_check_layers_have_correct_shape(
        self,
        mock_hidden_layer: Layer,
        mock_output_layer: Layer,
    ) -> None:
        initial_size = mock_hidden_layer.size
        initial_inputs = mock_output_layer.num_inputs

        mock_output_layer.mutate(0, 0, 1)

        expected_size = initial_size - 1
        expected_inputs = initial_inputs - 1
        actual_size = mock_hidden_layer.size
        actual_inputs = mock_output_layer.num_inputs

        assert actual_size == expected_size
        assert actual_inputs == expected_inputs
