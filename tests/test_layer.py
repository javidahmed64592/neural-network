from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.matrix import Matrix


class TestLayer:
    def test_given_layers_when_adding_nodes_then_check_layers_have_correct_sizes(
        self,
        mock_input_layer: InputLayer,
        mock_hidden_layer_1: HiddenLayer,
        mock_output_layer: OutputLayer,
        mock_len_inputs: int,
        mock_len_hidden: list[int],
        mock_len_outputs: int,
    ) -> None:
        assert mock_input_layer.size == mock_len_inputs
        assert mock_hidden_layer_1.size == mock_len_hidden[0]
        assert mock_hidden_layer_1.num_inputs == mock_len_inputs
        assert mock_output_layer.size == mock_len_outputs
        assert mock_output_layer.num_inputs == mock_len_hidden[-1]

    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(
        self, mock_hidden_layer_1: HiddenLayer, mock_len_hidden: list[int], mock_len_inputs: int
    ) -> None:
        expected_weights_shape = (mock_len_hidden[0], mock_len_inputs)
        expected_bias_shape = (mock_len_hidden[0], 1)

        actual_weights_shape = mock_hidden_layer_1.weights.shape
        actual_bias_shape = mock_hidden_layer_1.bias.shape
        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape

    def test_given_weights_when_getting_random_weight_then_check_value_is_within_range(
        self, mock_hidden_layer_1: HiddenLayer
    ) -> None:
        random_weight = mock_hidden_layer_1.random_weight
        weight_range = mock_hidden_layer_1._weights_range
        assert weight_range[0] <= random_weight <= weight_range[1]

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_hidden_layer_1: HiddenLayer, mock_len_hidden: list[int], mock_input_matrix: Matrix
    ) -> None:
        output = mock_hidden_layer_1.feedforward(mock_input_matrix)

        expected_output_shape = (mock_len_hidden[0], 1)
        actual_output_shape = output.shape
        assert actual_output_shape == expected_output_shape
