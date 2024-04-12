from neural_network.math import nn_math
from neural_network.math.matrix import Matrix


class TestNNMath:
    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_input_matrix, mock_inputs, mock_len_hidden, mock_weights_range, mock_bias_range, mock_activation
    ):
        weights_ih = Matrix.random_matrix(
            rows=mock_len_hidden, cols=len(mock_inputs), low=mock_weights_range[0], high=mock_weights_range[1]
        )
        bias_h = Matrix.random_column(rows=mock_len_hidden, low=mock_bias_range[0], high=mock_bias_range[1])

        output = nn_math.feedforward_through_layer(
            input_vals=mock_input_matrix, weights=weights_ih, bias=bias_h, activation=mock_activation
        )

        expected_shape = (mock_len_hidden, 1)
        actual_shape = output.shape
        assert actual_shape == expected_shape

    def test_given_errors_when_calculating_gradient_then_check_gradient_has_correct_shape(self, mock_input_matrix):
        errors = Matrix.from_array([0.0, 1.0, 0.5])
        lr = 0.1

        gradient = nn_math.calculate_gradient(mock_input_matrix, errors, lr)

        expected_shape = (3, 1)
        actual_shape = gradient.shape
        assert actual_shape == expected_shape

    def test_given_gradients_when_calculating_delta_then_check_delta_has_correct_shape(self, mock_input_matrix):
        gradient = Matrix.from_array([0.0, 1.0, 0.2])

        delta = nn_math.calculate_delta(mock_input_matrix, gradient)

        expected_shape = (3, 3)
        actual_shape = delta.shape
        assert actual_shape == expected_shape

    def test_given_errors_when_backpropagating_then_check_errors_have_correct_shape(self):
        weights = Matrix.random_matrix(2, 2, -1, 1)
        errors = Matrix.from_array([0.0, 1.0])

        delta = nn_math.calculate_next_errors(weights, errors)

        expected_shape = (2, 1)
        actual_shape = delta.shape
        assert actual_shape == expected_shape
