from src.math import nn_math
from src.math.matrix import Matrix


def activation(x):
    return x


class TestNNMath:
    test_inputs = [1.0, 2.0]
    input_matrix = Matrix.from_matrix_array(test_inputs)

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(self):
        num_hidden_nodes = 5
        weights_ih = Matrix.random_matrix(rows=num_hidden_nodes, cols=len(self.test_inputs), low=-1, high=1)
        bias_h = Matrix.random_column(rows=num_hidden_nodes, low=-1, high=1)

        output = nn_math.feedforward_through_layer(
            input_vals=self.input_matrix, weights=weights_ih, bias=bias_h, activation=activation
        )

        expected_shape = (5, 1)
        actual_shape = output.shape
        assert actual_shape == expected_shape

    def test_given_errors_when_calculating_gradient_then_check_gradient_has_correct_shape(self):
        errors = Matrix.from_matrix_array([0.0, 1.0])
        lr = 0.1

        gradient = nn_math.calculate_gradient(self.input_matrix, errors, lr)

        expected_shape = (2, 1)
        actual_shape = gradient.shape

        assert actual_shape == expected_shape

    def test_given_gradients_when_calculating_delta_then_check_delta_has_correct_shape(self):
        gradient = Matrix.from_matrix_array([0.0, 1.0])

        delta = nn_math.calculate_delta(self.input_matrix, gradient)

        expected_shape = (2, 2)
        actual_shape = delta.shape

        assert actual_shape == expected_shape

    def test_given_errors_when_backpropagating_then_check_errors_have_correct_shape(self):
        weights = Matrix.random_matrix(2, 2, -1, 1)
        errors = Matrix.from_matrix_array([0.0, 1.0])

        delta = nn_math.calculate_next_errors(weights, errors)

        expected_shape = (2, 1)
        actual_shape = delta.shape

        assert actual_shape == expected_shape
