from src.math.matrix import Matrix
from src.nn.layer import Layer


def activation(x):
    return x


class TestLayer:
    num_nodes = 5
    num_inputs = 3
    test_layer = Layer(size=num_nodes, num_inputs=num_inputs, activation=activation)

    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(self):
        expected_weights_shape = (self.num_nodes, self.num_inputs)
        expected_bias_shape = (self.num_nodes, 1)

        actual_weights_shape = self.test_layer.weights.shape
        actual_bias_shape = self.test_layer.bias.shape

        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(self):
        test_inputs = Matrix.from_array([1.0, 0.3, 0.7])

        output = self.test_layer.feedforward(test_inputs)

        expected_output_shape = (self.num_nodes, 1)
        actual_output_shape = output.shape

        assert actual_output_shape == expected_output_shape
