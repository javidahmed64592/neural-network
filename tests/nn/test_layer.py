from src.nn.layer import Layer


def activation(x):
    return x


class TestLayer:
    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(self):
        num_nodes = 5
        num_inputs = 3

        test_layer = Layer(num_nodes=num_nodes, num_inputs=num_inputs, activation=activation)

        expected_weights_shape = (num_nodes, num_inputs)
        expected_bias_shape = (num_nodes, 1)

        actual_weights_shape = test_layer.weights.shape
        actual_bias_shape = test_layer.bias.shape

        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape
