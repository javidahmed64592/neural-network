from src.nn.neural_network import NeuralNetwork


class TestNeuralNetwork:
    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(self):
        test_inputs = [1, 0, 1]
        num_inputs = len(test_inputs)
        num_outputs = 4
        test_nn = NeuralNetwork(num_inputs, 5, num_outputs)

        output = test_nn.feedforward(test_inputs)

        expected_shape = (num_outputs,)
        actual_shape = output.shape

        assert actual_shape == expected_shape
