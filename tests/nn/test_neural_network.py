from src.nn.neural_network import NeuralNetwork


class TestNeuralNetwork:
    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(self):
        test_inputs = [1.0, 0.0, 1.0]
        num_inputs = len(test_inputs)
        num_outputs = 4
        test_nn = NeuralNetwork(num_inputs, 5, num_outputs)

        output = test_nn.feedforward(test_inputs)

        expected_len = num_outputs
        actual_len = len(output)

        assert actual_len == expected_len

    def test_given_inputs_and_outputs_when_training_then_check_error_has_correct_shape(self):
        test_inputs = [1.0, 0.0, 1.0]
        num_inputs = len(test_inputs)

        expected_outputs = [2.0, 3.0]
        num_outputs = len(expected_outputs)
        test_nn = NeuralNetwork(num_inputs, 5, num_outputs)

        output = test_nn.train(test_inputs, expected_outputs)

        expected_len = num_outputs
        actual_len = len(output)

        assert actual_len == expected_len
