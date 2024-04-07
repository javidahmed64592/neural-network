import numpy as np

from src.nn.node import Node


class TestNode:
    test_inputs = [1.0, 2.0]
    test_expected_output = 1
    test_num_inputs = len(test_inputs)
    test_node = Node(test_num_inputs)

    def test_given_inputs_when_calculating_output_then_check_output_correctly_calculated(self):
        expected_output = (
            sum([(self.test_inputs[i] * self.test_node._weights[i]) for i in range(self.test_num_inputs)])
            + self.test_node._weights[-1]
        )
        actual_output = self.test_node._calculate_output(self.test_inputs)
        assert actual_output == expected_output

    def test_given_inputs_when_calculating_error_then_check_error_correctly_calculated(self):
        output = self.test_node._calculate_output(self.test_inputs)
        expected_error = self.test_expected_output - output
        actual_error = self.test_node._calculate_error(output, self.test_expected_output)
        assert actual_error == expected_error

    def test_given_inputs_and_error_when_calculating_delta_w_then_check_delta_w_correctly_calculated(self):
        output = self.test_node._calculate_output(self.test_inputs)
        error = self.test_node._calculate_error(output, self.test_expected_output)
        expected_delta_w = [(self.test_inputs[i] * error * self.test_node.LR) for i in range(self.test_num_inputs)]
        actual_delta_w = self.test_node._calculate_delta_w(self.test_inputs, error)
        for actual, expected in zip(actual_delta_w, expected_delta_w):
            assert actual == expected
