class TestNode:
    test_expected_output = 1

    def test_given_inputs_when_calculating_output_then_check_output_correctly_calculated(
        self, mock_node, mock_inputs, mock_len_inputs
    ):
        expected_output = (
            sum([(mock_inputs[i] * mock_node._weights[i]) for i in range(mock_len_inputs)]) + mock_node._bias
        )
        actual_output = mock_node._calculate_output(mock_inputs)
        assert actual_output == expected_output

    def test_given_inputs_when_calculating_error_then_check_error_correctly_calculated(self, mock_node, mock_inputs):
        output = mock_node._calculate_output(mock_inputs)
        expected_error = self.test_expected_output - output
        actual_error = mock_node._calculate_error(output, self.test_expected_output)
        assert actual_error == expected_error

    def test_given_inputs_and_error_when_calculating_delta_w_then_check_delta_w_correctly_calculated(
        self, mock_node, mock_inputs, mock_len_inputs
    ):
        output = mock_node._calculate_output(mock_inputs)
        error = mock_node._calculate_error(output, self.test_expected_output)
        expected_delta_w = [(mock_inputs[i] * error * mock_node.LR) for i in range(mock_len_inputs)]
        actual_delta_w = mock_node._calculate_delta_w(mock_inputs, error)
        for actual, expected in zip(actual_delta_w, expected_delta_w):
            assert actual == expected

    def test_given_inputs_and_error_when_calculating_delta_b_then_check_delta_b_correctly_calculated(
        self, mock_node, mock_inputs
    ):
        output = mock_node._calculate_output(mock_inputs)
        error = mock_node._calculate_error(output, self.test_expected_output)
        expected_delta_b = error * mock_node.LR
        actual_delta_b = mock_node._calculate_delta_b(error)
        assert actual_delta_b == expected_delta_b
