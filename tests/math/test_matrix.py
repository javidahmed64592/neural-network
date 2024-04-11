import numpy as np

from src.math.matrix import Matrix


class TestMatrix:
    def test_given_shape_when_creating_random_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range, mock_len_inputs, mock_len_outputs
    ):
        test_matrix = Matrix.random_matrix(
            rows=mock_len_inputs, cols=mock_len_outputs, low=mock_weights_range[0], high=mock_weights_range[1]
        )

        expected_shape = (mock_len_inputs, mock_len_outputs)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_shape_when_creating_random_column_then_check_matrix_has_correct_shape(
        self, mock_weights_range, mock_len_inputs
    ):
        test_matrix = Matrix.random_column(rows=mock_len_inputs, low=mock_weights_range[0], high=mock_weights_range[1])

        expected_shape = (mock_len_inputs, 1)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_2d_array_when_creating_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range, mock_len_inputs, mock_len_hidden
    ):
        test_array = np.random.uniform(
            low=mock_weights_range[0], high=mock_weights_range[1], size=(mock_len_inputs, mock_len_hidden)
        )
        test_matrix = Matrix.from_array(matrix_array=test_array)

        expected_shape = (mock_len_inputs, mock_len_hidden)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_1d_array_when_creating_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range, mock_len_hidden
    ):
        test_array = np.random.uniform(low=mock_weights_range[0], high=mock_weights_range[1], size=(mock_len_hidden))
        test_matrix = Matrix.from_array(matrix_array=test_array)

        expected_shape = (mock_len_hidden, 1)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_two_matrices_when_adding_then_check_new_matrix_correctly_calculated(self):
        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.add(matrix_1, matrix_2)

        expected_vals = np.array([[0, 3], [6, -2]])
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_subtracting_then_check_new_matrix_correctly_calculated(self):
        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.subtract(matrix_1, matrix_2)

        expected_vals = np.array([[2, 1], [2, 8]])
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_then_check_new_matrix_correctly_calculated(self):
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.multiply(matrix_1, matrix_2)

        expected_vals = np.array([[3, -9], [2, -11], [6, -18]])
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_element_wise_then_check_new_matrix_correctly_calculated(self):
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5], [3, 2]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.multiply_element_wise(matrix_1, matrix_2)

        expected_vals = np.array([[-1, 2], [8, -15], [6, 8]])
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_and_scalar_when_multiplying_then_check_new_matrix_correctly_calculated(self):
        array = np.array([[1, 2], [4, 3], [2, 4]])
        multiplier = 3

        matrix = Matrix.from_array(array)
        new_matrix = Matrix.multiply(matrix, multiplier)

        expected_vals = array * multiplier
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_when_mapping_then_check_new_matrix_correctly_calculated(self, mock_activation):
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])

        matrix_1 = Matrix.from_array(array_1)
        new_matrix = Matrix.map(matrix_1, mock_activation)

        expected_vals = np.array([[3, 6], [12, 9], [6, 12]])
        actual_vals = new_matrix.data
        assert np.all(actual_vals == expected_vals)
