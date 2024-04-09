import numpy as np

from src.math.matrix import Matrix


def activation_function(x):
    return x * 3


class TestMatrix:
    test_weights_range = [-1, 1]
    test_bias_range = [-1, 1]

    def test_given_shape_when_creating_random_matrix_then_check_matrix_has_correct_shape(self):
        test_rows, test_cols = (3, 2)
        test_matrix = Matrix.random_matrix(
            rows=test_rows, cols=test_cols, low=self.test_weights_range[0], high=self.test_weights_range[1]
        )

        expected_shape = (test_rows, test_cols)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_shape_when_creating_random_column_then_check_matrix_has_correct_shape(self):
        test_rows, test_cols = (3, 1)
        test_matrix = Matrix.random_column(
            rows=test_rows, low=self.test_weights_range[0], high=self.test_weights_range[1]
        )

        expected_shape = (test_rows, test_cols)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_2d_array_when_creating_matrix_then_check_matrix_has_correct_shape(self):
        test_rows, test_cols = (4, 5)
        test_array = np.random.uniform(
            low=self.test_weights_range[0], high=self.test_weights_range[1], size=(test_rows, test_cols)
        )
        test_matrix = Matrix.from_matrix_array(matrix_array=test_array)

        expected_shape = (test_rows, test_cols)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_1d_array_when_creating_matrix_then_check_matrix_has_correct_shape(self):
        test_rows, test_cols = (4, 1)
        test_array = np.random.uniform(
            low=self.test_weights_range[0], high=self.test_weights_range[1], size=(test_rows)
        )
        test_matrix = Matrix.from_matrix_array(matrix_array=test_array)

        expected_shape = (test_rows, test_cols)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape):
            assert actual == expected

    def test_given_two_matrices_when_adding_then_check_new_matrix_correctly_calculated(self):
        test_array_1 = np.array([[1, 2], [4, 3]])
        test_array_2 = np.array([[-1, 1], [2, -5]])

        test_matrix_1 = Matrix.from_matrix_array(test_array_1)
        test_matrix_2 = Matrix.from_matrix_array(test_array_2)

        new_matrix = Matrix.add(test_matrix_1, test_matrix_2)

        expected_vals = np.array([[0, 3], [6, -2]])
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_subtracting_then_check_new_matrix_correctly_calculated(self):
        test_array_1 = np.array([[1, 2], [4, 3]])
        test_array_2 = np.array([[-1, 1], [2, -5]])

        test_matrix_1 = Matrix.from_matrix_array(test_array_1)
        test_matrix_2 = Matrix.from_matrix_array(test_array_2)

        new_matrix = Matrix.subtract(test_matrix_1, test_matrix_2)

        expected_vals = np.array([[2, 1], [2, 8]])
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_then_check_new_matrix_correctly_calculated(self):
        test_array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        test_array_2 = np.array([[-1, 1], [2, -5]])

        test_matrix_1 = Matrix.from_matrix_array(test_array_1)
        test_matrix_2 = Matrix.from_matrix_array(test_array_2)

        new_matrix = Matrix.multiply(test_matrix_1, test_matrix_2)

        expected_vals = np.array([[3, -9], [2, -11], [6, -18]])
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_element_wise_then_check_new_matrix_correctly_calculated(self):
        test_array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        test_array_2 = np.array([[-1, 1], [2, -5], [3, 2]])

        test_matrix_1 = Matrix.from_matrix_array(test_array_1)
        test_matrix_2 = Matrix.from_matrix_array(test_array_2)

        new_matrix = Matrix.multiply_element_wise(test_matrix_1, test_matrix_2)

        expected_vals = np.array([[-1, 2], [8, -15], [6, 8]])
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_and_scalar_when_multiplying_then_check_new_matrix_correctly_calculated(self):
        test_array = np.array([[1, 2], [4, 3], [2, 4]])
        test_scale = 3

        test_matrix = Matrix.from_matrix_array(test_array)

        new_matrix = Matrix.multiply(test_matrix, test_scale)

        expected_vals = test_array * test_scale
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_when_mapping_then_check_new_matrix_correctly_calculated(self):
        test_array_1 = np.array([[1, 2], [4, 3], [2, 4]])

        test_matrix_1 = Matrix.from_matrix_array(test_array_1)

        new_matrix = Matrix.map(test_matrix_1, activation_function)

        expected_vals = np.array([[3, 6], [12, 9], [6, 12]])
        actual_vals = new_matrix.data

        assert np.all(actual_vals == expected_vals)
