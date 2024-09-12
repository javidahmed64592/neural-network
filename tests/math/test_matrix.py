import numpy as np

from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix


class TestMatrix:
    def test_given_no_vals_when_creating_matrix_then_check_matrix_has_zero_vals(
        self, mock_len_inputs: int, mock_len_outputs: int
    ) -> None:
        test_matrix = Matrix(rows=mock_len_inputs, cols=mock_len_outputs)
        assert not np.any(test_matrix.vals)

    def test_given_shape_when_creating_random_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range: list[float], mock_len_inputs: int, mock_len_outputs: int
    ) -> None:
        test_matrix = Matrix.random_matrix(
            rows=mock_len_inputs, cols=mock_len_outputs, low=mock_weights_range[0], high=mock_weights_range[1]
        )

        expected_shape = (mock_len_inputs, mock_len_outputs)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape, strict=False):
            assert actual == expected

    def test_given_shape_when_creating_random_column_then_check_matrix_has_correct_shape(
        self, mock_weights_range: list[float], mock_len_inputs: int
    ) -> None:
        test_matrix = Matrix.random_column(rows=mock_len_inputs, low=mock_weights_range[0], high=mock_weights_range[1])

        expected_shape = (mock_len_inputs, 1)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape, strict=False):
            assert actual == expected

    def test_given_2d_array_when_creating_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range: list[float], mock_len_inputs: int, mock_len_hidden: list[int]
    ) -> None:
        test_array = np.random.uniform(
            low=mock_weights_range[0], high=mock_weights_range[1], size=(mock_len_inputs, mock_len_hidden[0])
        )
        test_matrix = Matrix.from_array(matrix_array=test_array)

        expected_shape = (mock_len_inputs, mock_len_hidden[0])
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape, strict=False):
            assert actual == expected

    def test_given_1d_array_when_creating_matrix_then_check_matrix_has_correct_shape(
        self, mock_weights_range: list[float], mock_len_hidden: list[int]
    ) -> None:
        test_array = np.random.uniform(low=mock_weights_range[0], high=mock_weights_range[1], size=(mock_len_hidden[0]))
        test_matrix = Matrix.from_array(matrix_array=test_array)

        expected_shape = (mock_len_hidden[0], 1)
        actual_shape = test_matrix.shape
        for actual, expected in zip(actual_shape, expected_shape, strict=False):
            assert actual == expected

    def test_given_two_matrices_when_adding_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.add(matrix_1, matrix_2)

        expected_vals = np.array([[0, 3], [6, -2]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_subtracting_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.subtract(matrix_1, matrix_2)

        expected_vals = np.array([[2, 1], [2, 8]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.multiply(matrix_1, matrix_2)

        expected_vals = np.array([[3, -9], [2, -11], [6, -18]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_element_wise_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5], [3, 2]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.multiply_element_wise(matrix_1, matrix_2)

        expected_vals = np.array([[-1, 2], [8, -15], [6, 8]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_and_scalar_when_multiplying_then_check_new_matrix_correctly_calculated(self) -> None:
        array = np.array([[1, 2], [4, 3], [2, 4]])
        multiplier = 3

        matrix = Matrix.from_array(array)
        new_matrix = Matrix.multiply(matrix, multiplier)

        expected_vals = array * multiplier
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_when_mapping_then_check_new_matrix_correctly_calculated(
        self, mock_activation: ActivationFunction
    ) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])

        matrix_1 = Matrix.from_array(array_1)
        new_matrix = Matrix.map(matrix_1, mock_activation)

        expected_vals = np.array([[1, 2], [4, 3], [2, 4]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_calculating_average_matrix_then_check_new_matrix_correctly_calculated(
        self,
    ) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5], [3, 2]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = Matrix.average_matrix(matrix_1, matrix_2)

        expected_vals = np.array([[0, 1.5], [3, -1], [2.5, 3]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_when_mutating_then_check_new_matrix_with_same_shape_returned(self) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        mutation_rate = 0.5
        random_range = [1.0, 4.0]

        matrix_1 = Matrix.from_array(array_1)
        new_matrix = Matrix.mutated_matrix(matrix_1, mutation_rate, random_range)

        expected_shape = matrix_1.shape
        actual_shape = new_matrix.shape
        assert np.all(actual_shape == expected_shape)
        assert not np.all(matrix_1.vals == new_matrix.vals)

    def test_given_matrices_when_mixing_then_check_output_has_correct_shape(self) -> None:
        array_in_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_in_2 = np.array([[-1, 1], [2, -5], [3, 2], [3, 1]])
        array_out = np.array([[-1, 1, 2], [2, -5, 3], [3, 2, 1]])

        matrix_in_1 = Matrix.from_array(array_in_1)
        matrix_in_2 = Matrix.from_array(array_in_2)
        matrix_out = Matrix.from_array(array_out)

        new_matrix = Matrix.mix_matrices(matrix_in_1, matrix_in_2, matrix_out)

        assert np.all(new_matrix.shape == matrix_out.shape)

    def test_given_matrix_when_shifting_vals_then_check_vals_are_different(self) -> None:
        array = np.array([[1, 2], [4, 3], [2, 4]])

        matrix_1 = Matrix.from_array(array)
        matrix_2 = Matrix.from_array(array)

        assert np.all(matrix_1.vals == matrix_2.vals)

        matrix_1.shift_vals(0.5)
        assert not np.all(matrix_1.vals == matrix_2.vals)
