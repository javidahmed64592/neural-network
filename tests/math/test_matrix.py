from unittest.mock import patch

import numpy as np
import pytest

from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix
from neural_network.protobuf.neural_network_types import MatrixDataType


class TestMatrix:
    def test_given_protobuf_message_when_creating_matrix_then_check_has_correct_vals(self) -> None:
        test_vals = [1.0, 2.0, 3.0]
        test_rows = 1
        test_cols = 3
        matrix_data = MatrixDataType(data=test_vals, rows=test_rows, cols=test_cols)

        test_matrix = Matrix.from_protobuf(matrix_data=matrix_data)
        actual_vals = test_matrix.vals.flatten().tolist()

        assert actual_vals == test_vals
        assert test_matrix.shape == (test_rows, test_cols)

    def test_given_matrix_when_converting_to_protobuf_then_check_has_correct_vals(
        self, mock_weights_range: list[float], mock_len_inputs: int, mock_len_hidden: list[int]
    ) -> None:
        test_matrix = Matrix.random_matrix(
            rows=mock_len_inputs, cols=mock_len_hidden[0], low=mock_weights_range[0], high=mock_weights_range[1]
        )
        matrix_data = Matrix.to_protobuf(test_matrix)

        assert matrix_data.data == pytest.approx(test_matrix.vals.flatten().tolist())
        assert matrix_data.rows == mock_len_inputs
        assert matrix_data.cols == mock_len_hidden[0]

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
        test_array = Matrix._uniform(
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
        test_array = Matrix._uniform(low=mock_weights_range[0], high=mock_weights_range[1], size=(mock_len_hidden[0]))
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
        new_matrix = matrix_1 + matrix_2

        expected_vals = np.array([[0, 3], [6, -2]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_subtracting_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = matrix_1 - matrix_2

        expected_vals = np.array([[2, 1], [2, 8]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = matrix_1 @ matrix_2

        expected_vals = np.array([[3, -9], [2, -11], [6, -18]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_multiplying_element_wise_then_check_new_matrix_correctly_calculated(self) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])
        array_2 = np.array([[-1, 1], [2, -5], [3, 2]])

        matrix_1 = Matrix.from_array(array_1)
        matrix_2 = Matrix.from_array(array_2)
        new_matrix = matrix_1 * matrix_2

        expected_vals = np.array([[-1, 2], [8, -15], [6, 8]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_and_scalar_when_multiplying_then_check_new_matrix_correctly_calculated(self) -> None:
        array = np.array([[1, 2], [4, 3], [2, 4]])
        multiplier = 3

        matrix = Matrix.from_array(array)
        new_matrix = matrix * multiplier

        expected_vals = array * multiplier
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_matrix_when_mapping_then_check_new_matrix_correctly_calculated(
        self, mock_activation: type[ActivationFunction]
    ) -> None:
        array_1 = np.array([[1, 2], [4, 3], [2, 4]])

        matrix_1 = Matrix.from_array(array_1)
        new_matrix = Matrix.map(matrix_1, mock_activation)

        expected_vals = np.array([[1, 2], [4, 3], [2, 4]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)

    def test_given_two_matrices_when_performing_crossover_then_check_new_matrix_correctly_calculated(
        self, mock_weights_range: list[float]
    ) -> None:
        mutation_rate = 0.5

        def _mock_crossover_func(element: float, other_element: float, roll: float) -> float:
            if roll < mutation_rate:
                return element
            return other_element

        array_1 = np.array([[1, 2], [4, 3]])
        array_2 = np.array([[-1, 1], [2, -5]])
        roll_array = np.array([[0.1, 0.6], [0.4, 0.8]])

        with patch("neural_network.math.matrix.Matrix._uniform", return_value=roll_array):
            matrix_1 = Matrix.from_array(array_1)
            matrix_2 = Matrix.from_array(array_2)
            new_matrix = Matrix.crossover(
                matrix=matrix_1,
                other_matrix=matrix_2,
                crossover_func=_mock_crossover_func,
            )

        expected_vals = np.array([[1, 1], [4, -5]])
        actual_vals = new_matrix.vals
        assert np.all(actual_vals == expected_vals)
