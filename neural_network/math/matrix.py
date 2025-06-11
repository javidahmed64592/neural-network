"""Matrix class for neural network operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from neural_network.math.activation_functions import ActivationFunction
from neural_network.protobuf.neural_network_types import MatrixDataType

rng = np.random.default_rng()


class Matrix:
    """Class for matrix operations in neural networks."""

    def __init__(self, vals: NDArray) -> None:
        """Initialize Matrix with values.

        :param NDArray vals:
            Matrix values.
        """
        self.vals = vals

    def __str__(self) -> str:
        """Return string representation of the matrix.

        :return str:
            String representation.
        """
        return str(self.vals)

    def __add__(self, other: Matrix) -> Matrix:
        """Add two matrices.

        :param Matrix other:
            Matrix to add.
        :return Matrix:
            Resulting matrix.
        """
        return Matrix.from_array(self.vals + other.vals)

    def __sub__(self, other: Matrix) -> Matrix:
        """Subtract two matrices.

        :param Matrix other:
            Matrix to subtract.
        :return Matrix:
            Resulting matrix.
        """
        return Matrix.from_array(self.vals - other.vals)

    def __mul__(self, other: float | int | Matrix) -> Matrix:
        """Multiply matrix by scalar or element-wise by another matrix.

        :param float|int|Matrix other:
            Scalar or matrix to multiply.
        :return Matrix:
            Resulting matrix.
        """
        if isinstance(other, Matrix):
            return Matrix.from_array(self.vals * other.vals)
        return Matrix.from_array(self.vals * other)

    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication.

        :param Matrix other:
            Matrix to multiply.
        :return Matrix:
            Resulting matrix.
        """
        return Matrix.from_array(self.vals @ other.vals)

    @property
    def as_list(self) -> list[float]:
        """Return matrix as a flat list.

        :return list[float]:
            Matrix values as list.
        """
        matrix_list = self.vals.tolist()[0]
        return cast(list[float], matrix_list)

    @property
    def shape(self) -> tuple:
        """Return shape of the matrix.

        :return tuple:
            Shape of the matrix.
        """
        return self.vals.shape

    @classmethod
    def from_array(cls, matrix_array: NDArray | list[list[float]] | list[float]) -> Matrix:
        """Create a Matrix from an array.

        :param NDArray|list[list[float]]|list[float] matrix_array:
            Array of matrix values.
        :return Matrix:
            Matrix with assigned values.
        """
        matrix_array = np.array(matrix_array, dtype=np.float64)
        if matrix_array.ndim == 1:
            matrix_array = np.expand_dims(matrix_array, axis=1)
        return cls(matrix_array)

    @classmethod
    def from_protobuf(cls, matrix_data: MatrixDataType) -> Matrix:
        """Create a Matrix from Protobuf data.

        :param MatrixDataType matrix_data:
            Protobuf data containing matrix values.
        :return Matrix:
            Matrix with assigned values.
        """
        matrix_array = np.array(matrix_data.data, dtype=np.float64).reshape((matrix_data.rows, matrix_data.cols))
        return cls.from_array(matrix_array)

    @staticmethod
    def to_protobuf(matrix: Matrix) -> MatrixDataType:
        """Convert Matrix to Protobuf data.

        :param Matrix matrix:
            Matrix to convert.
        :return MatrixDataType:
            Protobuf data containing matrix values.
        """
        return MatrixDataType(
            data=matrix.vals.flatten().tolist(),
            rows=matrix.shape[0],
            cols=matrix.shape[1],
        )

    @staticmethod
    def _uniform(low: float, high: float, size: tuple[int, int] | int | None = None) -> NDArray:
        """Create an array of random values in specified range.

        :param float low:
            Lower boundary for random number.
        :param float high:
            Upper boundary for random number.
        :param tuple[int, int]|int|None size:
            Shape of the array to create.
        :return NDArray:
            Array with random values.
        """
        return rng.uniform(low=low, high=high, size=size)

    @classmethod
    def random_matrix(cls, rows: int, cols: int, low: float, high: float) -> Matrix:
        """Create Matrix of specified shape with random values in specified range.

        :param int rows:
            Number of rows in matrix.
        :param int cols:
            Number of columns in matrix.
        :param float low:
            Lower boundary for random number.
        :param float high:
            Upper boundary for random number.
        :return Matrix:
            Matrix with random values.
        """
        return cls.from_array(cls._uniform(low=low, high=high, size=(rows, cols)))

    @classmethod
    def random_column(cls, rows: int, low: float, high: float) -> Matrix:
        """Create column Matrix with random values in specified range.

        :param int rows:
            Number of rows in matrix.
        :param float low:
            Lower boundary for random number.
        :param float high:
            Upper boundary for random number.
        :return Matrix:
            Column Matrix with random values.
        """
        return cls.random_matrix(rows=rows, cols=1, low=low, high=high)

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        """Return transpose of Matrix.

        :param Matrix matrix:
            Matrix to transpose.
        :return Matrix:
            Transposed Matrix.
        """
        return Matrix.from_array(matrix.vals.transpose())

    @staticmethod
    def map(matrix: Matrix, activation: type[ActivationFunction]) -> Matrix:
        """Map all values of Matrix through specified activation function.

        :param Matrix matrix:
            Matrix to map.
        :param type[ActivationFunction] activation:
            Activation function to use for mapping.
        :return Matrix:
            Matrix with mapped values.
        """
        return Matrix.from_array(np.vectorize(activation.func)(matrix.vals))

    @staticmethod
    def crossover(
        matrix: Matrix,
        other_matrix: Matrix,
        crossover_func: Callable,
    ) -> Matrix:
        """Crossover two Matrix objects by mixing their values.

        :param Matrix matrix:
            Matrix to use for average.
        :param Matrix other_matrix:
            Other Matrix to use for average.
        :param Callable crossover_func:
            Custom function for crossover operations.
            Should accept (element, other_element, roll) and return a float.
        :return Matrix:
            New Matrix with mixed values.
        """
        vectorized_crossover = np.vectorize(crossover_func)
        crossover_rolls = Matrix._uniform(low=0, high=1, size=matrix.shape)
        new_matrix = vectorized_crossover(matrix.vals, other_matrix.vals, crossover_rolls)
        return Matrix.from_array(new_matrix)
