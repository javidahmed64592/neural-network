from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from neural_network.math.activation_functions import ActivationFunction
from neural_network.protobuf.neural_network_types import MatrixDataType

rng = np.random.default_rng()


class Matrix:
    """
    This class handles the matrix mathematics required to pass data through neural networks.
    """

    def __init__(self, vals: NDArray) -> None:
        """
        Initialise Matrix with number of rows and columns, and optionally the matrix values.

        Parameters:
            vals (NDArray): Matrix values
        """
        self.vals = vals

    def __str__(self) -> str:
        return str(self.vals)

    def __add__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals + other.vals)

    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals - other.vals)

    def __mul__(self, other: float | int | Matrix) -> Matrix:
        if isinstance(other, Matrix):
            return Matrix.from_array(self.vals * other.vals)
        return Matrix.from_array(self.vals * other)

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals @ other.vals)

    @property
    def as_list(self) -> list[float]:
        matrix_list = self.vals.tolist()[0]
        return cast(list[float], matrix_list)

    @property
    def shape(self) -> tuple:
        return self.vals.shape

    @classmethod
    def from_array(cls, matrix_array: NDArray | list[list[float]] | list[float]) -> Matrix:
        """
        Create a Matrix from an array.

        Parameters:
            matrix_array (NDArray | list[list[float]] | list[float]): Array of matrix values

        Returns:
            matrix (Matrix): Matrix with assigned values
        """
        matrix_array = np.array(matrix_array, dtype=np.float64)
        if matrix_array.ndim == 1:
            matrix_array = np.expand_dims(matrix_array, axis=1)
        return cls(matrix_array)

    @classmethod
    def from_protobuf(cls, matrix_data: MatrixDataType) -> Matrix:
        """
        Create a Matrix from Protobuf data.

        Parameters:
            matrix_data (MatrixDataType): Protobuf data containing matrix values

        Returns:
            matrix (Matrix): Matrix with assigned values
        """
        matrix_array = np.array(matrix_data.data, dtype=np.float64).reshape((matrix_data.rows, matrix_data.cols))
        return cls.from_array(matrix_array)

    @staticmethod
    def to_protobuf(matrix: Matrix) -> MatrixDataType:
        """
        Convert Matrix to Protobuf data.

        Parameters:
            matrix (Matrix): Matrix to convert

        Returns:
            matrix_data (MatrixDataType): Protobuf data containing matrix values
        """
        return MatrixDataType(
            data=matrix.vals.flatten().tolist(),
            rows=matrix.shape[0],
            cols=matrix.shape[1],
        )

    @classmethod
    def random_matrix(cls, rows: int, cols: int, low: float, high: float) -> Matrix:
        """
        Create Matrix of specified shape with random values in specified range.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
            low (float): Lower boundary for random number
            high (float): Upper boundary for random number

        Returns:
            matrix (Matrix): Matrix with random values
        """
        return cls.from_array(cls._uniform(low=low, high=high, size=(rows, cols)))

    @classmethod
    def random_column(cls, rows: int, low: float, high: float) -> Matrix:
        """
        Create column Matrix with random values in specified range.

        Parameters:
            rows (int): Number of rows in matrix
            low (float): Lower boundary for random number
            high (float): Upper boundary for random number

        Returns:
            matrix (Matrix): Column Matrix with random values
        """
        return cls.random_matrix(rows=rows, cols=1, low=low, high=high)

    @staticmethod
    def _uniform(low: float, high: float, size: tuple[int, int] | int | None = None) -> NDArray:
        """
        Create an array of random values in specified range.

        Parameters:
            low (float): Lower boundary for random number
            high (float): Upper boundary for random number
            size (tuple[int, int] | int | None): Shape of the array to create

        Returns:
            NDArray: Array with random values
        """
        return rng.uniform(low=low, high=high, size=size)

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        """
        Return transpose of Matrix.

        Parameters:
            matrix (Matrix): Matrix to transpose

        Returns:
            new_matrix (Matrix): Transposed Matrix
        """
        return Matrix.from_array(matrix.vals.transpose())

    @staticmethod
    def map(matrix: Matrix, activation: type[ActivationFunction]) -> Matrix:
        """
        Map all values of Matrix through specified function.

        Parameters:
            matrix (Matrix): Matrix to map
            activation (type[ActivationFunction]): Activation function to use for mapping

        Returns:
            new_matrix (Matrix): Matrix with mapped values
        """
        return Matrix.from_array(np.vectorize(activation.func)(matrix.vals))

    @staticmethod
    def crossover(
        matrix: Matrix,
        other_matrix: Matrix,
        crossover_func: Callable,
    ) -> Matrix:
        """
        Crossover two Matrix objects by mixing their values.

        Parameters:
            matrix (Matrix): Matrix to use for average
            other_matrix (Matrix): Other Matrix to use for average
            crossover_func (Callable): Custom function for crossover operations.
                Should accept (element, other_element, roll) and return a float.

        Returns:
            new_matrix (Matrix): New Matrix with mixed values
        """
        vectorized_crossover = np.vectorize(crossover_func)
        crossover_rolls = Matrix._uniform(low=0, high=1, size=matrix.shape)
        new_matrix = vectorized_crossover(matrix.vals, other_matrix.vals, crossover_rolls)
        return Matrix.from_array(new_matrix)
