from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray


class Matrix:
    """
    This class handles the matrix mathematics required to pass data through neural networks.
    """

    def __init__(self, rows: int, cols: int, data: Optional[NDArray] = None) -> None:
        """
        Initialise Matrix with number of rows and columns, and optionally the matrix values.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
            data (Optional[NDArray]): Matrix values if specified
        """
        self._rows = rows
        self._cols = cols
        self._data = data

    def __str__(self) -> str:
        return str(self.data)

    @property
    def data(self):
        if self._data is None:
            self._data = np.zeros(shape=self.shape)
        return self._data

    @property
    def shape(self) -> tuple:
        return (self._rows, self._cols)

    @classmethod
    def from_array(cls, matrix_array: NDArray | List[List[float]] | List[float]) -> Matrix:
        """
        Create a Matrix from an array.

        Parameters:
            matrix_array (NDArray | List[List[float]] | List[float]): Array of matrix values

        Returns:
            matrix (Matrix): Matrix with assigned values
        """
        matrix_array = np.array(matrix_array)
        try:
            _rows, _cols = matrix_array.shape
        except ValueError:
            matrix_array = np.expand_dims(matrix_array, axis=1)
            _rows, _cols = matrix_array.shape

        matrix = cls(_rows, _cols, matrix_array)
        return matrix

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
        _data = np.random.uniform(low=low, high=high, size=(rows, cols))
        matrix = cls.from_array(_data)
        return matrix

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
        matrix = cls.random_matrix(rows=rows, cols=1, low=low, high=high)
        return matrix

    @staticmethod
    def add(matrix: Matrix, other_matrix: Matrix) -> Matrix:
        """
        Add two Matrix objects.

        Parameters:
            matrix (Matrix): Matrix to use in sum
            other_matrix (Matrix): Other Matrix to use in sum

        Returns:
            new_matrix (Matrix): Sum of both matrices
        """
        new_matrix = matrix.data + other_matrix.data
        return Matrix.from_array(new_matrix)

    @staticmethod
    def subtract(matrix: Matrix, other_matrix: Matrix) -> Matrix:
        """
        Subtract two Matrix objects.

        Parameters:
            matrix (Matrix): Matrix to use in subtraction
            other_matrix (Matrix): Other Matrix to use in subtraction

        Returns:
            new_matrix (Matrix): Difference between both matrices
        """
        new_matrix = matrix.data - other_matrix.data
        return Matrix.from_array(new_matrix)

    @staticmethod
    def multiply(matrix: Matrix, val: Matrix | float) -> Matrix:
        """
        Multiply Matrix with scalar or Matrix.

        Parameters:
            matrix (Matrix): Matrix to to use for multiplication
            val (Matrix | float): Matrix or scalar to use for multiplication

        Returns:
            new_matrix (Matrix): Multiplied Matrix
        """
        if isinstance(val, Matrix):
            val = val.data
        new_matrix = matrix.data.dot(val)
        return Matrix.from_array(new_matrix)

    @staticmethod
    def multiply_element_wise(matrix: Matrix, other_matrix: Matrix) -> Matrix:
        """
        Multiply Matrix element wise with Matrix.

        Parameters:
            matrix (Matrix): Matrix to use for multiplication
            other_matrix (Matrix): Other Matrix to use for multiplication

        Returns:
            new_matrix (Matrix): Multiplied Matrix
        """
        new_matrix = matrix.data * other_matrix.data
        return Matrix.from_array(new_matrix)

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        """
        Return transpose of Matrix.

        Parameters:
            matrix (Matrix): Matrix to transpose

        Returns:
            new_matrix (Matrix): Transposed Matrix
        """
        new_matrix = matrix.data.transpose()
        return Matrix.from_array(new_matrix)

    @staticmethod
    def map(matrix: Matrix, func: Callable) -> Matrix:
        """
        Map all values of Matrix through specified function.

        Parameters:
            matrix (Matrix): Matrix to map

        Returns:
            new_matrix (Matrix): Matrix with mapped values
        """
        new_matrix = np.vectorize(func)(matrix.data)
        return Matrix.from_array(new_matrix)
