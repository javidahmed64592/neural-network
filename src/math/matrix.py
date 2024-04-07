from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Matrix:
    def __init__(self, rows: int, cols: int, data: NDArray = Optional[None]) -> None:
        self._rows = rows
        self._cols = cols
        self._data = data

    def __str__(self) -> str:
        return str(self.data)

    @property
    def data(self):
        if not self._data.any():
            self._data = np.zeros(shape=self.shape)
        return self._data

    @property
    def shape(self) -> tuple:
        return (self._rows, self._cols)

    @classmethod
    def from_matrix_array(cls, matrix_array: NDArray) -> Matrix:
        rows, cols = matrix_array.shape
        _matrix = cls(rows, cols, matrix_array)
        return _matrix

    @classmethod
    def column_from_array(cls, matrix_array: NDArray) -> Matrix:
        rows = matrix_array.shape[0]
        cols = 1
        matrix_array = np.expand_dims(matrix_array, axis=1)
        _matrix = cls(rows, cols, matrix_array)
        return _matrix

    @classmethod
    def random_matrix(cls, rows: int, cols: int, low: float, high: float) -> Matrix:
        _data = np.random.uniform(low=low, high=high, size=(rows, cols))
        _matrix = cls(rows, cols, _data)
        return _matrix

    @classmethod
    def random_column(cls, rows: int, low: float, high: float) -> Matrix:
        _data = np.random.uniform(low=low, high=high, size=(rows, 1))
        _matrix = cls(rows, 1, _data)
        return _matrix

    @staticmethod
    def add(matrix: Matrix, val: Matrix | float) -> Matrix:
        if isinstance(val, Matrix):
            val = val.data

        new_matrix = matrix.data + val
        return Matrix.from_matrix_array(new_matrix)

    @staticmethod
    def multiply(matrix: Matrix, val: Matrix | float) -> Matrix:
        if isinstance(val, Matrix):
            val = val.data
        new_matrix = matrix.data.dot(val)
        return Matrix.from_matrix_array(new_matrix)

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        new_matrix = matrix.data.transpose()
        return Matrix.from_matrix_array(new_matrix)

    @staticmethod
    def map(matrix: Matrix, func: function) -> Matrix:
        new_matrix = np.vectorize(func)(matrix.data)
        return Matrix.from_matrix_array(new_matrix)
