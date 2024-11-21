from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from neural_network.math.activation_functions import ActivationFunction

rng = np.random.default_rng()


class Matrix:
    """
    This class handles the matrix mathematics required to pass data through neural networks.
    """

    def __init__(self, rows: int, cols: int, vals: NDArray | None = None) -> None:
        """
        Initialise Matrix with number of rows and columns, and optionally the matrix values.

        Parameters:
            rows (int): Number of rows in matrix
            cols (int): Number of columns in matrix
            vals (NDArray | None): Matrix values if specified
        """
        self._rows = rows
        self._cols = cols
        self._vals = vals

    def __str__(self) -> str:
        return str(self.vals)

    def __add__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals + other.vals)

    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals - other.vals)

    def __mul__(self, other: float | int) -> Matrix:
        return Matrix.from_array(self.vals * other)

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix.from_array(self.vals @ other.vals)

    @property
    def vals(self) -> NDArray:
        if self._vals is None:
            self._vals = np.zeros(shape=self.shape)
        return self._vals

    @property
    def as_list(self) -> list[float]:
        matrix_list = self.vals.tolist()[0]
        return cast(list[float], matrix_list)

    @property
    def shape(self) -> tuple:
        return (self._rows, self._cols)

    @classmethod
    def from_array(cls, matrix_array: NDArray | list[list[float]] | list[float]) -> Matrix:
        """
        Create a Matrix from an array.

        Parameters:
            matrix_array (NDArray | list[list[float]] | list[float]): Array of matrix values

        Returns:
            matrix (Matrix): Matrix with assigned values
        """
        matrix_array = np.array(matrix_array, dtype=object)
        try:
            _rows, _cols = matrix_array.shape
        except ValueError:
            matrix_array = np.expand_dims(matrix_array, axis=1)
            _rows, _cols = matrix_array.shape

        return cls(_rows, _cols, matrix_array)

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
        _vals = rng.uniform(low=low, high=high, size=(rows, cols))
        return cls.from_array(_vals)

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
    def multiply_element_wise(matrix: Matrix, other_matrix: Matrix) -> Matrix:
        """
        Multiply Matrix element wise with Matrix.

        Parameters:
            matrix (Matrix): Matrix to use for multiplication
            other_matrix (Matrix): Other Matrix to use for multiplication

        Returns:
            new_matrix (Matrix): Multiplied Matrix
        """
        new_matrix = matrix.vals * other_matrix.vals
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
        new_matrix = matrix.vals.transpose()
        return Matrix.from_array(new_matrix)

    @staticmethod
    def map(matrix: Matrix, activation: ActivationFunction) -> Matrix:
        """
        Map all values of Matrix through specified function.

        Parameters:
            matrix (Matrix): Matrix to map
            activation (ActivationFunction): Activation function to use for mapping

        Returns:
            new_matrix (Matrix): Matrix with mapped values
        """
        new_matrix = np.vectorize(activation.func)(matrix.vals)
        return Matrix.from_array(new_matrix)

    @staticmethod
    def average_matrix(matrix: Matrix, other_matrix: Matrix) -> Matrix:
        """
        Get average of two Matrix objects.

        Parameters:
            matrix (Matrix): Matrix to use for average
            other_matrix (Matrix): Other Matrix to use for average

        Returns:
            new_matrix (Matrix): Average of both matrices
        """
        new_matrix = np.average([matrix.vals, other_matrix.vals], axis=0)
        return Matrix.from_array(new_matrix)

    @staticmethod
    def mutated_matrix(matrix: Matrix, mutation_rate: float, random_range: list[float]) -> Matrix:
        """
        Mutate Matrix with a mutation rate.

        Parameters:
            matrix (Matrix): Matrix to use for average
            mutation_rate (float): Probability for mutation
            random_range (list[float]): Range for random number

        Returns:
            new_matrix (Matrix): Mutated Matrix
        """
        _mutation_matrix = rng.uniform(low=0, high=1, size=matrix.shape)
        new_matrix = np.where(
            _mutation_matrix < mutation_rate, rng.uniform(low=random_range[0], high=random_range[1]), matrix.vals
        )
        return Matrix.from_array(new_matrix)

    @staticmethod
    def pad_matrix(matrix: Matrix, new_shape: tuple[int, int]) -> Matrix:
        """
        Pad a Matrix with 0s to match a new shape.

        Parameters:
            matrix (Matrix): Matrix to pad
            new_shape (tuple[int, int]): New shape for Matrix

        Returns:
            new_matrix (Matrix): Padded Matrix
        """
        new_matrix = np.pad(
            matrix.vals,
            ((0, new_shape[0] - matrix.vals.shape[0]), (0, new_shape[1] - matrix.vals.shape[1])),
            mode="constant",
            constant_values=0,
        )
        return Matrix.from_array(new_matrix)

    @staticmethod
    def mix_matrices(input_matrix_a: Matrix, input_matrix_b: Matrix, output_matrix: Matrix) -> Matrix:
        """
        Mix two input Matrices and fill to match output Matrix shape. This can be used to perform crossover on neural
        networks with different topologies.

        Parameters:
            input_matrix_a (Matrix): Matrix to use for average
            input_matrix_b (Matrix): Other Matrix to use for average
            output_matrix (Matrix): Matrix to use for output shape and fill values

        Returns:
            new_matrix (Matrix): Average Matrix of both inputs with shape of output
        """
        max_shape = np.max([input_matrix_a.shape, input_matrix_b.shape, output_matrix.shape], axis=0)
        out_shape = output_matrix.shape

        padded_i1 = Matrix.pad_matrix(input_matrix_a, max_shape)
        padded_i2 = Matrix.pad_matrix(input_matrix_b, max_shape)
        padded_o = Matrix.pad_matrix(output_matrix, max_shape)

        avg_matrix = np.where(
            padded_i1.vals == 0,
            padded_i2.vals,
            np.where(
                padded_i2.vals == 0,
                padded_i1.vals,
                (padded_i1.vals + padded_i2.vals) / 2,
            ),
        )
        new_matrix = np.where(avg_matrix == 0, padded_o.vals, avg_matrix)[: out_shape[0], : out_shape[1]]
        return Matrix.from_array(new_matrix)

    def shift_vals(self, shift: float) -> None:
        """
        Adjust Matrix values by multiplying with some percentage.

        Parameters:
            shift (float): Factor to shift values by
        """
        _mult_array = rng.uniform(low=(1 - shift), high=(1 + shift), size=self.shape)
        self._vals = _mult_array
