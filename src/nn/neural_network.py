from typing import List, cast

import numpy as np

from src.math.matrix import Matrix


class NeuralNetwork:
    """
    This class can be used to create a NeuralNetwork with specified layer sizes. This neural network can feedforward a
    list of inputs, and be trained through backpropagation of errors.
    """

    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
        Initialise NeuralNetwork object with specified layer sizes.

        Parameters:
            input_nodes (int): Number of inputs
            hidden_nodes (int): Number of hidden nodes
            output_nodes (int): Number of outputs
        """
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes

        self._weights_ih = Matrix.random_matrix(
            rows=self._hidden_nodes, cols=self._input_nodes, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1]
        )
        self._weights_ho = Matrix.random_matrix(
            rows=self._output_nodes, cols=self._hidden_nodes, low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1]
        )

        self._bias_h = Matrix.random_column(rows=self._hidden_nodes, low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])
        self._bias_o = Matrix.random_column(rows=self._output_nodes, low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])

    @staticmethod
    def _activation(x):
        return x * 2

    def feedforward(self, inputs: List[float]) -> List[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (List[float]): List of input values

        Returns:
            output (List[float]): List of outputs
        """
        input_matrix = Matrix.from_matrix_array(np.array(inputs))

        hidden = Matrix.multiply(self._weights_ih, input_matrix)
        hidden = Matrix.add(hidden, self._bias_h)
        hidden = Matrix.map(hidden, NeuralNetwork._activation)

        output = Matrix.multiply(self._weights_ho, hidden)
        output = Matrix.add(output, self._bias_o)
        output = Matrix.map(output, NeuralNetwork._activation)
        output = Matrix.transpose(output)
        return cast(List[float], output.data[0])

    def train(self, inputs: List[float], expected_outputs: List[float]) -> None:
        """
        Train NeuralNetwork using a list of input values and expected  output values.

        Parameters:
            inputs (List[float]): List of input values
            expected_outputs (List[float]): List of output values
        """
        pass
