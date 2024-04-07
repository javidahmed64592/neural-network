from typing import List

import numpy as np

from src.math.matrix import Matrix


class NeuralNetwork:
    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
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

    def feedforward(self, inputs: List[float]) -> float:
        input_matrix = Matrix.column_from_array(np.array(inputs))

        hidden = Matrix.multiply(self._weights_ih, input_matrix)
        hidden = Matrix.add(hidden, self._bias_h)
        hidden = Matrix.map(hidden, NeuralNetwork._activation)

        output = Matrix.multiply(self._weights_ho, hidden)
        output = Matrix.add(output, self._bias_o)
        output = Matrix.map(output, NeuralNetwork._activation)
        output = Matrix.transpose(output)
        return output.data[0]

    def train(self, inputs: List[float], expected_output: float) -> None:
        pass
