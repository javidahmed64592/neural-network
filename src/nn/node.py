from typing import List, cast

import numpy as np
from numpy.typing import NDArray


class Node:
    WEIGHTS_RANGE = [-1, 1]
    BIAS_RANGE = [-1, 1]
    LR = 0.001

    def __init__(self, num_weights: int) -> None:
        self._weights = np.random.uniform(low=self.WEIGHTS_RANGE[0], high=self.WEIGHTS_RANGE[1], size=(num_weights))
        self._bias = np.random.uniform(low=self.BIAS_RANGE[0], high=self.BIAS_RANGE[1])

        self._train_num_correct = 0
        self._train_num_wrong = 0

    def _activation(self, x: float) -> float:
        output = cast(float, np.sign(x))
        return output

    def _calculate_output(self, inputs: NDArray) -> float:
        output = cast(float, np.sum(self._weights * inputs) + self._bias)
        return output

    def _calculate_error(self, predicted_output: float, actual_output: float) -> float:
        error = predicted_output - actual_output
        return error

    def _calculate_delta_w(self, inputs: NDArray, error: float) -> NDArray:
        delta_w = inputs * error * self.LR
        return delta_w

    def _backpropagate(self, inputs: tuple, error: float) -> None:
        self._weights += self._calculate_delta_w(inputs, error)

    def feedforward(self, inputs: NDArray) -> float:
        sum = self._calculate_output(inputs=inputs)
        return self._activation(sum)

    def train(self, inputs: tuple, target: float) -> None:
        inputs = np.array(inputs)
        guess = self.feedforward(inputs)
        error = self._calculate_error(guess, target)
        if guess == target:
            self._train_num_correct += 1
        else:
            self._train_num_wrong += 1
            self._backpropagate(inputs, error)

    def train_with_dataset(self, training_data: List[tuple], targets: List[float]) -> None:
        self._train_num_wrong = 0
        self._train_num_correct = 0

        for point, target in zip(training_data, targets):
            self.train(point, target)

        accuracy = self._train_num_correct / len(targets)
        print(f"Training complete! Accuracy: {accuracy}")
