import numpy as np
from numpy.typing import NDArray


class ActivationFunctions:
    """
    This class is used to define activation functions.
    """

    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def relu(x: float) -> float:
        return max(x, 0)

    @staticmethod
    def sigmoid(x: NDArray | float) -> float:
        try:
            y = 1 / (1 + np.exp(-x))
        except TypeError:
            y = 1 / (1 + np.exp(-x.astype(float)))
        return float(y)
