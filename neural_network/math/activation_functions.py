import numpy as np
from numpy.typing import NDArray


class ActivationFunction:
    """
    This class is used to define activation functions.
    """

    @staticmethod
    def func(x: float) -> float:
        pass

    @staticmethod
    def derivative(x: float) -> float:
        pass


class LinearActivation(ActivationFunction):
    @staticmethod
    def func(x: float) -> float:
        return x

    @staticmethod
    def derivative(x: float) -> float:
        return 1


class ReluActivation(ActivationFunction):
    @staticmethod
    def func(x: float) -> float:
        return max(x, 0)

    @staticmethod
    def derivative(x: float) -> float:
        return [0, 1][x > 0]


class SigmoidActivation(ActivationFunction):
    @staticmethod
    def func(x: NDArray | float) -> float:
        try:
            y = 1 / (1 + np.exp(-x))
        except TypeError:
            y = 1 / (1 + np.exp(-x.astype(float)))
        except RuntimeWarning:
            return 1
        return float(y)

    @staticmethod
    def derivative(x: float) -> float:
        return x * (1 - x)
