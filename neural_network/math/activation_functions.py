import numpy as np


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
    def __str__(self) -> str:
        return "LinearActivation"

    @staticmethod
    def func(x: float) -> float:
        return x

    @staticmethod
    def derivative(x: float) -> float:
        return 1


class ReluActivation(ActivationFunction):
    def __str__(self) -> str:
        return "ReluActivation"

    @staticmethod
    def func(x: float) -> float:
        return max(x, 0)

    @staticmethod
    def derivative(x: float) -> float:
        return [0, 1][x > 0]


class SigmoidActivation(ActivationFunction):
    def __str__(self) -> str:
        return "SigmoidActivation"

    @staticmethod
    def func(x: float) -> float:
        try:
            y = 1 / (1 + np.exp(-x))
        except RuntimeWarning:
            return 1.0
        return y

    @staticmethod
    def derivative(x: float) -> float:
        return x * (1 - x)
