import numpy as np


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
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))
