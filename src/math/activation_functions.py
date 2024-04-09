import numpy as np


class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return max(x, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
