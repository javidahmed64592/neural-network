"""Activation function classes for neural networks."""

from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def func(self, x: float) -> float:
        """Compute the activation function.

        :param float x:
            Input value.
        :return float:
            Activated value.
        """

    @abstractmethod
    def derivative(self, x: float) -> float:
        """Compute the derivative of the activation function.

        :param float x:
            Input value.
        :return float:
            Derivative value.
        """


class LinearActivation(ActivationFunction):
    """Linear activation function."""

    def __str__(self) -> str:
        """Return string representation."""
        return "LinearActivation"

    @staticmethod
    def func(x: float) -> float:
        """Compute the linear activation.

        :param float x:
            Input value.
        :return float:
            Output value (same as input).
        """
        return x

    @staticmethod
    def derivative(x: float) -> float:
        """Compute the derivative of the linear activation.

        :param float x:
            Input value.
        :return float:
            Derivative value (always 1).
        """
        return 1


class ReluActivation(ActivationFunction):
    """ReLU activation function."""

    def __str__(self) -> str:
        """Return string representation."""
        return "ReluActivation"

    @staticmethod
    def func(x: float) -> float:
        """Compute the ReLU activation.

        :param float x:
            Input value.
        :return float:
            Output value (max(x, 0)).
        """
        return max(x, 0)

    @staticmethod
    def derivative(x: float) -> float:
        """Compute the derivative of the ReLU activation.

        :param float x:
            Input value.
        :return float:
            Derivative value (1 if x > 0 else 0).
        """
        return [0, 1][x > 0]


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function."""

    def __str__(self) -> str:
        """Return string representation."""
        return "SigmoidActivation"

    @staticmethod
    def func(x: float) -> float:
        """Compute the sigmoid activation.

        :param float x:
            Input value.
        :return float:
            Output value (sigmoid(x)).
        """
        return float(1 / (1 + np.exp(-x)))

    @staticmethod
    def derivative(x: float) -> float:
        """Compute the derivative of the sigmoid activation.

        :param float x:
            Input value (should be sigmoid(x) for correct derivative).
        :return float:
            Derivative value.
        """
        return x * (1 - x)


class TanhActivation(ActivationFunction):
    """Tanh activation function."""

    def __str__(self) -> str:
        """Return string representation."""
        return "TanhActivation"

    @staticmethod
    def func(x: float) -> float:
        """Compute the tanh activation.

        :param float x:
            Input value.
        :return float:
            Output value (tanh(x)).
        """
        return float(np.tanh(x))

    @staticmethod
    def derivative(x: float) -> float:
        """Compute the derivative of the tanh activation.

        :param float x:
            Input value.
        :return float:
            Derivative value.
        """
        t = np.tanh(x)
        return float(1 - t * t)
