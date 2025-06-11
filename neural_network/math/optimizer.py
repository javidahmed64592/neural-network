"""Optimization algorithms for neural network training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from neural_network.math.matrix import Matrix


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""

    def __init__(self, learning_rate: float = 0.001) -> None:
        """Initialize the optimizer with a learning rate.

        :param float learning_rate:
            The learning rate for parameter updates.
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(self, weights: Matrix, gradients: Matrix) -> Matrix:
        """Update weights using the optimization algorithm.

        :param Matrix weights:
            Current weights.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated weights.
        """

    @abstractmethod
    def update_bias(self, bias: Matrix, gradients: Matrix) -> Matrix:
        """Update bias using the optimization algorithm.

        :param Matrix bias:
            Current bias.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated bias.
        """

    @abstractmethod
    def step(self) -> None:
        """Increment the optimizer's internal state (e.g., time step for Adam)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state (e.g., momentum terms)."""


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        """Initialize SGD optimizer.

        :param float learning_rate:
            The learning rate for parameter updates.
        """
        super().__init__(learning_rate)

    def update_weights(self, weights: Matrix, gradients: Matrix) -> Matrix:
        """Update weights using SGD.

        :param Matrix weights:
            Current weights.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated weights.
        """
        return weights + gradients * self.learning_rate

    def update_bias(self, bias: Matrix, gradients: Matrix) -> Matrix:
        """Update bias using SGD.

        :param Matrix bias:
            Current bias.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated bias.
        """
        return bias + gradients * self.learning_rate

    def step(self) -> None:
        """Increment the optimizer's internal state (no state in SGD)."""

    def reset(self) -> None:
        """Reset optimizer state (no state in SGD)."""


class AdamOptimizer(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines the advantages of two extensions of stochastic gradient descent:
    - AdaGrad: adapts the learning rate to parameters
    - RMSProp: uses a moving average of squared gradients
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Adam optimizer.

        :param float learning_rate:
            The step size for parameter updates.
        :param float beta1:
            Exponential decay rate for first moment estimates (momentum).
        :param float beta2:
            Exponential decay rate for second moment estimates (RMSProp).
        :param float epsilon:
            Small constant to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State variables (one per optimizer instance)
        self._weight_m: Matrix | None = None  # First moment estimate for weights
        self._weight_v: Matrix | None = None  # Second moment estimate for weights
        self._bias_m: Matrix | None = None  # First moment estimate for bias
        self._bias_v: Matrix | None = None  # Second moment estimate for bias
        self._t: int = 1  # Time step

    def update_weights(self, weights: Matrix, gradients: Matrix) -> Matrix:
        """Update weights using Adam optimization.

        :param Matrix weights:
            Current weights.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated weights.
        """
        # Initialize moments if not present
        if self._weight_m is None:
            self._weight_m = Matrix.zeros(weights.rows, weights.cols)
            self._weight_v = Matrix.zeros(weights.rows, weights.cols)

        # Update biased first moment estimate
        self._weight_m = self._weight_m * self.beta1 + gradients * (1 - self.beta1)

        # Update biased second raw moment estimate
        gradients_squared = Matrix.from_array(np.square(gradients.vals))
        self._weight_v = self._weight_v * self.beta2 + gradients_squared * (1 - self.beta2)

        # Compute bias-corrected first moment estimate
        m_hat = self._weight_m * (1 / (1 - self.beta1**self._t))

        # Compute bias-corrected second raw moment estimate
        v_hat = self._weight_v * (1 / (1 - self.beta2**self._t))

        # Update parameters
        v_hat_sqrt = Matrix.from_array(np.sqrt(v_hat.vals))
        denominator = v_hat_sqrt + Matrix.filled(v_hat_sqrt.rows, v_hat_sqrt.cols, self.epsilon)
        update = m_hat / denominator * self.learning_rate

        return weights + update

    def update_bias(self, bias: Matrix, gradients: Matrix) -> Matrix:
        """Update bias using Adam optimization.

        :param Matrix bias:
            Current bias.
        :param Matrix gradients:
            Computed gradients.
        :return Matrix:
            Updated bias.
        """
        # Initialize moments if not present
        if self._bias_m is None:
            self._bias_m = Matrix.zeros(bias.rows, bias.cols)
            self._bias_v = Matrix.zeros(bias.rows, bias.cols)

        # Update biased first moment estimate
        self._bias_m = self._bias_m * self.beta1 + gradients * (1 - self.beta1)

        # Update biased second raw moment estimate
        gradients_squared = Matrix.from_array(np.square(gradients.vals))
        self._bias_v = self._bias_v * self.beta2 + gradients_squared * (1 - self.beta2)

        # Compute bias-corrected first moment estimate
        m_hat = self._bias_m * (1 / (1 - self.beta1**self._t))

        # Compute bias-corrected second raw moment estimate
        v_hat = self._bias_v * (1 / (1 - self.beta2**self._t))

        # Update parameters
        v_hat_sqrt = Matrix.from_array(np.sqrt(v_hat.vals))
        denominator = v_hat_sqrt + Matrix.filled(v_hat_sqrt.rows, v_hat_sqrt.cols, self.epsilon)
        update = m_hat / denominator * self.learning_rate

        return bias + update

    def step(self) -> None:
        """Increment the time step for bias correction."""
        self._t += 1

    def reset(self) -> None:
        """Reset optimizer state."""
        self._weight_m = None
        self._weight_v = None
        self._bias_m = None
        self._bias_v = None
        self._t = 0
