"""Optimization algorithms for neural network training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from neural_network.math.learning_rate_scheduler import LearningRateScheduler, StepDecayScheduler
from neural_network.math.matrix import Matrix


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""

    def __init__(self, lr: float, lr_scheduler: LearningRateScheduler | None = None) -> None:
        """Initialize the optimizer with a learning rate scheduler.

        :param float lr:
            Initial learning rate for the optimizer.
        :param LearningRateScheduler | None lr_scheduler:
            The learning rate scheduler for parameter updates.
        """
        self.lr = lr
        self.lr_scheduler = lr_scheduler or StepDecayScheduler()
        self._t = 1

    @property
    def learning_rate(self) -> float:
        """Get the current learning rate from the scheduler.

        :return float:
            The current learning rate.
        """
        return self.lr_scheduler.step(self.lr, self._t)

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

    def step(self) -> None:
        """Increment the time step."""
        self._t += 1

    def reset(self) -> None:
        """Reset optimizer state."""
        self._t = 1


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, lr: float = 0.1, lr_scheduler: LearningRateScheduler | None = None) -> None:
        """Initialize SGD optimizer.

        :param float lr:
            Initial learning rate for the optimizer.
        :param LearningRateScheduler | None lr_scheduler:
            The learning rate scheduler for parameter updates.
        """
        super().__init__(lr, lr_scheduler)

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


class AdamOptimizer(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines the advantages of two extensions of stochastic gradient descent:
    - AdaGrad: adapts the learning rate to parameters
    - RMSProp: uses a moving average of squared gradients
    """

    def __init__(
        self,
        lr: float = 0.001,
        lr_scheduler: LearningRateScheduler | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Adam optimizer.

        :param float lr:
            Initial learning rate for the optimizer.
        :param LearningRateScheduler | None lr_scheduler:
            The learning rate scheduler for parameter updates.
        :param float beta1:
            Exponential decay rate for first moment estimates (momentum).
        :param float beta2:
            Exponential decay rate for second moment estimates (RMSProp).
        :param float epsilon:
            Small constant to prevent division by zero.
        """
        super().__init__(lr, lr_scheduler)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # State variables (one per optimizer instance)
        self._weight_m: Matrix | None = None  # First moment estimate for weights
        self._weight_v: Matrix | None = None  # Second moment estimate for weights
        self._bias_m: Matrix | None = None  # First moment estimate for bias
        self._bias_v: Matrix | None = None  # Second moment estimate for bias

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
        if self._weight_m is None or self._weight_v is None:
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
        if self._bias_m is None or self._bias_v is None:
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

    def reset(self) -> None:
        """Reset optimizer state."""
        super().reset()
        self._weight_m = None
        self._weight_v = None
        self._bias_m = None
        self._bias_v = None
