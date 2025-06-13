"""Learning rate scheduler base class and implementations for step decay and exponential decay."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LearningRateScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    def __init__(self, decay_rate: float = 0.5, decay_steps: int = 10) -> None:
        """Initialize learning rate scheduler.

        :param float decay_rate:
            Rate at which the learning rate decays.
        :param int decay_steps:
            Number of steps after which the learning rate will be reduced.
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    @abstractmethod
    def step(self, lr: float, epoch: int) -> float:
        """Return the learning rate for the current epoch."""


class StepDecayScheduler(LearningRateScheduler):
    """Step decay learning rate scheduler."""

    def __init__(self, decay_rate: float = 0.5, decay_steps: int = 10) -> None:
        """Initialize step decay learning rate scheduler.

        :param float decay_rate:
            Rate at which the learning rate decays.
        :param int decay_steps:
            Number of steps after which the learning rate will be reduced.
        """
        self.decay_rate = decay_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, lr: float, epoch: int) -> float:
        """Step decay learning rate scheduler.

        :param float epoch:
            Current training epoch.

        :return float:
            Learning rate for current epoch.
        """
        return float(lr * np.power(self.decay_rate, np.floor((1 + epoch) / self.decay_steps)))


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""

    def __init__(self, decay_rate: float = 0.96, decay_steps: int = 100) -> None:
        """Initialize exponential decay learning rate scheduler.

        :param float decay_rate:
            Rate at which the learning rate decays.
        :param int decay_steps:
            Number of steps after which the learning rate will be decayed.
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, lr: float, epoch: int) -> float:
        """Exponential decay learning rate scheduler.

        :param float epoch:
            Current training epoch.

        :return float:
            Learning rate for current epoch.
        """
        return float(lr * np.power(self.decay_rate, epoch / self.decay_steps))
