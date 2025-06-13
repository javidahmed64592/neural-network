"""Learning rate scheduler base class and implementations for step decay and exponential decay."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LearningRateScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    @abstractmethod
    def step(self, epoch: int) -> float:
        """Return the learning rate for the current epoch."""


class StepDecayScheduler(LearningRateScheduler):
    """Step decay learning rate scheduler."""

    def __init__(self, initial_lr: float = 0.1, drop_factor: float = 0.5, epochs_drop: int = 10) -> None:
        """Initialize step decay learning rate scheduler.

        :param float initial_lr:
            Initial learning rate.
        :param float drop_factor:
            Factor by which the learning rate will be reduced.
        :param int epochs_drop:
            Number of epochs after which the learning rate will be reduced.
        """
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.epochs_drop = epochs_drop

    def step(self, epoch: int) -> float:
        """Step decay learning rate scheduler.

        :param float epoch:
            Current training epoch.

        :return float:
            Learning rate for current epoch.
        """
        return float(self.initial_lr * np.power(self.drop_factor, np.floor((1 + epoch) / self.epochs_drop)))


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""

    def __init__(self, initial_lr: float = 0.1, decay_rate: float = 0.96, decay_steps: int = 100) -> None:
        """Initialize exponential decay learning rate scheduler.

        :param float initial_lr:
            Initial learning rate.
        :param float decay_rate:
            Rate at which the learning rate decays.
        :param int decay_steps:
            Number of steps after which the learning rate will be decayed.
        """
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, epoch: int) -> float:
        """Exponential decay learning rate scheduler.

        :param float epoch:
            Current training epoch.

        :return float:
            Learning rate for current epoch.
        """
        return float(self.initial_lr * np.power(self.decay_rate, epoch / self.decay_steps))
