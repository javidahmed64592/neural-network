"""Unit tests for the neural_network.math.learning_rate_scheduler module."""

import numpy as np
import pytest

from neural_network.math.learning_rate_scheduler import ExponentialDecayScheduler, StepDecayScheduler


class TestStepDecayScheduler:
    """Test cases for the StepDecayScheduler class."""

    @pytest.mark.parametrize(
        ("initial_lr", "decay_rate", "decay_steps", "epoch", "expected_lr"),
        [
            (0.1, 0.5, 10, 0, 0.1),
            (0.1, 0.5, 10, 5, 0.1),
            (0.1, 0.5, 10, 10, 0.05),
        ],
    )
    def test_step_decay_scheduler(
        self, initial_lr: float, decay_rate: float, decay_steps: int, epoch: int, expected_lr: float
    ) -> None:
        """Test the step decay learning rate scheduler."""
        scheduler = StepDecayScheduler(initial_lr, decay_rate, decay_steps)
        lr = scheduler.step(epoch)
        assert np.isclose(lr, expected_lr, rtol=1e-4)


class TestExponentialDecayScheduler:
    """Test cases for the ExponentialDecayScheduler class."""

    @pytest.mark.parametrize(
        ("initial_lr", "decay_rate", "decay_steps", "epoch", "expected_lr"),
        [
            (0.1, 0.96, 100, 0, 0.1),
            (0.1, 0.96, 100, 50, 0.1 * np.power(0.96, 50 / 100)),
            (0.1, 0.96, 100, 100, 0.1 * np.power(0.96, 1)),
        ],
    )
    def test_exponential_decay_scheduler(
        self, initial_lr: float, decay_rate: float, decay_steps: int, epoch: int, expected_lr: float
    ) -> None:
        """Test the exponential decay learning rate scheduler."""
        scheduler = ExponentialDecayScheduler(initial_lr, decay_rate, decay_steps)
        lr = scheduler.step(epoch)
        assert np.isclose(lr, expected_lr, rtol=1e-4)
