# mypy: disable-error-code=unreachable
"""Unit tests for the neural_network.math.optimizer module."""

import numpy as np
import pytest

from neural_network.math.learning_rate_scheduler import StepDecayScheduler
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import AdamOptimizer, SGDOptimizer


class TestSGDOptimizer:
    """Test cases for SGD optimizer."""

    @pytest.fixture
    def sgd_optimizer(self) -> SGDOptimizer:
        """Fixture to create an instance of SGDOptimizer."""
        return SGDOptimizer(lr_scheduler=StepDecayScheduler())

    def test_given_weights_when_updating_then_check_weights_change_correctly(self, sgd_optimizer: SGDOptimizer) -> None:
        """Test SGD optimizer updates weights correctly."""
        weights = Matrix.from_array([[1.0, 2.0], [3.0, 4.0]])
        gradients = Matrix.from_array([[0.1, 0.2], [0.3, 0.4]])

        updated_weights = sgd_optimizer.update_weights(weights, gradients)

        expected = weights + (gradients * sgd_optimizer.learning_rate)
        assert np.allclose(updated_weights.vals, expected.vals)

    def test_given_bias_when_updating_then_check_bias_change_correctly(self, sgd_optimizer: SGDOptimizer) -> None:
        """Test SGD optimizer updates bias correctly."""
        bias = Matrix.from_array([[1.0], [2.0]])
        gradients = Matrix.from_array([[0.1], [0.2]])

        updated_bias = sgd_optimizer.update_bias(bias, gradients)

        expected = bias + (gradients * sgd_optimizer.learning_rate)
        assert np.allclose(updated_bias.vals, expected.vals)

    def test_given_optimizer_when_stepping_then_check_no_error_raised(self, sgd_optimizer: SGDOptimizer) -> None:
        """Test SGD optimizer step method increments timestep."""
        sgd_optimizer.step()

    def test_given_optimizer_when_resetting_then_check_no_error_raised(self, sgd_optimizer: SGDOptimizer) -> None:
        """Test SGD optimizer reset method (no state to reset)."""
        sgd_optimizer.reset()


class TestAdamOptimizer:
    """Test cases for Adam optimizer."""

    @pytest.fixture
    def adam_optimizer(self) -> AdamOptimizer:
        """Fixture to create an instance of AdamOptimizer."""
        return AdamOptimizer(lr_scheduler=StepDecayScheduler(), beta1=0.9, beta2=0.999, epsilon=1e-8)

    def test_given_adam_optimizer_when_first_weight_update_then_moments_initialized(
        self, adam_optimizer: AdamOptimizer
    ) -> None:
        """Test Adam optimizer initializes moment estimates on first update."""
        weights = Matrix.from_array([[1.0, 2.0], [3.0, 4.0]])
        gradients = Matrix.from_array([[0.1, 0.2], [0.3, 0.4]])  # Before update, no moments exist
        assert adam_optimizer._weight_m is None
        assert adam_optimizer._weight_v is None

        # Set timestep for bias correction
        adam_optimizer._t = 1

        updated_weights = adam_optimizer.update_weights(weights, gradients)
        assert isinstance(updated_weights, Matrix)
        assert updated_weights.shape == weights.shape

        # After update, moments should be initialized
        assert isinstance(adam_optimizer._weight_m, Matrix)
        assert adam_optimizer._weight_m.shape == weights.shape
        assert isinstance(adam_optimizer._weight_v, Matrix)
        assert adam_optimizer._weight_v.shape == weights.shape

    def test_given_adam_optimizer_when_multiple_weight_updates_then_moments_accumulate(
        self, adam_optimizer: AdamOptimizer
    ) -> None:
        """Test Adam optimizer accumulates moment estimates over multiple updates."""
        weights = Matrix.from_array([[1.0, 2.0]])
        gradients1 = Matrix.from_array([[0.1, 0.2]])
        gradients2 = Matrix.from_array([[0.3, 0.4]])  # First update
        weights1 = adam_optimizer.update_weights(weights, gradients1)
        assert adam_optimizer._weight_m is not None
        assert adam_optimizer._weight_v is not None
        m1 = adam_optimizer._weight_m.vals.copy()
        v1 = adam_optimizer._weight_v.vals.copy()
        adam_optimizer.step()

        # Second update
        weights2 = adam_optimizer.update_weights(weights1, gradients2)
        assert adam_optimizer._weight_m is not None
        assert adam_optimizer._weight_v is not None
        m2 = adam_optimizer._weight_m.vals.copy()
        v2 = adam_optimizer._weight_v.vals.copy()
        adam_optimizer.step()

        # Moments should have changed
        assert not np.allclose(m1, m2)
        assert not np.allclose(v1, v2)

        # Weights should have changed
        assert not np.allclose(weights.vals, weights1.vals)
        assert not np.allclose(weights1.vals, weights2.vals)

    def test_given_adam_optimizer_when_updating_bias_then_bias_moments_initialized(
        self, adam_optimizer: AdamOptimizer
    ) -> None:
        """Test Adam optimizer initializes bias moment estimates on first update."""
        bias = Matrix.from_array([[1.0], [2.0]])
        gradients = Matrix.from_array([[0.1], [0.2]])  # Before update, no bias moments exist
        assert adam_optimizer._bias_m is None
        assert adam_optimizer._bias_v is None

        updated_bias = adam_optimizer.update_bias(bias, gradients)
        assert isinstance(updated_bias, Matrix)
        assert updated_bias.shape == bias.shape

        # After update, bias moments should be initialized
        assert isinstance(adam_optimizer._bias_m, Matrix)
        assert adam_optimizer._bias_m.shape == bias.shape
        assert isinstance(adam_optimizer._bias_v, Matrix)
        assert adam_optimizer._bias_v.shape == bias.shape

    def test_given_adam_optimizer_when_step_called_then_timestep_increments(
        self, adam_optimizer: AdamOptimizer
    ) -> None:
        """Test Adam optimizer step method increments timestep."""
        expected_timestep = 1
        assert adam_optimizer._t == expected_timestep

        adam_optimizer.step()
        expected_timestep += 1
        assert adam_optimizer._t == expected_timestep

    def test_given_adam_optimizer_when_reset_then_state_cleared(self, adam_optimizer: AdamOptimizer) -> None:
        """Test Adam optimizer reset method clears all state."""
        weights = Matrix.from_array([[1.0, 2.0]])
        gradients = Matrix.from_array([[0.1, 0.2]])
        bias = Matrix.from_array([[1.0]])
        bias_gradients = Matrix.from_array([[0.1]])  # Set up some state
        adam_optimizer.update_weights(weights, gradients)
        adam_optimizer.update_bias(bias, bias_gradients)
        adam_optimizer.step()

        # Verify state exists
        assert adam_optimizer._t == 1 + 1
        assert adam_optimizer._weight_m is not None
        assert adam_optimizer._weight_v is not None
        assert adam_optimizer._bias_m is not None
        assert adam_optimizer._bias_v is not None

        # Reset
        adam_optimizer.reset()

        # Verify state is cleared
        assert adam_optimizer._t == 1
        assert adam_optimizer._weight_m is None
        assert adam_optimizer._weight_v is None
        assert adam_optimizer._bias_m is None
        assert adam_optimizer._bias_v is None
