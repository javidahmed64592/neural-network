"""Unit tests for the neural_network.math.activation_functions module."""

import numpy as np

from neural_network.math.activation_functions import (
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)


class TestLinearActivation:
    """Test cases for the LinearActivation class."""

    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        """Test that the linear activation function returns the input value."""
        x = 5
        expected_y = x
        actual_y = LinearActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        """Test that the linear activation function derivative returns 1."""
        x = 5
        expected_y = 1
        actual_y = LinearActivation.derivative(x)
        assert actual_y == expected_y


class TestReluActivation:
    """Test cases for the ReluActivation class."""

    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        """Test that the ReLU activation function returns correct values for positive and negative inputs."""
        x = -5
        expected_y = 0
        actual_y = ReluActivation.func(x)
        assert actual_y == expected_y

        x = 5
        expected_y = x
        actual_y = ReluActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        """Test that the ReLU activation function derivative returns correct values for positive and negative inputs."""
        x = -5
        expected_y = 0
        actual_y = ReluActivation.derivative(x)
        assert actual_y == expected_y

        x = 5
        expected_y = 1
        actual_y = ReluActivation.derivative(x)
        assert actual_y == expected_y


class TestSigmoidActivation:
    """Test cases for the SigmoidActivation class."""

    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        """Test that the sigmoid activation function returns the correct value."""
        x = 5
        expected_y = 1 / (1 + np.exp(-x))
        actual_y = SigmoidActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        """Test that the sigmoid activation function derivative returns the correct value."""
        x = 5
        expected_y = x * (1 - x)
        actual_y = SigmoidActivation.derivative(x)
        assert actual_y == expected_y


class TestTanhActivation:
    """Test cases for the TanhActivation class."""

    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        """Test that the tanh activation function returns the correct value."""
        x = 1.5
        expected_y = np.tanh(x)
        actual_y = TanhActivation.func(x)
        assert np.isclose(actual_y, expected_y)

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        """Test that the tanh activation function derivative returns the correct value."""
        x = 1.5
        t = np.tanh(x)
        expected_y = 1 - t * t
        actual_y = TanhActivation.derivative(x)
        assert np.isclose(actual_y, expected_y)
