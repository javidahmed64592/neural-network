import numpy as np

from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)


class TestActivationFunction:
    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = x
        actual_y = ActivationFunction.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = 1
        actual_y = ActivationFunction.derivative(x)
        assert actual_y == expected_y


class TestLinearActivation:
    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = x
        actual_y = LinearActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = 1
        actual_y = LinearActivation.derivative(x)
        assert actual_y == expected_y


class TestReluActivation:
    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        x = -5
        expected_y = 0
        actual_y = ReluActivation.func(x)
        assert actual_y == expected_y

        x = 5
        expected_y = x
        actual_y = ReluActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        x = -5
        expected_y = 0
        actual_y = ReluActivation.derivative(x)
        assert actual_y == expected_y

        x = 5
        expected_y = 1
        actual_y = ReluActivation.derivative(x)
        assert actual_y == expected_y


class TestSigmoidActivation:
    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = 1 / (1 + np.exp(-x))
        actual_y = SigmoidActivation.func(x)
        assert actual_y == expected_y

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        x = 5
        expected_y = x * (1 - x)
        actual_y = SigmoidActivation.derivative(x)
        assert actual_y == expected_y


class TestTanhActivation:
    def test_given_x_when_calculating_y_then_check_calculated_correctly(self) -> None:
        x = 1.5
        expected_y = np.tanh(x)
        actual_y = TanhActivation.func(x)
        assert np.isclose(actual_y, expected_y)

    def test_given_x_when_calculating_derivative_then_check_calculated_correctly(self) -> None:
        x = 1.5
        t = np.tanh(x)
        expected_y = 1 - t * t
        actual_y = TanhActivation.derivative(x)
        assert np.isclose(actual_y, expected_y)
