"""Pytest fixtures for neural network unit tests."""

import pytest

from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.activation_functions import LinearActivation
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork


@pytest.fixture
def mock_activation() -> type[LinearActivation]:
    """Fixture for a mock activation function class."""
    return LinearActivation


@pytest.fixture
def mock_weights_range() -> tuple[float, float]:
    """Fixture for a mock weights range."""
    return (-0.1, 0.1)


@pytest.fixture
def mock_bias_range() -> tuple[float, float]:
    """Fixture for a mock bias range."""
    return (-0.1, 0.1)


@pytest.fixture
def mock_inputs() -> list[float]:
    """Fixture for mock input values."""
    return [1.0, 0.5, 0.7]


@pytest.fixture
def mock_len_inputs(mock_inputs: list[float]) -> int:
    """Fixture for the number of mock inputs."""
    return len(mock_inputs)


@pytest.fixture
def mock_len_hidden() -> list[int]:
    """Fixture for mock hidden layer sizes."""
    return [5, 4, 3]


@pytest.fixture
def mock_outputs() -> list[float]:
    """Fixture for mock output values."""
    return [0.0, 1.0]


@pytest.fixture
def mock_len_outputs(mock_outputs: list[float]) -> int:
    """Fixture for the number of mock outputs."""
    return len(mock_outputs)


@pytest.fixture
def mock_input_layer(mock_len_inputs: int, mock_activation: type[LinearActivation]) -> InputLayer:
    """Fixture for a mock input layer."""
    return InputLayer(mock_len_inputs, mock_activation)


@pytest.fixture
def mock_hidden_layer_1(
    mock_len_hidden: list[int],
    mock_activation: type[LinearActivation],
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_input_layer: InputLayer,
) -> HiddenLayer:
    """Fixture for the first mock hidden layer."""
    mock_layer = HiddenLayer(mock_len_hidden[0], mock_activation, mock_weights_range, mock_bias_range)
    mock_layer.set_prev_layer(mock_input_layer)
    return mock_layer


@pytest.fixture
def mock_hidden_layer_2(
    mock_len_hidden: list[int],
    mock_activation: type[LinearActivation],
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_1: HiddenLayer,
) -> HiddenLayer:
    """Fixture for the second mock hidden layer."""
    mock_layer = HiddenLayer(mock_len_hidden[1], mock_activation, mock_weights_range, mock_bias_range)
    mock_layer.set_prev_layer(mock_hidden_layer_1)
    return mock_layer


@pytest.fixture
def mock_hidden_layer_3(
    mock_len_hidden: list[int],
    mock_activation: type[LinearActivation],
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_2: HiddenLayer,
) -> HiddenLayer:
    """Fixture for the third mock hidden layer."""
    mock_layer = HiddenLayer(mock_len_hidden[2], mock_activation, mock_weights_range, mock_bias_range)
    mock_layer.set_prev_layer(mock_hidden_layer_2)
    return mock_layer


@pytest.fixture
def mock_output_layer(
    mock_len_outputs: int,
    mock_activation: type[LinearActivation],
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_3: HiddenLayer,
) -> OutputLayer:
    """Fixture for a mock output layer."""
    mock_layer = OutputLayer(mock_len_outputs, mock_activation, mock_weights_range, mock_bias_range)
    mock_layer.set_prev_layer(mock_hidden_layer_3)
    return mock_layer


@pytest.fixture
def mock_input_matrix(mock_inputs: list[float]) -> Matrix:
    """Fixture for a mock input matrix."""
    return Matrix.from_array(mock_inputs)


@pytest.fixture
def mock_nn(
    mock_input_layer: InputLayer,
    mock_hidden_layer_1: HiddenLayer,
    mock_hidden_layer_2: HiddenLayer,
    mock_hidden_layer_3: HiddenLayer,
    mock_output_layer: OutputLayer,
) -> NeuralNetwork:
    """Fixture for a mock neural network."""
    return NeuralNetwork.from_layers(
        layers=[mock_input_layer, mock_hidden_layer_1, mock_hidden_layer_2, mock_hidden_layer_3, mock_output_layer]
    )


@pytest.fixture
def mock_training_inputs() -> list[list[float]]:
    """Fixture for mock training input data."""
    return [[1.0, 0.5, 0.7], [0.8, 0.3, 0.9], [0.2, 0.6, 0.4]]


@pytest.fixture
def mock_training_outputs() -> list[list[float]]:
    """Fixture for mock training output data."""
    return [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]


@pytest.fixture
def mock_fitnesses() -> list[float]:
    """Fixture for mock fitness values."""
    return [0.8, 0.7, 0.9]
