from collections.abc import Callable

import pytest

from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork
from neural_network.nn.layer import Layer
from neural_network.nn.node import Node


@pytest.fixture
def mock_activation() -> Callable:
    return lambda x: x * 3


@pytest.fixture
def mock_weights_range() -> tuple[float, float]:
    return [-1.0, 1.0]


@pytest.fixture
def mock_bias_range() -> tuple[float, float]:
    return [-1.0, 1.0]


@pytest.fixture
def mock_inputs() -> list[float]:
    return [1.0, 0.5, 0.7]


@pytest.fixture
def mock_len_inputs(mock_inputs: list[float]) -> int:
    return len(mock_inputs)


@pytest.fixture
def mock_len_hidden() -> int:
    return 5


@pytest.fixture
def mock_outputs() -> list[float]:
    return [0.0, 1.0]


@pytest.fixture
def mock_len_outputs(mock_outputs: list[float]) -> int:
    return len(mock_outputs)


@pytest.fixture
def mock_nn(mock_len_inputs: int, mock_len_outputs: int) -> NeuralNetwork:
    return NeuralNetwork(mock_len_inputs, mock_len_outputs, [5])


@pytest.fixture
def mock_hidden_layer(
    mock_len_hidden: int,
    mock_len_inputs: int,
    mock_activation: Callable,
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
) -> Layer:
    return Layer(mock_len_hidden, mock_len_inputs, mock_activation, mock_weights_range, mock_bias_range)


@pytest.fixture
def mock_node(
    mock_len_inputs: int, mock_weights_range: list[float], mock_bias_range: list[float], mock_activation: Callable
) -> Node:
    return Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation)


@pytest.fixture
def mock_input_matrix(mock_inputs: int) -> Matrix:
    return Matrix.from_array(mock_inputs)
