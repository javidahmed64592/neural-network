import pytest

from src.math.matrix import Matrix
from src.nn.layer import Layer
from src.nn.neural_network import NeuralNetwork
from src.nn.node import Node


def activation(x):
    return x


@pytest.fixture
def mock_weights_range():
    return [-1.0, 1.0]


@pytest.fixture
def mock_bias_range():
    return [-1.0, 1.0]


@pytest.fixture
def mock_inputs():
    return [1.0, 0.5, 0.7]


@pytest.fixture
def mock_len_inputs(mock_inputs):
    return len(mock_inputs)


@pytest.fixture
def mock_len_hidden():
    return 5


@pytest.fixture
def mock_outputs():
    return [0.0, 1.0]


@pytest.fixture
def mock_len_outputs(mock_outputs):
    return len(mock_outputs)


@pytest.fixture
def mock_nn(mock_len_inputs, mock_len_outputs):
    return NeuralNetwork(mock_len_inputs, 5, mock_len_outputs)


@pytest.fixture
def mock_layer(mock_len_hidden, mock_len_inputs):
    return Layer(size=mock_len_hidden, num_inputs=mock_len_inputs, activation=activation)


@pytest.fixture
def mock_node(mock_len_inputs, mock_weights_range, mock_bias_range):
    return Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, activation)


@pytest.fixture
def mock_input_matrix(mock_inputs):
    return Matrix.from_array(mock_inputs)
