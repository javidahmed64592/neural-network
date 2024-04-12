import pytest

from neural_network.math.matrix import Matrix
from neural_network.nn.layer import Layer
from neural_network.nn.neural_network import NeuralNetwork
from neural_network.nn.node import Node


@pytest.fixture
def mock_activation():
    return lambda x: x * 3


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
def mock_layer(mock_len_hidden, mock_len_inputs, mock_activation):
    return Layer(mock_len_hidden, mock_len_inputs, mock_activation)


@pytest.fixture
def mock_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation):
    return Node.random_node(mock_len_inputs, mock_weights_range, mock_bias_range, mock_activation)


@pytest.fixture
def mock_input_matrix(mock_inputs):
    return Matrix.from_array(mock_inputs)
