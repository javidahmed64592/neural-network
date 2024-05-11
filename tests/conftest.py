import pytest

from neural_network.math.activation_functions import LinearActivation
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork
from neural_network.nn.layer import HiddenLayer, InputLayer, OutputLayer


@pytest.fixture
def mock_activation() -> LinearActivation:
    return LinearActivation


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
def mock_len_hidden() -> list[int]:
    return [5, 4, 3]


@pytest.fixture
def mock_outputs() -> list[float]:
    return [0.0, 1.0]


@pytest.fixture
def mock_len_outputs(mock_outputs: list[float]) -> int:
    return len(mock_outputs)


@pytest.fixture
def mock_layer_sizes(mock_len_inputs: int, mock_len_hidden: list[int], mock_len_outputs: int) -> list[int]:
    return [mock_len_inputs, *mock_len_hidden, mock_len_outputs]


@pytest.fixture
def mock_nn(mock_len_inputs: int, mock_len_hidden: list[int], mock_len_outputs: int) -> NeuralNetwork:
    return NeuralNetwork(mock_len_inputs, mock_len_outputs, mock_len_hidden)


@pytest.fixture
def mock_input_layer(mock_len_inputs: int, mock_activation: LinearActivation) -> InputLayer:
    return InputLayer(mock_len_inputs, mock_activation)


@pytest.fixture
def mock_hidden_layer_1(
    mock_len_hidden: list[int],
    mock_activation: LinearActivation,
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_input_layer: InputLayer,
) -> HiddenLayer:
    return HiddenLayer(mock_len_hidden[0], mock_activation, mock_weights_range, mock_bias_range, mock_input_layer)


@pytest.fixture
def mock_hidden_layer_2(
    mock_len_hidden: list[int],
    mock_activation: LinearActivation,
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_1: HiddenLayer,
) -> HiddenLayer:
    return HiddenLayer(mock_len_hidden[1], mock_activation, mock_weights_range, mock_bias_range, mock_hidden_layer_1)


@pytest.fixture
def mock_hidden_layer_3(
    mock_len_hidden: list[int],
    mock_activation: LinearActivation,
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_2: HiddenLayer,
) -> HiddenLayer:
    return HiddenLayer(mock_len_hidden[2], mock_activation, mock_weights_range, mock_bias_range, mock_hidden_layer_2)


@pytest.fixture
def mock_output_layer(
    mock_len_outputs: int,
    mock_activation: LinearActivation,
    mock_weights_range: tuple[float, float],
    mock_bias_range: tuple[float, float],
    mock_hidden_layer_3: HiddenLayer,
) -> OutputLayer:
    return OutputLayer(mock_len_outputs, mock_activation, mock_weights_range, mock_bias_range, mock_hidden_layer_3)


@pytest.fixture
def mock_input_matrix(mock_inputs: int) -> Matrix:
    return Matrix.from_array(mock_inputs)
