"""Unit tests for the neural_network.neural_network module."""

from pathlib import Path
from unittest.mock import mock_open, patch

from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.activation_functions import ActivationFunction
from neural_network.neural_network import NeuralNetwork
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    NeuralNetworkDataType,
    OptimizerDataType,
)


def make_hidden_layer(
    size: int,
    activation: type[ActivationFunction],
    weights_range: tuple[float, float],
    bias_range: tuple[float, float],
) -> HiddenLayer:
    """Create a HiddenLayer for testing."""
    return HiddenLayer(size, activation, weights_range, bias_range)


class TestNeuralNetwork:
    """Test cases for the NeuralNetwork class."""

    def test_to_protobuf(self, mock_nn: NeuralNetwork) -> None:
        """Test converting a neural network to protobuf format."""
        nn_data = NeuralNetwork.to_protobuf(mock_nn)
        assert nn_data.num_inputs == mock_nn._num_inputs
        assert nn_data.hidden_layer_sizes == mock_nn._hidden_layer_sizes
        assert nn_data.num_outputs == mock_nn._num_outputs
        assert nn_data.input_activation == ActivationFunctionEnum.from_class(mock_nn._input_layer._activation)
        assert nn_data.hidden_activation == ActivationFunctionEnum.from_class(mock_nn._hidden_layers[0]._activation)
        assert nn_data.output_activation == ActivationFunctionEnum.from_class(mock_nn._output_layer._activation)
        assert len(nn_data.weights) == len(mock_nn.weights)
        assert len(nn_data.biases) == len(mock_nn.bias)
        assert nn_data.optimizer == OptimizerDataType.from_class_instance(mock_nn._input_layer._optimizer)

    def test_from_protobuf(self, mock_nn: NeuralNetwork) -> None:
        """Test creating a neural network from protobuf format."""
        nn_data = NeuralNetwork.to_protobuf(mock_nn)
        new_nn = NeuralNetwork.from_protobuf(nn_data)
        assert new_nn._num_inputs == mock_nn._num_inputs
        assert new_nn._hidden_layer_sizes == mock_nn._hidden_layer_sizes
        assert new_nn._num_outputs == mock_nn._num_outputs
        assert new_nn._input_layer._activation == mock_nn._input_layer._activation
        assert new_nn._hidden_layers[0]._activation == mock_nn._hidden_layers[0]._activation
        assert new_nn._output_layer._activation == mock_nn._output_layer._activation
        assert len(new_nn.weights) == len(mock_nn.weights)
        assert len(new_nn.bias) == len(mock_nn.bias)
        assert isinstance(new_nn._optimizer, mock_nn._optimizer.__class__)

    def test_load_from_file(self, mock_nn: NeuralNetwork) -> None:
        """Test loading a neural network from a file."""
        nn_data = NeuralNetwork.to_protobuf(mock_nn)
        with patch(
            "builtins.open", mock_open(read_data=NeuralNetworkDataType.to_protobuf(nn_data).SerializeToString())
        ):
            loaded_nn = NeuralNetwork.load_from_file("mock_file_path")
            assert loaded_nn._num_inputs == mock_nn._num_inputs
            assert loaded_nn._hidden_layer_sizes == mock_nn._hidden_layer_sizes
            assert loaded_nn._num_outputs == mock_nn._num_outputs
            assert loaded_nn._input_layer._activation == mock_nn._input_layer._activation
            assert loaded_nn._hidden_layers[0]._activation == mock_nn._hidden_layers[0]._activation
            assert loaded_nn._output_layer._activation == mock_nn._output_layer._activation
            assert len(loaded_nn.weights) == len(mock_nn.weights)
            assert len(loaded_nn.bias) == len(mock_nn.bias)
            assert isinstance(loaded_nn._optimizer, mock_nn._optimizer.__class__)

    def test_save_to_file(self, mock_nn: NeuralNetwork) -> None:
        """Test saving a neural network to a file."""
        nn_data = NeuralNetwork.to_protobuf(mock_nn)
        mock_folder = Path("mock_folder_path")
        mock_filename = "mock_nn"

        with patch("builtins.open", mock_open()) as mock_file:
            NeuralNetwork.save_to_file(mock_nn, mock_filename, mock_folder)
            mock_file.assert_called_once_with(mock_folder / f"{mock_filename}.pb", "wb")
            mock_file().write.assert_called_once_with(NeuralNetworkDataType.to_protobuf(nn_data).SerializeToString())

    def test_given_nn_when_creating_layers_then_check_nn_has_correct_layers(
        self, mock_nn: NeuralNetwork, mock_len_inputs: int, mock_len_hidden: list[int], mock_len_outputs: int
    ) -> None:
        """Test that the neural network has correct layers after creation."""
        expected_sizes = [mock_len_inputs, *mock_len_hidden, mock_len_outputs]
        for index, layer in enumerate(mock_nn.layers):
            assert layer.size == expected_sizes[index]

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_len_outputs: int
    ) -> None:
        """Test feedforward produces output of correct shape."""
        output = mock_nn.feedforward(mock_inputs)
        actual_len = len(output)
        assert actual_len == mock_len_outputs

    def test_given_inputs_and_outputs_when_training_then_check_error_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_outputs: list[float], mock_len_outputs: int
    ) -> None:
        """Test training produces error of correct shape."""
        output_errors = mock_nn.train(mock_inputs, mock_outputs)
        actual_len = len(output_errors)
        assert actual_len == mock_len_outputs

    def test_given_inputs_and_fitnesses_when_training_then_check_error_has_correct_shape(
        self, mock_nn: NeuralNetwork, mock_inputs: list[float], mock_len_outputs: int
    ) -> None:
        """Test training with fitness produces error of correct shape."""
        output_errors = mock_nn.train_with_fitness(mock_inputs, 1, 0.8)
        actual_len = len(output_errors)
        assert actual_len == mock_len_outputs

    def test_given_training_data_when_running_supervised_training_then_check_feedforward_maintains_shape(
        self,
        mock_nn: NeuralNetwork,
        mock_training_inputs: list[list[float]],
        mock_training_outputs: list[list[float]],
        mock_len_outputs: int,
    ) -> None:
        """Test supervised training updates weights and maintains output shape."""
        initial_weights = [layer.weights.vals.copy() for layer in mock_nn.layers[1:]]

        mock_nn.run_supervised_training(mock_training_inputs, mock_training_outputs, epochs=2)

        final_weights = [layer.weights.vals for layer in mock_nn.layers[1:]]
        for initial, final in zip(initial_weights, final_weights, strict=True):
            assert not (initial == final).all()

        output = mock_nn.feedforward(mock_training_inputs[0])
        assert len(output) == mock_len_outputs

    def test_given_fitness_data_when_running_fitness_training_then_check_feedforward_maintains_shape(
        self,
        mock_nn: NeuralNetwork,
        mock_training_inputs: list[list[float]],
        mock_fitnesses: list[float],
        mock_len_outputs: int,
    ) -> None:
        """Test fitness-based training updates weights and maintains output shape."""
        initial_weights = [layer.weights.vals.copy() for layer in mock_nn.layers[1:]]

        mock_nn.run_fitness_training(mock_training_inputs, mock_fitnesses, epochs=2, alpha=0.2)

        final_weights = [layer.weights.vals for layer in mock_nn.layers[1:]]
        for initial, final in zip(initial_weights, final_weights, strict=True):
            assert not (initial == final).all()

        output = mock_nn.feedforward(mock_training_inputs[0])
        assert len(output) == mock_len_outputs

    def test_given_two_nns_when_performing_crossover_then_check_feedforward_maintains_correct_shape(
        self,
        mock_inputs: list[float],
        mock_input_layer: InputLayer,
        mock_output_layer: OutputLayer,
        mock_len_outputs: int,
        mock_activation: type[ActivationFunction],
        mock_weights_range: tuple[float, float],
        mock_bias_range: tuple[float, float],
    ) -> None:
        """Test crossover between two neural networks maintains correct output shape."""
        mutation_rate = 0.05

        def _mock_crossover_func(element: float, other_element: float, roll: float) -> float:
            if roll < mutation_rate:
                return element
            return other_element

        nn_layers = [
            mock_input_layer,
            *[
                make_hidden_layer(5, mock_activation, mock_weights_range, mock_bias_range),
                make_hidden_layer(4, mock_activation, mock_weights_range, mock_bias_range),
                make_hidden_layer(3, mock_activation, mock_weights_range, mock_bias_range),
            ],
            mock_output_layer,
        ]

        mock_nn_1 = NeuralNetwork.from_layers(layers=nn_layers)
        mock_nn_2 = NeuralNetwork.from_layers(layers=nn_layers)
        mock_nn_3 = NeuralNetwork.from_layers(layers=nn_layers)

        mock_nn_3.weights, mock_nn_3.bias = NeuralNetwork.crossover(
            mock_nn_1, mock_nn_2, _mock_crossover_func, _mock_crossover_func
        )

        output = mock_nn_3.feedforward(mock_inputs)

        assert len(output) == mock_len_outputs
