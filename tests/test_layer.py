"""Unit tests for the neural_network.layer module."""

import numpy as np

from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix


class TestLayer:
    """Test cases for generic neural network layers."""

    def test_given_layers_when_adding_nodes_then_check_layers_have_correct_sizes(
        self,
        mock_input_layer: InputLayer,
        mock_hidden_layer_1: HiddenLayer,
        mock_output_layer: OutputLayer,
        mock_len_inputs: int,
        mock_len_hidden: list[int],
        mock_len_outputs: int,
    ) -> None:
        """Test that layers have correct sizes after adding nodes."""
        assert mock_input_layer.size == mock_len_inputs
        assert mock_input_layer.num_inputs == 1
        assert mock_hidden_layer_1.size == mock_len_hidden[0]
        assert mock_hidden_layer_1.num_inputs == mock_len_inputs
        assert mock_output_layer.size == mock_len_outputs
        assert mock_output_layer.num_inputs == mock_len_hidden[-1]

    def test_given_number_of_nodes_when_creating_layer_then_check_weights_and_bias_have_correct_shape(
        self, mock_hidden_layer_1: HiddenLayer, mock_len_hidden: list[int], mock_len_inputs: int
    ) -> None:
        """Test that weights and bias have correct shapes for a layer."""
        expected_weights_shape = (mock_len_hidden[0], mock_len_inputs)
        expected_bias_shape = (mock_len_hidden[0], 1)

        actual_weights_shape = mock_hidden_layer_1.weights.shape
        actual_bias_shape = mock_hidden_layer_1.bias.shape
        assert actual_weights_shape == expected_weights_shape
        assert actual_bias_shape == expected_bias_shape

    def test_given_layers_when_setting_previous_layer_then_check_previous_layer_is_set(self) -> None:
        """Test setting the previous layer for a hidden layer."""
        input_layer = InputLayer(3, ActivationFunction)
        hidden_layer = HiddenLayer(4, ActivationFunction, (0.0, 1.0), (0.0, 1.0))
        hidden_layer.set_prev_layer(input_layer)
        assert hidden_layer._prev_layer == input_layer

    def test_given_inputs_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_hidden_layer_1: HiddenLayer, mock_len_hidden: list[int], mock_input_matrix: Matrix
    ) -> None:
        """Test feedforward through a hidden layer produces output of correct shape."""
        output = mock_hidden_layer_1.feedforward(mock_input_matrix)

        expected_output_shape = (mock_len_hidden[0], 1)
        actual_output_shape = output.shape
        assert actual_output_shape == expected_output_shape

    def test_given_errors_when_backpropagating_then_check_weights_and_biases_are_updated(
        self, mock_hidden_layer_1: HiddenLayer, mock_len_hidden: list[int], mock_input_matrix: Matrix
    ) -> None:
        """Test that weights and biases are updated after backpropagation."""
        original_weights = mock_hidden_layer_1.weights.vals.copy()
        original_bias = mock_hidden_layer_1.bias.vals.copy()

        mock_hidden_layer_1.feedforward(mock_input_matrix)

        errors = Matrix.random_matrix(mock_len_hidden[0], 1, -1, 1)
        mock_hidden_layer_1.backpropagate_error(errors)

        assert not np.array_equal(mock_hidden_layer_1.weights.vals, original_weights)
        assert not np.array_equal(mock_hidden_layer_1.bias.vals, original_bias)


class TestInputLayer:
    """Test cases for the InputLayer class."""

    def test_given_input_layer_when_creating_then_check_layer_has_correct_size(
        self, mock_input_layer: InputLayer, mock_len_inputs: int
    ) -> None:
        """Test that input layer has correct size and no previous layer."""
        assert mock_input_layer.size == mock_len_inputs
        assert mock_input_layer.num_inputs == 1
        assert mock_input_layer._prev_layer is None

    def test_given_input_layer_when_performing_feedforward_then_check_output_has_correct_shape(
        self, mock_input_layer: InputLayer, mock_len_inputs: int, mock_input_matrix: Matrix
    ) -> None:
        """Test feedforward through input layer produces output of correct shape."""
        output = mock_input_layer.feedforward(mock_input_matrix)
        expected_output_shape = (mock_len_inputs, 1)
        actual_output_shape = output.shape
        assert actual_output_shape == expected_output_shape
        assert np.array_equal(output.vals, mock_input_matrix.vals)
        assert np.array_equal(output.vals, mock_input_layer._layer_input.vals)
        assert np.array_equal(output.vals, mock_input_layer._layer_output.vals)


class TestHiddenLayer:
    """Test cases for the HiddenLayer class."""

    def test_given_hidden_layer_when_creating_then_check_layer_has_correct_size(
        self,
        mock_input_layer: InputLayer,
        mock_hidden_layer_1: HiddenLayer,
        mock_len_hidden: list[int],
        mock_len_inputs: int,
    ) -> None:
        """Test that hidden layer has correct size and connections."""
        assert mock_hidden_layer_1.size == mock_len_hidden[0]
        assert mock_hidden_layer_1.num_inputs == mock_len_inputs
        assert mock_hidden_layer_1._prev_layer == mock_input_layer
        assert mock_input_layer._next_layer == mock_hidden_layer_1
        assert mock_hidden_layer_1._next_layer is None


class TestOutputLayer:
    """Test cases for the OutputLayer class."""

    def test_given_output_layer_when_creating_then_check_layer_has_correct_size(
        self,
        mock_hidden_layer_3: HiddenLayer,
        mock_output_layer: OutputLayer,
        mock_len_hidden: list[int],
        mock_len_outputs: int,
    ) -> None:
        """Test that output layer has correct size and connections."""
        assert mock_output_layer.size == mock_len_outputs
        assert mock_output_layer.num_inputs == mock_len_hidden[-1]
        assert mock_output_layer._prev_layer == mock_hidden_layer_3
        assert mock_hidden_layer_3._next_layer == mock_output_layer
        assert mock_output_layer._next_layer is None
