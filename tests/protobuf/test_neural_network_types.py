"""Unit tests for the neural_network.protobuf.neural_network_types module."""

import numpy as np
import pytest

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import AdamOptimizer, SGDOptimizer
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunctionData,
    MatrixData,
    NeuralNetworkData,
    OptimizationAlgorithm,
    OptimizerData,
)
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    MatrixDataType,
    NeuralNetworkDataType,
    OptimizationAlgorithmEnum,
    OptimizerDataType,
)

rng = np.random.default_rng()


class TestActivationFunctionEnum:
    """Test cases for ActivationFunctionEnum conversions."""

    def test_get_class(self) -> None:
        """Test getting the activation function class from enum."""
        assert ActivationFunctionEnum.LINEAR.get_class() == LinearActivation
        assert ActivationFunctionEnum.RELU.get_class() == ReluActivation
        assert ActivationFunctionEnum.SIGMOID.get_class() == SigmoidActivation
        assert ActivationFunctionEnum.TANH.get_class() == TanhActivation

    def test_from_class(self) -> None:
        """Test getting the enum from activation function class."""
        assert ActivationFunctionEnum.from_class(LinearActivation) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_class(ReluActivation) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_class(SigmoidActivation) == ActivationFunctionEnum.SIGMOID
        assert ActivationFunctionEnum.from_class(TanhActivation) == ActivationFunctionEnum.TANH

    def test_from_protobuf(self) -> None:
        """Test getting the enum from protobuf value."""
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.LINEAR) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.RELU) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.SIGMOID) == ActivationFunctionEnum.SIGMOID
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.TANH) == ActivationFunctionEnum.TANH

    def test_to_protobuf(self) -> None:
        """Test converting enum to protobuf value."""
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.LINEAR) == ActivationFunctionData.LINEAR
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.RELU) == ActivationFunctionData.RELU
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.SIGMOID) == ActivationFunctionData.SIGMOID
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.TANH) == ActivationFunctionData.TANH


class TestMatrixDataType:
    """Test cases for MatrixDataType conversions."""

    @pytest.fixture
    def matrix_data(self) -> MatrixData:
        """Fixture for a MatrixData protobuf message."""
        rows = 1
        cols = 3
        return MatrixData(data=rng.uniform(-1, 1, rows * cols).tolist(), rows=rows, cols=cols)

    @pytest.fixture
    def matrix_data_type(self, matrix_data: MatrixData) -> MatrixDataType:
        """Fixture for a MatrixDataType instance."""
        return MatrixDataType(data=matrix_data.data, rows=matrix_data.rows, cols=matrix_data.cols)

    def test_from_protobuf(self, matrix_data: MatrixData) -> None:
        """Test creating MatrixDataType from protobuf message."""
        matrix_data_type = MatrixDataType.from_protobuf(matrix_data)

        assert matrix_data_type.data == matrix_data.data
        assert matrix_data_type.rows == matrix_data.rows
        assert matrix_data_type.cols == matrix_data.cols

    def test_to_protobuf(self, matrix_data_type: MatrixDataType) -> None:
        """Test converting MatrixDataType to protobuf message."""
        matrix_data = MatrixDataType.to_protobuf(matrix_data_type)

        assert matrix_data.data == matrix_data_type.data
        assert matrix_data.rows == matrix_data_type.rows
        assert matrix_data.cols == matrix_data_type.cols

    def test_from_bytes(self, matrix_data_type: MatrixDataType) -> None:
        """Test deserializing MatrixDataType from bytes."""
        msg_bytes = MatrixDataType.to_bytes(matrix_data_type)
        result = MatrixDataType.from_bytes(msg_bytes)

        assert result.data == pytest.approx(matrix_data_type.data)
        assert result.rows == matrix_data_type.rows
        assert result.cols == matrix_data_type.cols

    def test_to_bytes(self, matrix_data_type: MatrixDataType) -> None:
        """Test serializing MatrixDataType to bytes."""
        assert isinstance(MatrixDataType.to_bytes(matrix_data_type), bytes)

    def test_from_matrix(self, matrix_data_type: MatrixDataType) -> None:
        """Test creating MatrixDataType from Matrix."""
        matrix = Matrix.from_array(
            np.array(matrix_data_type.data).reshape((matrix_data_type.rows, matrix_data_type.cols))
        )
        result = MatrixDataType.from_matrix(matrix)

        assert result.data == pytest.approx(matrix_data_type.data)
        assert result.rows == matrix_data_type.rows
        assert result.cols == matrix_data_type.cols

    def test_to_matrix(self, matrix_data_type: MatrixDataType) -> None:
        """Test converting MatrixDataType to Matrix."""
        matrix = MatrixDataType.to_matrix(matrix_data_type)

        assert isinstance(matrix, Matrix)
        assert matrix.vals.shape == (matrix_data_type.rows, matrix_data_type.cols)
        assert np.allclose(matrix.vals.flatten(), matrix_data_type.data)


class TestOptimizationAlgorithmEnum:
    """Test cases for OptimizationAlgorithmEnum conversions."""

    def test_get_class(self) -> None:
        """Test getting the optimization algorithm class from enum."""
        assert OptimizationAlgorithmEnum.SGD.get_class() == SGDOptimizer
        assert OptimizationAlgorithmEnum.ADAM.get_class() == AdamOptimizer

    def test_from_class(self) -> None:
        """Test getting the enum from optimization algorithm class."""
        assert OptimizationAlgorithmEnum.from_class(SGDOptimizer) == OptimizationAlgorithmEnum.SGD
        assert OptimizationAlgorithmEnum.from_class(AdamOptimizer) == OptimizationAlgorithmEnum.ADAM

    def test_from_protobuf(self) -> None:
        """Test getting the enum from protobuf value."""
        assert OptimizationAlgorithmEnum.from_protobuf(OptimizationAlgorithm.SGD) == OptimizationAlgorithmEnum.SGD
        assert OptimizationAlgorithmEnum.from_protobuf(OptimizationAlgorithm.ADAM) == OptimizationAlgorithmEnum.ADAM

    def test_to_protobuf(self) -> None:
        """Test converting enum to protobuf value."""
        assert OptimizationAlgorithmEnum.to_protobuf(OptimizationAlgorithmEnum.SGD) == OptimizationAlgorithm.SGD
        assert OptimizationAlgorithmEnum.to_protobuf(OptimizationAlgorithmEnum.ADAM) == OptimizationAlgorithm.ADAM


class TestOptimizerDataType:
    """Test cases for OptimizerDataType conversions."""

    @pytest.fixture
    def optimizer_data(self) -> OptimizerData:
        """Fixture for an OptimizerData protobuf message."""
        return OptimizerData(
            algorithm=OptimizationAlgorithm.SGD,
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

    @pytest.fixture
    def optimizer_data_type(self, optimizer_data: OptimizerData) -> OptimizerDataType:
        """Fixture for an OptimizerDataType instance."""
        return OptimizerDataType(
            algorithm=OptimizationAlgorithmEnum.from_protobuf(optimizer_data.algorithm),
            learning_rate=optimizer_data.learning_rate,
            beta1=optimizer_data.beta1,
            beta2=optimizer_data.beta2,
            epsilon=optimizer_data.epsilon,
        )

    def test_from_protobuf(self, optimizer_data: OptimizerData) -> None:
        """Test creating OptimizerDataType from protobuf message."""
        optimizer_data_type = OptimizerDataType.from_protobuf(optimizer_data)

        assert optimizer_data_type.algorithm == OptimizationAlgorithmEnum.from_protobuf(optimizer_data.algorithm)
        assert optimizer_data_type.learning_rate == optimizer_data.learning_rate
        assert optimizer_data_type.beta1 == optimizer_data.beta1
        assert optimizer_data_type.beta2 == optimizer_data.beta2
        assert optimizer_data_type.epsilon == optimizer_data.epsilon

    def test_to_protobuf(self, optimizer_data_type: OptimizerDataType) -> None:
        """Test converting OptimizerDataType to protobuf message."""
        protobuf_data = OptimizerDataType.to_protobuf(optimizer_data_type)

        assert protobuf_data.algorithm == OptimizationAlgorithmEnum.to_protobuf(optimizer_data_type.algorithm)
        assert protobuf_data.learning_rate == optimizer_data_type.learning_rate
        assert protobuf_data.beta1 == optimizer_data_type.beta1
        assert protobuf_data.beta2 == optimizer_data_type.beta2
        assert protobuf_data.epsilon == optimizer_data_type.epsilon

    def test_from_bytes(self, optimizer_data_type: OptimizerDataType) -> None:
        """Test deserializing OptimizerDataType from bytes."""
        msg_bytes = OptimizerDataType.to_bytes(optimizer_data_type)
        result = OptimizerDataType.from_bytes(msg_bytes)

        assert result.algorithm == optimizer_data_type.algorithm
        assert result.learning_rate == optimizer_data_type.learning_rate
        assert result.beta1 == optimizer_data_type.beta1
        assert result.beta2 == optimizer_data_type.beta2
        assert result.epsilon == optimizer_data_type.epsilon

    def test_to_bytes(self, optimizer_data_type: OptimizerDataType) -> None:
        """Test serializing OptimizerDataType to bytes."""
        assert isinstance(OptimizerDataType.to_bytes(optimizer_data_type), bytes)


class TestNeuralNetworkDataType:
    """Test cases for NeuralNetworkDataType conversions."""

    @pytest.fixture
    def neural_network_data(self) -> NeuralNetworkData:
        """Fixture for a NeuralNetworkData protobuf message."""
        test_num_inputs = 2
        test_hidden_layer_sizes = [3]
        test_num_outputs = 1
        test_input_activation = ActivationFunctionData.RELU
        test_hidden_activation = ActivationFunctionData.SIGMOID
        test_output_activation = ActivationFunctionData.LINEAR
        test_weights_range = (-1, 1)
        test_bias_range = (-1, 1)
        test_learning_rate = 0.01

        input_weights_array = rng.uniform(*test_weights_range, (test_num_inputs, 1))
        input_weights_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(data=input_weights_array.flatten().tolist(), rows=test_num_inputs, cols=1)
        )
        input_bias_array = rng.uniform(*test_bias_range, (test_num_inputs, 1))
        input_bias_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(data=input_bias_array.flatten().tolist(), rows=test_num_inputs, cols=1)
        )

        hidden_weights_array = rng.uniform(*test_weights_range, (test_hidden_layer_sizes[0], test_num_inputs))
        hidden_weights_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(
                data=hidden_weights_array.flatten().tolist(), rows=test_hidden_layer_sizes[0], cols=test_num_inputs
            )
        )
        hidden_bias_array = rng.uniform(*test_bias_range, (test_hidden_layer_sizes[0], 1))
        hidden_bias_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(data=hidden_bias_array.flatten().tolist(), rows=test_hidden_layer_sizes[0], cols=1)
        )

        output_weights_array = rng.uniform(*test_weights_range, (test_num_outputs, test_hidden_layer_sizes[0]))
        output_weights_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(
                data=output_weights_array.flatten().tolist(), rows=test_num_outputs, cols=test_hidden_layer_sizes[0]
            )
        )
        output_bias_array = rng.uniform(*test_bias_range, (test_num_outputs, 1))
        output_bias_matrix_data = MatrixDataType.to_protobuf(
            MatrixDataType(data=output_bias_array.flatten().tolist(), rows=test_num_outputs, cols=1)
        )

        return NeuralNetworkData(
            num_inputs=test_num_inputs,
            hidden_layer_sizes=test_hidden_layer_sizes,
            num_outputs=test_num_outputs,
            input_activation=test_input_activation,
            hidden_activation=test_hidden_activation,
            output_activation=test_output_activation,
            weights=[input_weights_matrix_data, hidden_weights_matrix_data, output_weights_matrix_data],
            biases=[input_bias_matrix_data, hidden_bias_matrix_data, output_bias_matrix_data],
            learning_rate=test_learning_rate,
        )

    @pytest.fixture
    def neural_network_data_type(self, neural_network_data: NeuralNetworkData) -> NeuralNetworkDataType:
        """Fixture for a NeuralNetworkDataType instance."""
        return NeuralNetworkDataType(
            num_inputs=neural_network_data.num_inputs,
            hidden_layer_sizes=neural_network_data.hidden_layer_sizes,
            num_outputs=neural_network_data.num_outputs,
            input_activation=ActivationFunctionEnum.from_protobuf(neural_network_data.input_activation),
            hidden_activation=ActivationFunctionEnum.from_protobuf(neural_network_data.hidden_activation),
            output_activation=ActivationFunctionEnum.from_protobuf(neural_network_data.output_activation),
            learning_rate=neural_network_data.learning_rate,
            weights=neural_network_data.weights,
            biases=neural_network_data.biases,
        )

    def test_from_protobuf(self, neural_network_data: NeuralNetworkData) -> None:
        """Test creating NeuralNetworkDataType from protobuf message."""
        neural_network_data_type = NeuralNetworkDataType.from_protobuf(neural_network_data)

        assert neural_network_data_type.num_inputs == neural_network_data.num_inputs
        assert neural_network_data_type.hidden_layer_sizes == neural_network_data.hidden_layer_sizes
        assert neural_network_data_type.num_outputs == neural_network_data.num_outputs
        assert neural_network_data_type.input_activation == neural_network_data.input_activation
        assert neural_network_data_type.hidden_activation == neural_network_data.hidden_activation
        assert neural_network_data_type.output_activation == neural_network_data.output_activation
        assert neural_network_data_type.learning_rate == neural_network_data.learning_rate
        assert neural_network_data_type.weights == neural_network_data.weights
        assert neural_network_data_type.biases == neural_network_data.biases

    def test_to_protobuf(self, neural_network_data_type: NeuralNetworkDataType) -> None:
        """Test converting NeuralNetworkDataType to protobuf message."""
        protobuf_data = NeuralNetworkDataType.to_protobuf(neural_network_data_type)

        assert protobuf_data.num_inputs == neural_network_data_type.num_inputs
        assert protobuf_data.hidden_layer_sizes == neural_network_data_type.hidden_layer_sizes
        assert protobuf_data.num_outputs == neural_network_data_type.num_outputs
        assert protobuf_data.input_activation == neural_network_data_type.input_activation
        assert protobuf_data.hidden_activation == neural_network_data_type.hidden_activation
        assert protobuf_data.output_activation == neural_network_data_type.output_activation
        assert protobuf_data.learning_rate == neural_network_data_type.learning_rate
        assert protobuf_data.weights == neural_network_data_type.weights
        assert protobuf_data.biases == neural_network_data_type.biases

    def test_from_bytes(self, neural_network_data_type: NeuralNetworkDataType) -> None:
        """Test deserializing NeuralNetworkDataType from bytes."""
        msg_bytes = NeuralNetworkDataType.to_bytes(neural_network_data_type)
        result = NeuralNetworkDataType.from_bytes(msg_bytes)

        assert result.num_inputs == neural_network_data_type.num_inputs
        assert result.hidden_layer_sizes == neural_network_data_type.hidden_layer_sizes
        assert result.num_outputs == neural_network_data_type.num_outputs
        assert result.input_activation == neural_network_data_type.input_activation
        assert result.hidden_activation == neural_network_data_type.hidden_activation
        assert result.output_activation == neural_network_data_type.output_activation
        assert result.weights == pytest.approx(neural_network_data_type.weights)
        assert result.biases == pytest.approx(neural_network_data_type.biases)
        assert result.learning_rate == neural_network_data_type.learning_rate

    def test_to_bytes(self, neural_network_data_type: NeuralNetworkDataType) -> None:
        """Test serializing NeuralNetworkDataType to bytes."""
        assert isinstance(NeuralNetworkDataType.to_bytes(neural_network_data_type), bytes)
