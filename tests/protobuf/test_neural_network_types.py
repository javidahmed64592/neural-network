"""Unit tests for the neural_network.protobuf.neural_network_types module."""

# mypy: disable-error-code="union-attr"
import numpy as np
import pytest

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from neural_network.math.learning_rate_scheduler import (
    ExponentialDecayScheduler,
    LearningRateScheduler,
    StepDecayScheduler,
)
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import AdamOptimizer, SGDOptimizer
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunctionData,
    AdamOptimizerData,
    LearningRateMethod,
    LearningRateSchedulerData,
    MatrixData,
    NeuralNetworkData,
    OptimizerData,
    SGDOptimizerData,
)
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    AdamOptimizerDataType,
    LearningRateMethodEnum,
    LearningRateSchedulerDataType,
    MatrixDataType,
    NeuralNetworkDataType,
    OptimizerDataType,
    SGDOptimizerDataType,
)

rng = np.random.default_rng()


class TestActivationFunctionEnum:
    """Test cases for ActivationFunctionEnum conversions."""

    @pytest.mark.parametrize(
        ("enum_value", "expected_class"),
        [
            (ActivationFunctionEnum.LINEAR, LinearActivation),
            (ActivationFunctionEnum.RELU, ReluActivation),
            (ActivationFunctionEnum.SIGMOID, SigmoidActivation),
            (ActivationFunctionEnum.TANH, TanhActivation),
        ],
    )
    def test_get_class(self, enum_value: ActivationFunctionEnum, expected_class: type) -> None:
        """Test getting the activation function class from enum."""
        assert enum_value.get_class() == expected_class

    @pytest.mark.parametrize(
        ("activation_class", "expected_enum"),
        [
            (LinearActivation, ActivationFunctionEnum.LINEAR),
            (ReluActivation, ActivationFunctionEnum.RELU),
            (SigmoidActivation, ActivationFunctionEnum.SIGMOID),
            (TanhActivation, ActivationFunctionEnum.TANH),
        ],
    )
    def test_from_class(self, activation_class: type, expected_enum: ActivationFunctionEnum) -> None:
        """Test getting the enum from activation function class."""
        assert ActivationFunctionEnum.from_class(activation_class) == expected_enum

    @pytest.mark.parametrize(
        ("protobuf_value", "expected_enum"),
        [
            (ActivationFunctionData.LINEAR, ActivationFunctionEnum.LINEAR),
            (ActivationFunctionData.RELU, ActivationFunctionEnum.RELU),
            (ActivationFunctionData.SIGMOID, ActivationFunctionEnum.SIGMOID),
            (ActivationFunctionData.TANH, ActivationFunctionEnum.TANH),
        ],
    )
    def test_from_protobuf(self, protobuf_value: ActivationFunctionData, expected_enum: ActivationFunctionEnum) -> None:
        """Test getting the enum from protobuf value."""
        assert ActivationFunctionEnum.from_protobuf(protobuf_value) == expected_enum

    @pytest.mark.parametrize(
        ("enum_value", "expected_protobuf"),
        [
            (ActivationFunctionEnum.LINEAR, ActivationFunctionData.LINEAR),
            (ActivationFunctionEnum.RELU, ActivationFunctionData.RELU),
            (ActivationFunctionEnum.SIGMOID, ActivationFunctionData.SIGMOID),
            (ActivationFunctionEnum.TANH, ActivationFunctionData.TANH),
        ],
    )
    def test_to_protobuf(self, enum_value: ActivationFunctionEnum, expected_protobuf: ActivationFunctionData) -> None:
        """Test converting enum to protobuf value."""
        assert ActivationFunctionEnum.to_protobuf(enum_value) == expected_protobuf


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


class TestSGDOptimizerDataType:
    """Test cases for SGDOptimizerDataType conversions."""

    @pytest.fixture
    def sgd_optimizer_data(self) -> SGDOptimizerData:
        """Fixture for an SGDOptimizerData protobuf message."""
        return SGDOptimizerData(learning_rate=0.1)

    @pytest.fixture
    def sgd_optimizer_data_type(self, sgd_optimizer_data: SGDOptimizerData) -> SGDOptimizerDataType:
        """Fixture for an SGDOptimizerDataType instance."""
        return SGDOptimizerDataType(learning_rate=sgd_optimizer_data.learning_rate)

    def test_from_protobuf(self, sgd_optimizer_data: SGDOptimizerData) -> None:
        """Test creating SGDOptimizerDataType from protobuf message."""
        sgd_optimizer_data_type = SGDOptimizerDataType.from_protobuf(sgd_optimizer_data)

        assert sgd_optimizer_data_type.learning_rate == sgd_optimizer_data.learning_rate

    def test_to_protobuf(self, sgd_optimizer_data_type: SGDOptimizerDataType) -> None:
        """Test converting SGDOptimizerDataType to protobuf message."""
        protobuf_data = SGDOptimizerDataType.to_protobuf(sgd_optimizer_data_type)

        assert protobuf_data.learning_rate == sgd_optimizer_data_type.learning_rate

    def test_from_class_instance(self, sgd_optimizer_data_type: SGDOptimizerDataType) -> None:
        """Test creating SGDOptimizerDataType from class instance."""
        optimizer = SGDOptimizer(lr=sgd_optimizer_data_type.learning_rate, lr_scheduler=StepDecayScheduler())
        sgd_data = SGDOptimizerDataType.from_class_instance(optimizer)

        assert sgd_data.learning_rate == pytest.approx(sgd_optimizer_data_type.learning_rate)


class TestAdamOptimizerDataType:
    """Test cases for AdamOptimizerDataType conversions."""

    @pytest.fixture
    def adam_optimizer_data(self) -> AdamOptimizerData:
        """Fixture for an AdamOptimizerData protobuf message."""
        return AdamOptimizerData(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)

    @pytest.fixture
    def adam_optimizer_data_type(self, adam_optimizer_data: AdamOptimizerData) -> AdamOptimizerDataType:
        """Fixture for an AdamOptimizerDataType instance."""
        return AdamOptimizerDataType(
            learning_rate=adam_optimizer_data.learning_rate,
            beta1=adam_optimizer_data.beta1,
            beta2=adam_optimizer_data.beta2,
            epsilon=adam_optimizer_data.epsilon,
        )

    def test_from_protobuf(self, adam_optimizer_data: AdamOptimizerData) -> None:
        """Test creating AdamOptimizerDataType from protobuf message."""
        adam_optimizer_data_type = AdamOptimizerDataType.from_protobuf(adam_optimizer_data)

        assert adam_optimizer_data_type.learning_rate == adam_optimizer_data.learning_rate
        assert adam_optimizer_data_type.beta1 == adam_optimizer_data.beta1
        assert adam_optimizer_data_type.beta2 == adam_optimizer_data.beta2
        assert adam_optimizer_data_type.epsilon == adam_optimizer_data.epsilon

    def test_to_protobuf(self, adam_optimizer_data_type: AdamOptimizerDataType) -> None:
        """Test converting AdamOptimizerDataType to protobuf message."""
        protobuf_data = AdamOptimizerDataType.to_protobuf(adam_optimizer_data_type)

        assert protobuf_data.learning_rate == adam_optimizer_data_type.learning_rate
        assert protobuf_data.beta1 == adam_optimizer_data_type.beta1
        assert protobuf_data.beta2 == adam_optimizer_data_type.beta2
        assert protobuf_data.epsilon == adam_optimizer_data_type.epsilon

    def test_from_class_instance(self, adam_optimizer_data_type: AdamOptimizerDataType) -> None:
        """Test creating AdamOptimizerDataType from class instance."""
        optimizer = AdamOptimizer(
            lr=adam_optimizer_data_type.learning_rate,
            lr_scheduler=StepDecayScheduler(),
            beta1=adam_optimizer_data_type.beta1,
            beta2=adam_optimizer_data_type.beta2,
            epsilon=adam_optimizer_data_type.epsilon,
        )
        adam_data = AdamOptimizerDataType.from_class_instance(optimizer)

        assert adam_data.learning_rate == pytest.approx(adam_optimizer_data_type.learning_rate)
        assert adam_data.beta1 == adam_optimizer_data_type.beta1
        assert adam_data.beta2 == adam_optimizer_data_type.beta2
        assert adam_data.epsilon == adam_optimizer_data_type.epsilon


class TestLearningRateMethodEnum:
    """Test cases for LearningRateMethodEnum conversions."""

    @pytest.mark.parametrize(
        ("method", "expected_class"),
        [
            (LearningRateMethodEnum.STEP_DECAY, StepDecayScheduler),
            (LearningRateMethodEnum.EXPONENTIAL_DECAY, ExponentialDecayScheduler),
        ],
    )
    def test_get_class(self, method: LearningRateMethodEnum, expected_class: type[LearningRateScheduler]) -> None:
        """Test getting the learning rate method class from enum."""
        assert method.get_class() == expected_class

    @pytest.mark.parametrize(
        ("method_class", "expected_enum"),
        [
            (StepDecayScheduler, LearningRateMethodEnum.STEP_DECAY),
            (ExponentialDecayScheduler, LearningRateMethodEnum.EXPONENTIAL_DECAY),
        ],
    )
    def test_from_class(self, method_class: type[LearningRateScheduler], expected_enum: LearningRateMethodEnum) -> None:
        """Test getting the enum from learning rate method class."""
        assert LearningRateMethodEnum.from_class(method_class) == expected_enum

    @pytest.mark.parametrize(
        ("protobuf_value", "expected_enum"),
        [
            (LearningRateMethod.STEP_DECAY, LearningRateMethodEnum.STEP_DECAY),
            (LearningRateMethod.EXPONENTIAL_DECAY, LearningRateMethodEnum.EXPONENTIAL_DECAY),
        ],
    )
    def test_from_protobuf(self, protobuf_value: LearningRateMethod, expected_enum: LearningRateMethodEnum) -> None:
        """Test getting the enum from protobuf value."""
        assert LearningRateMethodEnum.from_protobuf(protobuf_value) == expected_enum

    @pytest.mark.parametrize(
        ("enum_value", "expected_protobuf"),
        [
            (LearningRateMethodEnum.STEP_DECAY, LearningRateMethod.STEP_DECAY),
            (LearningRateMethodEnum.EXPONENTIAL_DECAY, LearningRateMethod.EXPONENTIAL_DECAY),
        ],
    )
    def test_to_protobuf(self, enum_value: LearningRateMethodEnum, expected_protobuf: LearningRateMethod) -> None:
        """Test converting enum to protobuf value."""
        assert LearningRateMethodEnum.to_protobuf(enum_value) == expected_protobuf


class TestLearningRateSchedulerDataType:
    """Test cases for LearningRateSchedulerDataType conversions."""

    @pytest.mark.parametrize(
        ("decay_rate", "decay_steps", "method"),
        [
            (5, 100, LearningRateMethodEnum.STEP_DECAY),
            (2, 50, LearningRateMethodEnum.EXPONENTIAL_DECAY),
        ],
    )
    def test_from_protobuf(self, decay_rate: int, decay_steps: int, method: LearningRateMethodEnum) -> None:
        """Test creating LearningRateSchedulerDataType from protobuf message."""
        lr_data = LearningRateSchedulerData(
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            method=LearningRateMethodEnum.to_protobuf(method),
        )
        lr_data_type = LearningRateSchedulerDataType.from_protobuf(lr_data)

        assert lr_data_type.decay_rate == decay_rate
        assert lr_data_type.decay_steps == decay_steps
        assert lr_data_type.method == method

    @pytest.mark.parametrize(
        ("decay_rate", "decay_steps", "method"),
        [
            (5, 100, LearningRateMethodEnum.STEP_DECAY),
            (2, 50, LearningRateMethodEnum.EXPONENTIAL_DECAY),
        ],
    )
    def test_to_protobuf(self, decay_rate: int, decay_steps: int, method: LearningRateMethodEnum) -> None:
        """Test converting LearningRateSchedulerDataType to protobuf message."""
        lr_data_type = LearningRateSchedulerDataType(
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            method=method,
        )

        protobuf_data = LearningRateSchedulerDataType.to_protobuf(lr_data_type)

        assert protobuf_data.decay_rate == lr_data_type.decay_rate
        assert protobuf_data.decay_steps == lr_data_type.decay_steps
        assert protobuf_data.method == LearningRateMethodEnum.to_protobuf(lr_data_type.method)

    @pytest.mark.parametrize(
        ("scheduler_class", "decay_rate", "decay_steps", "expected_method"),
        [
            (StepDecayScheduler, 5, 100, LearningRateMethodEnum.STEP_DECAY),
            (ExponentialDecayScheduler, 2, 50, LearningRateMethodEnum.EXPONENTIAL_DECAY),
        ],
    )
    def test_from_class_instance(
        self,
        scheduler_class: type[LearningRateScheduler],
        decay_rate: float,
        decay_steps: int,
        expected_method: LearningRateMethodEnum,
    ) -> None:
        """Test creating LearningRateSchedulerDataType from class instance."""
        scheduler = scheduler_class(
            decay_rate=decay_rate,
            decay_steps=decay_steps,
        )

        lr_data = LearningRateSchedulerDataType.from_class_instance(scheduler)

        assert lr_data.decay_rate == decay_rate
        assert lr_data.decay_steps == decay_steps
        assert lr_data.method == expected_method

    @pytest.mark.parametrize(
        ("decay_rate", "decay_steps", "method", "expected_class"),
        [
            (5, 100, LearningRateMethodEnum.STEP_DECAY, StepDecayScheduler),
            (2, 50, LearningRateMethodEnum.EXPONENTIAL_DECAY, ExponentialDecayScheduler),
        ],
    )
    def test_get_class_instance(
        self,
        decay_rate: float,
        decay_steps: int,
        method: LearningRateMethodEnum,
        expected_class: type[LearningRateScheduler],
    ) -> None:
        """Test getting the learning rate scheduler class instance."""
        lr_data = LearningRateSchedulerDataType(
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            method=method,
        )

        scheduler = lr_data.get_class_instance()

        assert isinstance(scheduler, expected_class)
        assert scheduler.decay_rate == decay_rate
        assert scheduler.decay_steps == decay_steps


class TestOptimizerDataType:
    """Test cases for OptimizerDataType conversions."""

    @pytest.fixture
    def sgd_optimizer_data(self) -> OptimizerData:
        """Fixture for an SGDOptimizerData protobuf message."""
        return OptimizerData(
            sgd=SGDOptimizerData(learning_rate=0.1),
            learning_rate_scheduler=LearningRateSchedulerData(
                decay_rate=0.5, decay_steps=10, method=LearningRateMethod.STEP_DECAY
            ),
        )

    @pytest.fixture
    def adam_optimizer_data(self) -> OptimizerData:
        """Fixture for an AdamOptimizerData protobuf message."""
        return OptimizerData(
            adam=AdamOptimizerData(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8),
            learning_rate_scheduler=LearningRateSchedulerData(
                decay_rate=0.5, decay_steps=10, method=LearningRateMethod.STEP_DECAY
            ),
        )

    @pytest.fixture
    def sgd_optimizer_data_type(self, sgd_optimizer_data: OptimizerData) -> OptimizerDataType:
        """Fixture for an OptimizerDataType instance from SGD optimizer."""
        return OptimizerDataType.from_protobuf(sgd_optimizer_data)

    @pytest.fixture
    def adam_optimizer_data_type(self, adam_optimizer_data: OptimizerData) -> OptimizerDataType:
        """Fixture for an OptimizerDataType instance from Adam optimizer."""
        return OptimizerDataType.from_protobuf(adam_optimizer_data)

    def test_from_protobuf(self, sgd_optimizer_data: OptimizerData, adam_optimizer_data: OptimizerData) -> None:
        """Test creating OptimizerDataType from protobuf message."""
        sgd_optimizer_data_type = OptimizerDataType.from_protobuf(sgd_optimizer_data)
        assert sgd_optimizer_data_type.adam is None
        assert sgd_optimizer_data_type.sgd == SGDOptimizerDataType.from_protobuf(sgd_optimizer_data.sgd)
        sgd_instance = sgd_optimizer_data_type.get_class_instance()
        assert isinstance(sgd_instance, SGDOptimizer)
        assert sgd_instance.lr == sgd_optimizer_data.sgd.learning_rate

        adam_optimizer_data_type = OptimizerDataType.from_protobuf(adam_optimizer_data)
        assert adam_optimizer_data_type.sgd is None
        assert adam_optimizer_data_type.adam == AdamOptimizerDataType.from_protobuf(adam_optimizer_data.adam)
        adam_instance = adam_optimizer_data_type.get_class_instance()
        assert isinstance(adam_instance, AdamOptimizer)
        assert adam_instance.lr == adam_optimizer_data.adam.learning_rate  # Check _lr directly
        assert adam_instance.beta1 == adam_optimizer_data.adam.beta1
        assert adam_instance.beta2 == adam_optimizer_data.adam.beta2
        assert adam_instance.epsilon == adam_optimizer_data.adam.epsilon

    def test_to_protobuf(
        self, sgd_optimizer_data_type: OptimizerDataType, adam_optimizer_data_type: OptimizerDataType
    ) -> None:
        """Test converting OptimizerDataType to protobuf message."""
        sgd_protobuf_data = OptimizerDataType.to_protobuf(sgd_optimizer_data_type)
        assert isinstance(sgd_protobuf_data.sgd, SGDOptimizerData)
        assert sgd_protobuf_data.sgd.learning_rate == sgd_optimizer_data_type.sgd.learning_rate

        adam_protobuf_data = OptimizerDataType.to_protobuf(adam_optimizer_data_type)
        assert adam_protobuf_data.adam.learning_rate == adam_optimizer_data_type.adam.learning_rate
        assert adam_protobuf_data.adam.beta1 == adam_optimizer_data_type.adam.beta1
        assert adam_protobuf_data.adam.beta2 == adam_optimizer_data_type.adam.beta2
        assert adam_protobuf_data.adam.epsilon == adam_optimizer_data_type.adam.epsilon

    def test_from_bytes(
        self, sgd_optimizer_data_type: OptimizerDataType, adam_optimizer_data_type: OptimizerDataType
    ) -> None:
        """Test deserializing OptimizerDataType from bytes."""
        # Test with SGD optimizer data type
        msg_bytes = OptimizerDataType.to_bytes(sgd_optimizer_data_type)
        result = OptimizerDataType.from_bytes(msg_bytes)
        assert result.sgd == sgd_optimizer_data_type.sgd

        # Test with Adam optimizer data type
        msg_bytes = OptimizerDataType.to_bytes(adam_optimizer_data_type)
        result = OptimizerDataType.from_bytes(msg_bytes)
        assert result.adam == adam_optimizer_data_type.adam

    def test_to_bytes(
        self, sgd_optimizer_data_type: OptimizerDataType, adam_optimizer_data_type: OptimizerDataType
    ) -> None:
        """Test serializing OptimizerDataType to bytes."""
        assert isinstance(OptimizerDataType.to_bytes(sgd_optimizer_data_type), bytes)
        assert isinstance(OptimizerDataType.to_bytes(adam_optimizer_data_type), bytes)

    def test_from_class_instance(
        self, sgd_optimizer_data_type: OptimizerDataType, adam_optimizer_data_type: OptimizerDataType
    ) -> None:
        """Test creating OptimizerDataType from class instance."""
        sgd_instance = sgd_optimizer_data_type.get_class_instance()
        sgd_data = OptimizerDataType.from_class_instance(sgd_instance)
        assert sgd_data.sgd.learning_rate == sgd_optimizer_data_type.sgd.learning_rate

        adam_instance = adam_optimizer_data_type.get_class_instance()
        adam_data = OptimizerDataType.from_class_instance(adam_instance)
        assert adam_data.adam.learning_rate == adam_optimizer_data_type.adam.learning_rate
        assert adam_data.adam.beta1 == adam_optimizer_data_type.adam.beta1
        assert adam_data.adam.beta2 == adam_optimizer_data_type.adam.beta2
        assert adam_data.adam.epsilon == adam_optimizer_data_type.adam.epsilon

    def test_get_class_instance(
        self, sgd_optimizer_data_type: OptimizerDataType, adam_optimizer_data_type: OptimizerDataType
    ) -> None:
        """Test getting the optimizer class instance."""
        sgd_instance = sgd_optimizer_data_type.get_class_instance()
        assert isinstance(sgd_instance, SGDOptimizer)
        assert sgd_instance.lr == sgd_optimizer_data_type.sgd.learning_rate

        adam_instance = adam_optimizer_data_type.get_class_instance()
        assert isinstance(adam_instance, AdamOptimizer)
        assert adam_instance.lr == adam_optimizer_data_type.adam.learning_rate
        assert adam_instance.beta1 == adam_optimizer_data_type.adam.beta1
        assert adam_instance.beta2 == adam_optimizer_data_type.adam.beta2
        assert adam_instance.epsilon == adam_optimizer_data_type.adam.epsilon


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
        test_optimizer = OptimizerData(sgd=SGDOptimizerData(learning_rate=test_learning_rate))

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
            optimizer=test_optimizer,
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
            weights=neural_network_data.weights,
            biases=neural_network_data.biases,
            optimizer=OptimizerDataType.from_protobuf(neural_network_data.optimizer),
        )

    def test_from_protobuf(self, neural_network_data: NeuralNetworkData) -> None:
        """Test creating NeuralNetworkDataType from protobuf message."""
        neural_network_data_type = NeuralNetworkDataType.from_protobuf(neural_network_data)

        assert neural_network_data_type.num_inputs == neural_network_data.num_inputs
        assert neural_network_data_type.hidden_layer_sizes == neural_network_data.hidden_layer_sizes
        assert neural_network_data_type.num_outputs == neural_network_data.num_outputs
        assert neural_network_data_type.input_activation == ActivationFunctionEnum.from_protobuf(
            neural_network_data.input_activation
        )
        assert neural_network_data_type.hidden_activation == ActivationFunctionEnum.from_protobuf(
            neural_network_data.hidden_activation
        )
        assert neural_network_data_type.output_activation == ActivationFunctionEnum.from_protobuf(
            neural_network_data.output_activation
        )
        assert len(neural_network_data_type.weights) == len(neural_network_data.weights)
        assert len(neural_network_data_type.biases) == len(neural_network_data.biases)
        assert neural_network_data_type.optimizer == OptimizerDataType.from_protobuf(neural_network_data.optimizer)

    def test_to_protobuf(self, neural_network_data_type: NeuralNetworkDataType) -> None:
        """Test converting NeuralNetworkDataType to protobuf message."""
        protobuf_data = NeuralNetworkDataType.to_protobuf(neural_network_data_type)

        assert protobuf_data.num_inputs == neural_network_data_type.num_inputs
        assert protobuf_data.hidden_layer_sizes == neural_network_data_type.hidden_layer_sizes
        assert protobuf_data.num_outputs == neural_network_data_type.num_outputs
        assert protobuf_data.input_activation == ActivationFunctionEnum.to_protobuf(
            neural_network_data_type.input_activation
        )
        assert protobuf_data.hidden_activation == ActivationFunctionEnum.to_protobuf(
            neural_network_data_type.hidden_activation
        )
        assert protobuf_data.output_activation == ActivationFunctionEnum.to_protobuf(
            neural_network_data_type.output_activation
        )
        assert len(protobuf_data.weights) == len(neural_network_data_type.weights)
        assert len(protobuf_data.biases) == len(neural_network_data_type.biases)
        assert protobuf_data.optimizer == OptimizerDataType.to_protobuf(neural_network_data_type.optimizer)

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
        assert len(result.weights) == len(neural_network_data_type.weights)
        assert len(result.biases) == len(neural_network_data_type.biases)
        assert result.optimizer == neural_network_data_type.optimizer

    def test_to_bytes(self, neural_network_data_type: NeuralNetworkDataType) -> None:
        """Test serializing NeuralNetworkDataType to bytes."""
        assert isinstance(NeuralNetworkDataType.to_bytes(neural_network_data_type), bytes)
