"""Dataclasses and enums for neural network and configuration Protobuf messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)
from neural_network.math.learning_rate_scheduler import (
    ExponentialDecayScheduler,
    LearningRateScheduler,
    StepDecayScheduler,
)
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import AdamOptimizer, Optimizer, SGDOptimizer
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


class ActivationFunctionEnum(IntEnum):
    """Enum for supported activation functions."""

    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

    @property
    def map(self) -> dict[ActivationFunctionEnum, type[ActivationFunction]]:
        """Return a mapping from enum to activation function class.

        :return dict[ActivationFunctionEnum, type[ActivationFunction]]:
            Mapping from enum to activation function class.
        """
        return {
            ActivationFunctionEnum.LINEAR: LinearActivation,
            ActivationFunctionEnum.RELU: ReluActivation,
            ActivationFunctionEnum.SIGMOID: SigmoidActivation,
            ActivationFunctionEnum.TANH: TanhActivation,
        }

    def get_class(self) -> type[ActivationFunction]:
        """Return the corresponding activation function class.

        :return type[ActivationFunction]:
            The activation function class.
        """
        return self.map[self]

    @classmethod
    def from_class(cls, activation_function: type[ActivationFunction]) -> ActivationFunctionEnum:
        """Return the enum value for a given activation function class.

        :param type[ActivationFunction] activation_function:
            The activation function class.
        :return ActivationFunctionEnum:
            The corresponding enum value.
        """
        reverse_map = {v: k for k, v in cls.LINEAR.map.items()}
        return reverse_map[activation_function]

    @classmethod
    def from_protobuf(cls, proto_enum_value: ActivationFunctionData) -> ActivationFunctionEnum:
        """Return the enum value from a Protobuf ActivationFunctionEnum value.

        :param ActivationFunctionEnum proto_enum_value:
            The Protobuf enum value.
        :return ActivationFunctionEnum:
            The corresponding enum value.
        """
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnum) -> ActivationFunctionData:
        """Return the Protobuf ActivationFunctionData from an enum value.

        :param ActivationFunctionEnum enum_value:
            The enum value.
        :return ActivationFunctionData:
            The Protobuf enum value.
        """
        return ActivationFunctionData.Value(enum_value.name)  # type: ignore[no-any-return]


@dataclass
class MatrixDataType:
    """Data class to hold matrix data."""

    data: list[float]
    rows: int
    cols: int

    @classmethod
    def from_protobuf(cls, matrix_data: MatrixData) -> MatrixDataType:
        """Create a MatrixDataType instance from Protobuf.

        :param MatrixData matrix_data:
            The Protobuf MatrixData message.
        :return MatrixDataType:
            The corresponding MatrixDataType instance.
        """
        return cls(
            data=list(matrix_data.data),
            rows=matrix_data.rows,
            cols=matrix_data.cols,
        )

    @staticmethod
    def to_protobuf(matrix_data: MatrixDataType) -> MatrixData:
        """Convert MatrixDataType to Protobuf.

        :param MatrixDataType matrix_data:
            The MatrixDataType instance.
        :return MatrixData:
            The corresponding Protobuf MatrixData message.
        """
        return MatrixData(
            data=matrix_data.data,
            rows=matrix_data.rows,
            cols=matrix_data.cols,
        )

    @classmethod
    def from_bytes(cls, matrix_data: bytes) -> MatrixDataType:
        """Create a MatrixDataType instance from Protobuf bytes.

        :param bytes matrix_data:
            The Protobuf-serialized MatrixData bytes.
        :return MatrixDataType:
            The corresponding MatrixDataType instance.
        """
        matrix = MatrixData()
        matrix.ParseFromString(matrix_data)
        return cls.from_protobuf(matrix)

    @staticmethod
    def to_bytes(matrix_data: MatrixDataType) -> bytes:
        """Convert MatrixDataType to Protobuf bytes.

        :param MatrixDataType matrix_data:
            The MatrixDataType instance.
        :return bytes:
            The Protobuf-serialized MatrixData bytes.
        """
        matrix = MatrixDataType.to_protobuf(matrix_data)
        return matrix.SerializeToString()  # type: ignore[no-any-return]

    @classmethod
    def from_matrix(cls, matrix: Matrix) -> MatrixDataType:
        """Create a MatrixDataType instance from a Matrix.

        :param Matrix matrix:
            The Matrix instance.
        :return MatrixDataType:
            The corresponding MatrixDataType instance.
        """
        return cls(
            data=matrix.vals.flatten().tolist(),
            rows=matrix.rows,
            cols=matrix.cols,
        )

    @staticmethod
    def to_matrix(matrix_data: MatrixDataType) -> Matrix:
        """Convert MatrixDataType to a Matrix.

        :param MatrixDataType matrix_data:
            The MatrixDataType instance.
        :return Matrix:
            The corresponding Matrix instance.
        """
        matrix_array = np.array(matrix_data.data, dtype=np.float64).reshape((matrix_data.rows, matrix_data.cols))
        return Matrix.from_array(matrix_array)


@dataclass
class SGDOptimizerDataType:
    """Data class to hold SGD optimizer data."""

    learning_rate: float

    @classmethod
    def from_protobuf(cls, sgd_data: SGDOptimizerData) -> SGDOptimizerDataType:
        """Create a SGDOptimizerDataType instance from Protobuf.

        :param SGDOptimizerData sgd_data:
            The Protobuf SGDOptimizerData message.
        :return SGDOptimizerDataType:
            The corresponding SGDOptimizerDataType instance.
        """
        return cls(learning_rate=sgd_data.learning_rate)

    @staticmethod
    def to_protobuf(sgd_data: SGDOptimizerDataType) -> SGDOptimizerData:
        """Convert SGDOptimizerDataType to Protobuf.

        :param SGDOptimizerDataType sgd_data:
            The SGDOptimizerDataType instance.
        :return SGDOptimizerData:
            The corresponding Protobuf SGDOptimizerData message.
        """
        return SGDOptimizerData(learning_rate=sgd_data.learning_rate)

    @classmethod
    def from_class_instance(cls, optimizer: SGDOptimizer) -> SGDOptimizerDataType:
        """Create a SGDOptimizerDataType instance from an SGDOptimizer class instance.

        :param SGDOptimizer optimizer:
            The SGDOptimizer class instance.
        :return SGDOptimizerDataType:
            The corresponding SGDOptimizerDataType instance.
        """
        return cls(learning_rate=optimizer._lr)


@dataclass
class AdamOptimizerDataType:
    """Data class to hold Adam optimizer data."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    @classmethod
    def from_protobuf(cls, adam_data: AdamOptimizerData) -> AdamOptimizerDataType:
        """Create a AdamOptimizerDataType instance from Protobuf.

        :param AdamOptimizerData adam_data:
            The Protobuf AdamOptimizerData message.
        :return AdamOptimizerDataType:
            The corresponding AdamOptimizerDataType instance.
        """
        return cls(
            learning_rate=adam_data.learning_rate,
            beta1=adam_data.beta1,
            beta2=adam_data.beta2,
            epsilon=adam_data.epsilon,
        )

    @staticmethod
    def to_protobuf(adam_data: AdamOptimizerDataType) -> AdamOptimizerData:
        """Convert AdamOptimizerDataType to Protobuf.

        :param AdamOptimizerDataType adam_data:
            The AdamOptimizerDataType instance.
        :return AdamOptimizerData:
            The corresponding Protobuf AdamOptimizerData message.
        """
        return AdamOptimizerData(
            learning_rate=adam_data.learning_rate,
            beta1=adam_data.beta1,
            beta2=adam_data.beta2,
            epsilon=adam_data.epsilon,
        )

    @classmethod
    def from_class_instance(cls, optimizer: AdamOptimizer) -> AdamOptimizerDataType:
        """Create an AdamOptimizerDataType instance from an AdamOptimizer class instance.

        :param AdamOptimizer optimizer:
            The AdamOptimizer class instance.
        :return AdamOptimizerDataType:
            The corresponding AdamOptimizerDataType instance.
        """
        return cls(
            learning_rate=optimizer._lr,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
        )


class LearningRateMethodEnum(IntEnum):
    """Enum for supported learning rate methods."""

    STEP_DECAY = 0
    EXPONENTIAL_DECAY = 1

    @property
    def map(self) -> dict[LearningRateMethodEnum, type[LearningRateScheduler]]:
        """Return a mapping from enum to learning rate scheduler class.

        :return dict[LearningRateMethodEnum, type[LearningRateScheduler]]:
            Mapping from enum to learning rate scheduler class.
        """
        return {
            LearningRateMethodEnum.STEP_DECAY: StepDecayScheduler,
            LearningRateMethodEnum.EXPONENTIAL_DECAY: ExponentialDecayScheduler,
        }

    def get_class(self) -> type[LearningRateScheduler]:
        """Return the corresponding learning rate scheduler class.

        :return type[LearningRateScheduler]:
            The learning rate scheduler class.
        """
        return self.map[self]

    @classmethod
    def from_class(cls, learning_rate_scheduler: type[LearningRateScheduler]) -> LearningRateMethodEnum:
        """Return the enum value for a given learning rate scheduler class.

        :param type[LearningRateScheduler] learning_rate_scheduler:
            The learning rate scheduler class.
        :return LearningRateMethodEnum:
            The corresponding enum value.
        """
        reverse_map = {v: k for k, v in cls.STEP_DECAY.map.items()}
        return reverse_map[learning_rate_scheduler]

    @classmethod
    def from_protobuf(cls, proto_enum_value: LearningRateMethod) -> LearningRateMethodEnum:
        """Return the enum value from a Protobuf LearningRateMethodEnum value.

        :param LearningRateMethod proto_enum_value:
            The Protobuf enum value.
        :return LearningRateMethodEnum:
            The corresponding enum value.
        """
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: LearningRateMethodEnum) -> LearningRateMethod:
        """Return the Protobuf LearningRateMethod from an enum value.

        :param LearningRateMethodEnum enum_value:
            The enum value.
        :return LearningRateMethod:
            The Protobuf enum value.
        """
        return LearningRateMethod.Value(enum_value.name)  # type: ignore[no-any-return]


@dataclass
class LearningRateSchedulerDataType:
    """Data class to hold learning rate scheduler data."""

    decay_rate: float
    decay_steps: int
    method: LearningRateMethodEnum

    @classmethod
    def from_protobuf(cls, lr_data: LearningRateSchedulerData) -> LearningRateSchedulerDataType:
        """Create a LearningRateSchedulerDataType instance from Protobuf.

        :param LearningRateSchedulerData lr_data:
            The Protobuf LearningRateSchedulerData message.
        :return LearningRateSchedulerDataType:
            The corresponding LearningRateSchedulerDataType instance.
        """
        return cls(
            decay_rate=lr_data.decay_rate,
            decay_steps=lr_data.decay_steps,
            method=LearningRateMethodEnum.from_protobuf(lr_data.method),
        )

    @staticmethod
    def to_protobuf(lr_data: LearningRateSchedulerDataType) -> LearningRateSchedulerData:
        """Convert LearningRateSchedulerDataType to Protobuf.

        :param LearningRateSchedulerDataType lr_data:
            The LearningRateSchedulerDataType instance.
        :return LearningRateSchedulerData:
            The corresponding Protobuf LearningRateSchedulerData message.
        """
        return LearningRateSchedulerData(
            decay_rate=lr_data.decay_rate,
            decay_steps=lr_data.decay_steps,
            method=LearningRateMethodEnum.to_protobuf(lr_data.method),
        )

    @classmethod
    def from_class_instance(cls, lr_scheduler: LearningRateScheduler) -> LearningRateSchedulerDataType:
        """Create a LearningRateSchedulerDataType instance from a LearningRateScheduler class instance.

        :param LearningRateScheduler lr_scheduler:
            The LearningRateScheduler class instance.
        :return LearningRateSchedulerDataType:
            The corresponding LearningRateSchedulerDataType instance.
        """
        return cls(
            decay_rate=lr_scheduler.decay_rate,
            decay_steps=lr_scheduler.decay_steps,
            method=LearningRateMethodEnum.from_class(lr_scheduler.__class__),
        )

    def get_class_instance(self) -> LearningRateScheduler:
        """Return an instance of the LearningRateScheduler with the stored parameters.

        :return LearningRateScheduler:
            An instance of LearningRateScheduler with the specified parameters.
        """
        return self.method.get_class()(
            decay_rate=self.decay_rate,
            decay_steps=self.decay_steps,
        )


@dataclass
class OptimizerDataType:
    """Data class to hold optimizer data."""

    sgd: SGDOptimizerDataType | None
    adam: AdamOptimizerDataType | None
    learning_rate_scheduler: LearningRateSchedulerDataType

    @classmethod
    def from_protobuf(cls, optimizer_data: OptimizerData) -> OptimizerDataType:
        """Create a OptimizerDataType instance from Protobuf.

        :param OptimizerData optimizer_data:
            The Protobuf OptimizerData message.
        :return OptimizerDataType:
            The corresponding OptimizerDataType instance.
        """
        which_oneof = optimizer_data.WhichOneof("algorithm")
        match which_oneof:
            case "sgd":
                return cls(
                    sgd=SGDOptimizerDataType.from_protobuf(optimizer_data.sgd),
                    adam=None,
                    learning_rate_scheduler=LearningRateSchedulerDataType.from_protobuf(
                        optimizer_data.learning_rate_scheduler
                    ),
                )
            case "adam":
                return cls(
                    sgd=None,
                    adam=AdamOptimizerDataType.from_protobuf(optimizer_data.adam),
                    learning_rate_scheduler=LearningRateSchedulerDataType.from_protobuf(
                        optimizer_data.learning_rate_scheduler
                    ),
                )
            case _:
                msg = "OptimizerData must contain either SGD or Adam optimizer data."
                raise ValueError(msg)

    @staticmethod
    def to_protobuf(optimizer_data: OptimizerDataType) -> OptimizerData:
        """Convert OptimizerDataType to Protobuf.

        :param OptimizerDataType optimizer_data:
            The OptimizerDataType instance.
        :return OptimizerData:
            The corresponding Protobuf OptimizerData message.
        """
        if optimizer_data.sgd:
            return OptimizerData(
                sgd=SGDOptimizerDataType.to_protobuf(optimizer_data.sgd),
                adam=None,
                learning_rate_scheduler=LearningRateSchedulerDataType.to_protobuf(
                    optimizer_data.learning_rate_scheduler
                ),
            )
        if optimizer_data.adam:
            return OptimizerData(
                sgd=None,
                adam=AdamOptimizerDataType.to_protobuf(optimizer_data.adam),
                learning_rate_scheduler=LearningRateSchedulerDataType.to_protobuf(
                    optimizer_data.learning_rate_scheduler
                ),
            )

        msg = "OptimizerDataType must contain either SGD or Adam optimizer data."
        raise ValueError(msg)

    @classmethod
    def from_bytes(cls, optimizer_data: bytes) -> OptimizerDataType:
        """Create a OptimizerDataType instance from Protobuf bytes.

        :param bytes optimizer_data:
            The Protobuf-serialized OptimizerData bytes.
        :return OptimizerDataType:
            The corresponding OptimizerDataType instance.
        """
        optimizer = OptimizerData()
        optimizer.ParseFromString(optimizer_data)
        return cls.from_protobuf(optimizer)

    @staticmethod
    def to_bytes(optimizer_data: OptimizerDataType) -> bytes:
        """Convert OptimizerDataType to Protobuf bytes.

        :param OptimizerDataType optimizer_data:
            The OptimizerDataType instance.
        :return bytes:
            The Protobuf-serialized OptimizerData bytes.
        """
        optimizer = OptimizerDataType.to_protobuf(optimizer_data)
        return optimizer.SerializeToString()  # type: ignore[no-any-return]

    @classmethod
    def from_class_instance(cls, optimizer: Optimizer) -> OptimizerDataType:
        """Create an OptimizerDataType instance from an optimizer class instance.

        :param Optimizer optimizer:
            The optimizer class instance.
        :return OptimizerDataType:
            The corresponding OptimizerDataType instance.
        """
        if isinstance(optimizer, SGDOptimizer):
            return cls(
                sgd=SGDOptimizerDataType.from_class_instance(optimizer),
                adam=None,
                learning_rate_scheduler=LearningRateSchedulerDataType.from_class_instance(optimizer.lr_scheduler),
            )
        if isinstance(optimizer, AdamOptimizer):
            return cls(
                sgd=None,
                adam=AdamOptimizerDataType.from_class_instance(optimizer),
                learning_rate_scheduler=LearningRateSchedulerDataType.from_class_instance(optimizer.lr_scheduler),
            )

        msg = "Optimizer must be an instance of SGDOptimizer or AdamOptimizer."
        raise ValueError(msg)

    def get_class_instance(self) -> SGDOptimizer | AdamOptimizer:
        """Return an instance of the optimizer based on the stored data.

        :param OptimizerDataType optimizer_data:
            The OptimizerDataType instance.
        :return SGDOptimizer | AdamOptimizer:
            An instance of the specified optimizer.
        """
        if self.sgd:
            return SGDOptimizer(
                lr=self.sgd.learning_rate, lr_scheduler=self.learning_rate_scheduler.get_class_instance()
            )
        if self.adam:
            return AdamOptimizer(
                lr=self.adam.learning_rate,
                lr_scheduler=self.learning_rate_scheduler.get_class_instance(),
                beta1=self.adam.beta1,
                beta2=self.adam.beta2,
                epsilon=self.adam.epsilon,
            )

        msg = "OptimizerDataType must contain either SGD or Adam optimizer data."
        raise ValueError(msg)


@dataclass
class NeuralNetworkDataType:
    """Data class to hold neural network data."""

    num_inputs: int
    hidden_layer_sizes: list[int]
    num_outputs: int
    input_activation: ActivationFunctionEnum
    hidden_activation: ActivationFunctionEnum
    output_activation: ActivationFunctionEnum
    weights: list[MatrixDataType]
    biases: list[MatrixDataType]
    optimizer: OptimizerDataType = field(default_factory=OptimizerDataType)

    @classmethod
    def from_protobuf(cls, nn_data: NeuralNetworkData) -> NeuralNetworkDataType:
        """Create a NeuralNetworkDataType instance from Protobuf.

        :param NeuralNetworkData nn_data:
            The Protobuf NeuralNetworkData message.
        :return NeuralNetworkDataType:
            The corresponding NeuralNetworkDataType instance.
        """
        return cls(
            num_inputs=nn_data.num_inputs,
            hidden_layer_sizes=list(nn_data.hidden_layer_sizes),
            num_outputs=nn_data.num_outputs,
            input_activation=ActivationFunctionEnum.from_protobuf(nn_data.input_activation),
            hidden_activation=ActivationFunctionEnum.from_protobuf(nn_data.hidden_activation),
            output_activation=ActivationFunctionEnum.from_protobuf(nn_data.output_activation),
            weights=[MatrixDataType.from_protobuf(weight) for weight in nn_data.weights],
            biases=[MatrixDataType.from_protobuf(bias) for bias in nn_data.biases],
            optimizer=OptimizerDataType.from_protobuf(nn_data.optimizer),
        )

    @staticmethod
    def to_protobuf(config_data: NeuralNetworkDataType) -> NeuralNetworkData:
        """Convert NeuralNetworkDataType to Protobuf.

        :param NeuralNetworkDataType config_data:
            The NeuralNetworkDataType instance.
        :return NeuralNetworkData:
            The corresponding Protobuf NeuralNetworkData message.
        """
        return NeuralNetworkData(
            num_inputs=config_data.num_inputs,
            hidden_layer_sizes=config_data.hidden_layer_sizes,
            num_outputs=config_data.num_outputs,
            input_activation=ActivationFunctionEnum.to_protobuf(config_data.input_activation),
            hidden_activation=ActivationFunctionEnum.to_protobuf(config_data.hidden_activation),
            output_activation=ActivationFunctionEnum.to_protobuf(config_data.output_activation),
            weights=[MatrixDataType.to_protobuf(weight) for weight in config_data.weights],
            biases=[MatrixDataType.to_protobuf(bias) for bias in config_data.biases],
            optimizer=OptimizerDataType.to_protobuf(config_data.optimizer),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuralNetworkDataType:
        """Create a NeuralNetworkDataType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf-serialized NeuralNetworkData bytes.
        :return NeuralNetworkDataType:
            The corresponding NeuralNetworkDataType instance.
        """
        config = NeuralNetworkData()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuralNetworkDataType) -> bytes:
        """Convert NeuralNetworkDataType to Protobuf bytes.

        :param NeuralNetworkDataType config_data:
            The NeuralNetworkDataType instance.
        :return bytes:
            The Protobuf-serialized NeuralNetworkData bytes.
        """
        config = NeuralNetworkDataType.to_protobuf(config_data)
        return config.SerializeToString()  # type: ignore[no-any-return]
