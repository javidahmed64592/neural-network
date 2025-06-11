"""Dataclasses and enums for neural network and configuration Protobuf messages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)
from neural_network.math.matrix import Matrix
from neural_network.math.optimizer import AdamOptimizer, Optimizer, SGDOptimizer
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunctionData,
    MatrixData,
    NeuralNetworkData,
    OptimizationAlgorithm,
    OptimizerData,
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


class OptimizationAlgorithmEnum(IntEnum):
    """Enum for supported activation functions."""

    SGD = 0
    ADAM = 1

    @property
    def map(self) -> dict[OptimizationAlgorithmEnum, type[Optimizer]]:
        """Return a mapping from enum to optimizer class.

        :return dict[OptimizationAlgorithmEnum, type[Optimizer]]:
            Mapping from enum to optimizer class.
        """
        return {
            OptimizationAlgorithmEnum.SGD: SGDOptimizer,
            OptimizationAlgorithmEnum.ADAM: AdamOptimizer,
        }

    def get_class(self) -> type[Optimizer]:
        """Return the corresponding optimizer class.

        :return type[Optimizer]:
            The optimizer class.
        """
        return self.map[self]

    @classmethod
    def from_class(cls, optimizer: type[Optimizer]) -> OptimizationAlgorithmEnum:
        """Return the enum value for a given optimizer class.

        :param type[Optimizer] optimizer:
            The optimizer class.
        :return OptimizationAlgorithmEnum:
            The corresponding enum value.
        """
        reverse_map = {v: k for k, v in cls.SGD.map.items()}
        return reverse_map[optimizer]

    @classmethod
    def from_protobuf(cls, proto_enum_value: OptimizationAlgorithm) -> OptimizationAlgorithmEnum:
        """Return the enum value from a Protobuf OptimizationAlgorithmEnum value.

        :param OptimizationAlgorithmEnum proto_enum_value:
            The Protobuf enum value.
        :return OptimizationAlgorithmEnum:
            The corresponding enum value.
        """
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: OptimizationAlgorithmEnum) -> OptimizationAlgorithm:
        """Return the Protobuf OptimizationAlgorithm from an enum value.

        :param OptimizationAlgorithmEnum enum_value:
            The enum value.
        :return OptimizationAlgorithm:
            The Protobuf enum value.
        """
        return OptimizationAlgorithm.Value(enum_value.name)  # type: ignore[no-any-return]


@dataclass
class OptimizerDataType:
    """Data class to hold optimizer data."""

    algorithm: OptimizationAlgorithmEnum
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    @classmethod
    def from_protobuf(cls, optimizer_data: OptimizerData) -> OptimizerDataType:
        """Create a OptimizerDataType instance from Protobuf.

        :param OptimizerData optimizer_data:
            The Protobuf OptimizerData message.
        :return OptimizerDataType:
            The corresponding OptimizerDataType instance.
        """
        return cls(
            algorithm=OptimizationAlgorithmEnum.from_protobuf(optimizer_data.algorithm),
            learning_rate=optimizer_data.learning_rate,
            beta1=optimizer_data.beta1,
            beta2=optimizer_data.beta2,
            epsilon=optimizer_data.epsilon,
        )

    @staticmethod
    def to_protobuf(optimizer_data: OptimizerDataType) -> OptimizerData:
        """Convert OptimizerDataType to Protobuf.

        :param OptimizerDataType optimizer_data:
            The OptimizerDataType instance.
        :return OptimizerData:
            The corresponding Protobuf OptimizerData message.
        """
        return OptimizerData(
            algorithm=OptimizationAlgorithmEnum.to_protobuf(optimizer_data.algorithm),
            learning_rate=optimizer_data.learning_rate,
            beta1=optimizer_data.beta1,
            beta2=optimizer_data.beta2,
            epsilon=optimizer_data.epsilon,
        )

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
    learning_rate: float

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
            weights=nn_data.weights,
            biases=nn_data.biases,
            learning_rate=nn_data.learning_rate,
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
            learning_rate=config_data.learning_rate,
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
