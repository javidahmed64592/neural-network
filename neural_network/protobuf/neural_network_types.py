"""Dataclasses and enums for neural network and configuration Protobuf messages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunctionData, MatrixData, NeuralNetworkData


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
