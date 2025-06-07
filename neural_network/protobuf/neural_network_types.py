from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunctionData, MatrixData, NeuralNetworkData


class ActivationFunctionEnum(IntEnum):
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

    def get_class(self) -> type:
        """Returns the corresponding activation function class."""
        _map: dict[ActivationFunctionEnum, type] = {
            ActivationFunctionEnum.LINEAR: LinearActivation,
            ActivationFunctionEnum.RELU: ReluActivation,
            ActivationFunctionEnum.SIGMOID: SigmoidActivation,
            ActivationFunctionEnum.TANH: TanhActivation,
        }
        return _map[self]

    @classmethod
    def from_protobuf(cls, proto_enum_value: ActivationFunctionData) -> ActivationFunctionEnum:
        """Maps a Protobuf ActivationFunction value to ActivationFunctionEnum."""
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnum) -> ActivationFunctionData:
        """Maps an ActivationFunctionEnum value to Protobuf ActivationFunction."""
        return ActivationFunctionData.Value(enum_value.name)  # type: ignore


@dataclass
class MatrixDataType:
    """Data class to hold genetic algorithm configuration."""

    data: list[float]
    rows: int
    cols: int

    @classmethod
    def from_protobuf(cls, matrix_data: MatrixData) -> MatrixDataType:
        """Creates a MatrixDataType instance from Protobuf."""
        return cls(
            data=list(matrix_data.data),
            rows=matrix_data.rows,
            cols=matrix_data.cols,
        )

    @staticmethod
    def to_protobuf(matrix_data: MatrixDataType) -> MatrixData:
        """Converts MatrixDataType to Protobuf."""
        return MatrixData(
            data=matrix_data.data,
            rows=matrix_data.rows,
            cols=matrix_data.cols,
        )

    @classmethod
    def from_bytes(cls, matrix_data: bytes) -> MatrixDataType:
        """Creates a MatrixDataType instance from Protobuf bytes."""
        matrix = MatrixData()
        matrix.ParseFromString(matrix_data)
        return cls.from_protobuf(matrix)

    @staticmethod
    def to_bytes(matrix_data: MatrixDataType) -> bytes:
        """Converts MatrixDataType to Protobuf bytes."""
        matrix = MatrixDataType.to_protobuf(matrix_data)
        return matrix.SerializeToString()
