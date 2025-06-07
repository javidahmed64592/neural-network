from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActivationFunctionData(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[ActivationFunctionData]
    RELU: _ClassVar[ActivationFunctionData]
    SIGMOID: _ClassVar[ActivationFunctionData]
    TANH: _ClassVar[ActivationFunctionData]
LINEAR: ActivationFunctionData
RELU: ActivationFunctionData
SIGMOID: ActivationFunctionData
TANH: ActivationFunctionData

class MatrixData(_message.Message):
    __slots__ = ("data", "rows", "cols")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    rows: int
    cols: int
    def __init__(self, data: _Optional[_Iterable[float]] = ..., rows: _Optional[int] = ..., cols: _Optional[int] = ...) -> None: ...

class NeuralNetworkData(_message.Message):
    __slots__ = ("num_inputs", "hidden_layer_sizes", "num_outputs", "input_activation", "hidden_activation", "output_activation", "weights", "biases", "learning_rate")
    NUM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_LAYER_SIZES_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    INPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    BIASES_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    num_inputs: int
    hidden_layer_sizes: _containers.RepeatedScalarFieldContainer[int]
    num_outputs: int
    input_activation: ActivationFunctionData
    hidden_activation: ActivationFunctionData
    output_activation: ActivationFunctionData
    weights: MatrixData
    biases: MatrixData
    learning_rate: float
    def __init__(self, num_inputs: _Optional[int] = ..., hidden_layer_sizes: _Optional[_Iterable[int]] = ..., num_outputs: _Optional[int] = ..., input_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., hidden_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., output_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., weights: _Optional[_Union[MatrixData, _Mapping]] = ..., biases: _Optional[_Union[MatrixData, _Mapping]] = ..., learning_rate: _Optional[float] = ...) -> None: ...
