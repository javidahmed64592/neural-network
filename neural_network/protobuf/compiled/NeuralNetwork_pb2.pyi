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

class LearningRateMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STEP_DECAY: _ClassVar[LearningRateMethod]
    EXPONENTIAL_DECAY: _ClassVar[LearningRateMethod]
LINEAR: ActivationFunctionData
RELU: ActivationFunctionData
SIGMOID: ActivationFunctionData
TANH: ActivationFunctionData
STEP_DECAY: LearningRateMethod
EXPONENTIAL_DECAY: LearningRateMethod

class MatrixData(_message.Message):
    __slots__ = ("data", "rows", "cols")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    rows: int
    cols: int
    def __init__(self, data: _Optional[_Iterable[float]] = ..., rows: _Optional[int] = ..., cols: _Optional[int] = ...) -> None: ...

class SGDOptimizerData(_message.Message):
    __slots__ = ("learning_rate",)
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    def __init__(self, learning_rate: _Optional[float] = ...) -> None: ...

class AdamOptimizerData(_message.Message):
    __slots__ = ("learning_rate", "beta1", "beta2", "epsilon")
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    BETA1_FIELD_NUMBER: _ClassVar[int]
    BETA2_FIELD_NUMBER: _ClassVar[int]
    EPSILON_FIELD_NUMBER: _ClassVar[int]
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    def __init__(self, learning_rate: _Optional[float] = ..., beta1: _Optional[float] = ..., beta2: _Optional[float] = ..., epsilon: _Optional[float] = ...) -> None: ...

class LearningRateSchedulerData(_message.Message):
    __slots__ = ("decay_rate", "decay_steps", "method")
    DECAY_RATE_FIELD_NUMBER: _ClassVar[int]
    DECAY_STEPS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    decay_rate: float
    decay_steps: int
    method: LearningRateMethod
    def __init__(self, decay_rate: _Optional[float] = ..., decay_steps: _Optional[int] = ..., method: _Optional[_Union[LearningRateMethod, str]] = ...) -> None: ...

class OptimizerData(_message.Message):
    __slots__ = ("sgd", "adam", "learning_rate_scheduler")
    SGD_FIELD_NUMBER: _ClassVar[int]
    ADAM_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_SCHEDULER_FIELD_NUMBER: _ClassVar[int]
    sgd: SGDOptimizerData
    adam: AdamOptimizerData
    learning_rate_scheduler: LearningRateSchedulerData
    def __init__(self, sgd: _Optional[_Union[SGDOptimizerData, _Mapping]] = ..., adam: _Optional[_Union[AdamOptimizerData, _Mapping]] = ..., learning_rate_scheduler: _Optional[_Union[LearningRateSchedulerData, _Mapping]] = ...) -> None: ...

class NeuralNetworkData(_message.Message):
    __slots__ = ("num_inputs", "hidden_layer_sizes", "num_outputs", "input_activation", "hidden_activation", "output_activation", "weights", "biases", "optimizer")
    NUM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_LAYER_SIZES_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    INPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    BIASES_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_FIELD_NUMBER: _ClassVar[int]
    num_inputs: int
    hidden_layer_sizes: _containers.RepeatedScalarFieldContainer[int]
    num_outputs: int
    input_activation: ActivationFunctionData
    hidden_activation: ActivationFunctionData
    output_activation: ActivationFunctionData
    weights: _containers.RepeatedCompositeFieldContainer[MatrixData]
    biases: _containers.RepeatedCompositeFieldContainer[MatrixData]
    optimizer: OptimizerData
    def __init__(self, num_inputs: _Optional[int] = ..., hidden_layer_sizes: _Optional[_Iterable[int]] = ..., num_outputs: _Optional[int] = ..., input_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., hidden_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., output_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., weights: _Optional[_Iterable[_Union[MatrixData, _Mapping]]] = ..., biases: _Optional[_Iterable[_Union[MatrixData, _Mapping]]] = ..., optimizer: _Optional[_Union[OptimizerData, _Mapping]] = ...) -> None: ...
