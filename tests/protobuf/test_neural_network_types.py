from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from neural_network.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunctionData
from neural_network.protobuf.neural_network_types import ActivationFunctionEnum


class TestActivationFunctionEnum:
    def test_get_class(self) -> None:
        assert ActivationFunctionEnum.LINEAR.get_class() == LinearActivation
        assert ActivationFunctionEnum.RELU.get_class() == ReluActivation
        assert ActivationFunctionEnum.SIGMOID.get_class() == SigmoidActivation
        assert ActivationFunctionEnum.TANH.get_class() == TanhActivation

    def test_from_protobuf(self) -> None:
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.LINEAR) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.RELU) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.SIGMOID) == ActivationFunctionEnum.SIGMOID
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.TANH) == ActivationFunctionEnum.TANH

    def test_to_protobuf(self) -> None:
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.LINEAR) == ActivationFunctionData.LINEAR
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.RELU) == ActivationFunctionData.RELU
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.SIGMOID) == ActivationFunctionData.SIGMOID
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.TANH) == ActivationFunctionData.TANH
