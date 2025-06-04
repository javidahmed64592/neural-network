__version__ = "1.15.0"

from neural_network.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
)
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork

# Optionally define what gets imported with "from neural_network import *"
__all__ = [
    "ActivationFunction",
    "HiddenLayer",
    "InputLayer",
    "Layer",
    "LinearActivation",
    "Matrix",
    "NeuralNetwork",
    "OutputLayer",
    "ReluActivation",
    "SigmoidActivation",
]
