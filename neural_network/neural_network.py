"""Neural network class for feedforward and training operations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from neural_network.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.math.matrix import Matrix
from neural_network.math.nn_math import calculate_error_from_expected, calculate_next_errors
from neural_network.math.optimizer import Optimizer, SGDOptimizer
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    MatrixDataType,
    NeuralNetworkDataType,
    OptimizerDataType,
)


class NeuralNetwork:
    """Class representing a feedforward neural network with training and serialization capabilities."""

    def __init__(
        self,
        input_layer: InputLayer,
        output_layer: OutputLayer,
        hidden_layers: list[HiddenLayer] | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        """Initialise NeuralNetwork with a list of Layers and a learning rate.

        :param InputLayer input_layer:
            Input layer of the neural network.
        :param OutputLayer output_layer:
            Output layer of the neural network.
        :param list[HiddenLayer] | None hidden_layers:
            List of hidden layers (optional).
        :param Optimizer | None optimizer:
            Optimizer for training (defaults to SGD). Each layer gets its own instance.
        """
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._hidden_layers = hidden_layers or []
        for index in range(1, len(self.layers)):
            self.layers[index].set_prev_layer(self.layers[index - 1])

        self._optimizer = optimizer or SGDOptimizer(learning_rate=0.1)
        optimizer_class = self._optimizer.__class__

        for layer in self.layers:
            optimizer_instance = optimizer_class(**dict(self._optimizer.__dict__.items()))
            layer.set_optimizer(optimizer_instance)

        self._num_inputs = self._input_layer.size
        self._num_outputs = self._output_layer.size
        self._hidden_layer_sizes = [layer.size for layer in self._hidden_layers]

    def __str__(self) -> str:
        """Return a string representation of the NeuralNetwork.

        :return str:
            String describing the neural network and its layers.
        """
        return "NeuralNetwork:\n" + "\n".join([str(layer) for layer in self.layers])

    @classmethod
    def from_layers(
        cls,
        layers: list[Layer],
        optimizer: Optimizer | None = None,
    ) -> NeuralNetwork:
        """Create a NeuralNetwork from a list of layers.

        :param list[Layer] layers:
            List of layers for the neural network.
        :param Optimizer | None optimizer:
            Optimizer for training (defaults to SGD). Each layer gets its own instance.
        :return NeuralNetwork:
            NeuralNetwork instance.
        """
        input_layer = cast(InputLayer, layers[0])
        output_layer = cast(OutputLayer, layers[-1])
        hidden_layers = cast(list[HiddenLayer], layers[1:-1])

        return cls(input_layer=input_layer, output_layer=output_layer, hidden_layers=hidden_layers, optimizer=optimizer)

    @classmethod
    def from_protobuf(cls, nn_data: NeuralNetworkDataType) -> NeuralNetwork:
        """Create a NeuralNetwork from Protobuf data.

        :param NeuralNetworkDataType nn_data:
            Neural network data from Protobuf.
        :return NeuralNetwork:
            NeuralNetwork instance.
        """
        input_layer = InputLayer(
            size=nn_data.num_inputs, activation=ActivationFunctionEnum(nn_data.input_activation).get_class()
        )
        output_layer = OutputLayer(
            size=nn_data.num_outputs,
            activation=ActivationFunctionEnum(nn_data.output_activation).get_class(),
            weights_range=(-1, 1),
            bias_range=(-1, 1),
        )
        hidden_layers = [
            HiddenLayer(
                size=size,
                activation=ActivationFunctionEnum(nn_data.hidden_activation).get_class(),
                weights_range=(-1, 1),
                bias_range=(-1, 1),
            )
            for size in nn_data.hidden_layer_sizes
        ]

        nn = cls(
            input_layer=input_layer,
            output_layer=output_layer,
            hidden_layers=hidden_layers,
            optimizer=nn_data.optimizer.get_class_instance(),
        )
        nn.weights = [MatrixDataType.to_matrix(weights) for weights in nn_data.weights]
        nn.bias = [MatrixDataType.to_matrix(bias) for bias in nn_data.biases]
        return nn

    @staticmethod
    def to_protobuf(nn: NeuralNetwork) -> NeuralNetworkDataType:
        """Convert a NeuralNetwork instance to Protobuf data.

        :param NeuralNetwork nn:
            Neural network instance.
        :return NeuralNetworkDataType:
            Protobuf data containing neural network information.
        """
        return NeuralNetworkDataType(
            num_inputs=nn._num_inputs,
            hidden_layer_sizes=nn._hidden_layer_sizes,
            num_outputs=nn._num_outputs,
            input_activation=ActivationFunctionEnum.from_class(nn._input_layer._activation),
            hidden_activation=ActivationFunctionEnum.from_class(nn._hidden_layers[0]._activation),
            output_activation=ActivationFunctionEnum.from_class(nn._output_layer._activation),
            weights=[MatrixDataType.to_protobuf(MatrixDataType.from_matrix(weights)) for weights in nn.weights],
            biases=[MatrixDataType.to_protobuf(MatrixDataType.from_matrix(bias)) for bias in nn.bias],
            optimizer=OptimizerDataType.from_class_instance(nn._input_layer._optimizer),
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> NeuralNetwork:
        """Load a NeuralNetwork from a file.

        :param str file_path:
            Path to the file containing the neural network data.
        :return NeuralNetwork:
            Loaded neural network instance.
        """
        with open(file_path, "rb") as file:
            data = file.read()
        nn_data = NeuralNetworkDataType.from_bytes(data)
        return cls.from_protobuf(nn_data)

    @staticmethod
    def save_to_file(nn: NeuralNetwork, filename: str, directory: Path) -> None:
        """Save a NeuralNetwork to a file.

        :param NeuralNetwork nn:
            Neural network instance to save.
        :param str filename:
            Name of the file where the neural network data will be saved.
        :param Path directory:
            Directory where the file will be saved.
        """
        nn_data = NeuralNetwork.to_protobuf(nn)
        file_path = directory / f"{filename}.pb"
        with open(file_path, "wb") as file:
            file.write(NeuralNetworkDataType.to_bytes(nn_data))

    @property
    def layers(self) -> list[Layer]:
        """Return the list of layers in the neural network.

        :return list[Layer]:
            List of all layers (input, hidden, output).
        """
        return [self._input_layer, *self._hidden_layers, self._output_layer]

    @property
    def weights(self) -> list[Matrix]:
        """Return the list of weights matrices for all layers.

        :return list[Matrix]:
            List of weights matrices.
        """
        return [layer.weights for layer in self.layers]

    @weights.setter
    def weights(self, new_weights: list[Matrix]) -> None:
        """Set the weights matrices for all layers.

        :param list[Matrix] new_weights:
            List of new weights matrices.
        """
        for layer, weights in zip(self.layers, new_weights, strict=False):
            layer.weights = weights

    @property
    def bias(self) -> list[Matrix]:
        """Return the list of bias matrices for all layers.

        :return list[Matrix]:
            List of bias matrices.
        """
        return [layer.bias for layer in self.layers]

    @bias.setter
    def bias(self, new_bias: list[Matrix]) -> None:
        """Set the bias matrices for all layers.

        :param list[Matrix] new_bias:
            List of new bias matrices.
        """
        for layer, bias in zip(self.layers, new_bias, strict=False):
            layer.bias = bias

    def feedforward(self, inputs: list[float]) -> list[float]:
        """Feedforward a list of input values through the network.

        :param list[float] inputs:
            List of input values.
        :return list[float]:
            List of output values.
        """
        vals = self._input_layer.feedforward(Matrix.from_array(inputs))

        for layer in self._hidden_layers:
            vals = layer.feedforward(vals)

        output = self._output_layer.feedforward(vals)
        output = Matrix.transpose(output)
        return output.as_list

    def train(self, inputs: list[float], expected_outputs: list[float]) -> list[float]:
        """Train the neural network using input and expected output values, and backpropagate errors.

        :param list[float] inputs:
            List of input values.
        :param list[float] expected_outputs:
            List of expected output values.
        :return list[float]:
            List of output errors.
        """
        vals = Matrix.from_array(inputs)

        for layer in self.layers:
            vals = layer.feedforward(vals)
            expected_output_matrix = Matrix.from_array(expected_outputs)
        errors = calculate_error_from_expected(expected_output_matrix, vals)
        self._output_layer.backpropagate_error(errors)
        output_errors = Matrix.transpose(errors)

        for layer in self._hidden_layers[::-1]:
            if next_layer := layer._next_layer:
                errors = calculate_next_errors(next_layer.weights, errors)
                layer.backpropagate_error(errors)

        return output_errors.as_list

    def train_with_fitness(self, inputs: list[float], fitness: float, prev_fitness: float) -> list[float]:
        """Train the neural network using fitness values.

        :param list[float] inputs:
            List of input values.
        :param float fitness:
            Fitness value for the current generation.
        :param float prev_fitness:
            Fitness value for the previous generation.
        :return list[float]:
            List of output errors.
        """
        vals = Matrix.from_array(inputs)

        for layer in self.layers:
            vals = layer.feedforward(vals)

        fitness_error = fitness - prev_fitness
        errors = vals * fitness_error

        self._output_layer.backpropagate_error(errors)
        output_errors = Matrix.transpose(errors)

        for layer in self._hidden_layers[::-1]:
            if next_layer := layer._next_layer:
                errors = calculate_next_errors(next_layer.weights, errors)
                layer.backpropagate_error(errors)

        return output_errors.as_list

    def run_supervised_training(
        self,
        inputs: list[list[float]],
        expected_outputs: list[list[float]],
        epochs: int = 1,
    ) -> None:
        """Train the neural network using supervised learning.

        :param list[list[float]] inputs:
            List of input value lists.
        :param list[list[float]] expected_outputs:
            List of expected output value lists.
        :param int epochs:
            Number of training epochs (default 1).
        """
        for _ in range(epochs):
            for input_data, expected_output in zip(inputs, expected_outputs, strict=False):
                self.train(input_data, expected_output)

    def run_fitness_training(
        self,
        inputs: list[list[float]],
        fitnesses: list[float],
        epochs: int = 1,
        alpha: float = 0.1,
    ) -> None:
        """Train the neural network using fitness values.

        :param list[list[float]] inputs:
            List of input value lists.
        :param list[float] fitnesses:
            List of fitness values for each input.
        :param int epochs:
            Number of training epochs (default 1).
        :param float alpha:
            Smoothing factor for fitness values (default 0.1).
        """
        prev_fitness = 0.0

        for _ in range(epochs):
            for i in range(len(inputs)):
                fitness = fitnesses[i]
                smoothed_fitness = prev_fitness * (1 - alpha) + fitness * alpha

                if fitness > prev_fitness:
                    smoothed_fitness += 0.05

                self.train_with_fitness(inputs[i], smoothed_fitness, prev_fitness)
                prev_fitness = smoothed_fitness

    @staticmethod
    def crossover(
        nn: NeuralNetwork,
        other_nn: NeuralNetwork,
        weights_crossover_func: Callable,
        bias_crossover_func: Callable,
    ) -> tuple[list[Matrix], list[Matrix]]:
        """Crossover two neural networks by mixing their weights and biases, matching the topology of the first network.

        :param NeuralNetwork nn:
            First neural network for crossover.
        :param NeuralNetwork other_nn:
            Second neural network for crossover.
        :param Callable weights_crossover_func:
            Function for crossover operations on layer weights.
        :param Callable bias_crossover_func:
            Function for crossover operations on layer biases.
        :return tuple[list[Matrix], list[Matrix]]:
            New layer weights and biases.
        """
        new_weights = [nn.weights[0]]
        new_biases = [nn.bias[0]]

        for index in range(1, len(nn.layers)):
            new_weight = Matrix.crossover(
                nn.weights[index],
                other_nn.weights[index],
                weights_crossover_func,
            )
            new_bias = Matrix.crossover(
                nn.bias[index],
                other_nn.bias[index],
                bias_crossover_func,
            )

            new_weights.append(new_weight)
            new_biases.append(new_bias)

        return (new_weights, new_biases)
