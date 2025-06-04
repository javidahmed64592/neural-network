from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from neural_network.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.math.activation_functions import LinearActivation, SigmoidActivation
from neural_network.math.matrix import Matrix
from neural_network.math.nn_math import calculate_error_from_expected, calculate_next_errors


class NeuralNetwork:
    """
    This class can be used to create a NeuralNetwork with specified layer sizes. This neural network can feedforward a
    list of inputs, and be trained through backpropagation of errors.
    """

    def __init__(
        self,
        input_layer: InputLayer,
        output_layer: OutputLayer,
        hidden_layers: list[HiddenLayer] | None = None,
        lr: float = 0.1,
    ) -> None:
        """
        Initialise NeuralNetwork with a list of Layers and a learning rate.

        Parameters:
            layers (list[Layer]): NeuralNetwork layers
            lr (float): Learning rate for training, defaults to 0.1
        """
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._hidden_layers = hidden_layers or []
        for index in range(1, len(self.layers)):
            self.layers[index].set_prev_layer(self.layers[index - 1])

        self._num_inputs = self._input_layer.size
        self._num_outputs = self._output_layer.size
        self._hidden_layer_sizes = [layer.size for layer in self._hidden_layers]
        self._lr = lr

    def __str__(self) -> str:
        return "NeuralNetwork:\n" + "\n".join([str(layer) for layer in self.layers])

    @classmethod
    def from_layers(cls, layers: list[Layer], lr: float = 0.1) -> NeuralNetwork:
        """
        Create a NeuralNetwork from a list of layers.

        Parameters:
            layers (list[Layer]): NeuralNetwork layers
        """
        input_layer = cast(InputLayer, layers[0])
        output_layer = cast(OutputLayer, layers[-1])
        hidden_layers = cast(list[HiddenLayer], layers[1:-1])

        return cls(input_layer=input_layer, output_layer=output_layer, hidden_layers=hidden_layers, lr=lr)

    @classmethod
    def from_file(cls, filepath: str) -> NeuralNetwork:
        """
        Load neural network layer with weights and biases from JSON file.

        Parameters:
            filepath (str): Path to file with neural network data
        """
        with open(filepath) as file:
            _data = json.load(file)

        input_layer = InputLayer(size=_data["num_inputs"], activation=LinearActivation)
        output_layer = OutputLayer(
            size=_data["num_outputs"], activation=SigmoidActivation, weights_range=(-1, 1), bias_range=(-1, 1)
        )
        hidden_layers = [
            HiddenLayer(size=size, activation=SigmoidActivation, weights_range=(-1, 1), bias_range=(-1, 1))
            for size in _data["hidden_layer_sizes"]
        ]

        nn = cls(input_layer, output_layer, hidden_layers)
        nn.weights = [Matrix.from_array(weights) for weights in _data["weights"]]
        nn.bias = [Matrix.from_array(bias) for bias in _data["bias"]]
        return nn

    @property
    def layers(self) -> list[Layer]:
        return [self._input_layer, *self._hidden_layers, self._output_layer]

    @property
    def weights(self) -> list[Matrix]:
        return [layer.weights for layer in self.layers]

    @weights.setter
    def weights(self, new_weights: list[Matrix]) -> None:
        for layer, weights in zip(self.layers, new_weights, strict=False):
            layer.weights = weights

    @property
    def bias(self) -> list[Matrix]:
        return [layer.bias for layer in self.layers]

    @bias.setter
    def bias(self, new_bias: list[Matrix]) -> None:
        for layer, bias in zip(self.layers, new_bias, strict=False):
            layer.bias = bias

    def save(self, filepath: str) -> None:
        """
        Save neural network layer weights and biases to JSON file.

        Parameters:
            filepath (str): Path to file with weights and biases
        """
        _data = {
            "num_inputs": self._num_inputs,
            "num_outputs": self._num_outputs,
            "hidden_layer_sizes": self._hidden_layer_sizes,
            "weights": [weights.vals.tolist() for weights in self.weights],
            "bias": [bias.vals.tolist() for bias in self.bias],
        }
        with open(filepath, "w") as file:
            json.dump(_data, file)

    def feedforward(self, inputs: list[float]) -> list[float]:
        """
        Feedforward a list of inputs.

        Parameters:
            inputs (list[float]): List of input values

        Returns:
            output (list[float]): List of outputs
        """
        vals = self._input_layer.feedforward(Matrix.from_array(inputs))

        for layer in self._hidden_layers:
            vals = layer.feedforward(vals)

        output = self._output_layer.feedforward(vals)
        output = Matrix.transpose(output)
        return output.as_list

    def train(self, inputs: list[float], expected_outputs: list[float]) -> list[float]:
        """
        Train NeuralNetwork using a list of input values and expected output values and backpropagate errors.

        Parameters:
            inputs (list[float]): List of input values
            expected_outputs (list[float]): List of output values

        Returns:
            output_errors (list[float]): List of output errors
        """
        vals = Matrix.from_array(inputs)

        for layer in self.layers:
            vals = layer.feedforward(vals)

        expected_output_matrix = Matrix.from_array(expected_outputs)
        errors = calculate_error_from_expected(expected_output_matrix, vals)
        self._output_layer.backpropagate_error(errors, self._lr)
        output_errors = Matrix.transpose(errors)

        for layer in self._hidden_layers[::-1]:
            if next_layer := layer._next_layer:
                errors = calculate_next_errors(next_layer.weights, errors)
                layer.backpropagate_error(errors, self._lr)

        return output_errors.as_list

    def train_with_fitness(self, inputs: list[float], fitness: float, prev_fitness: float) -> list[float]:
        """
        Train the neural network using fitness values.

        Parameters:
            inputs (list[float]): List of input values
            fitness (float): Fitness value for the current generation
            prev_fitness (float): Fitness value for the previous generation

        Returns:
            output_errors (list[float]): List of output errors
        """
        vals = Matrix.from_array(inputs)

        for layer in self.layers:
            vals = layer.feedforward(vals)

        fitness_error = fitness - prev_fitness
        errors = vals * fitness_error
        self._output_layer.backpropagate_error(errors, self._lr)
        output_errors = Matrix.transpose(errors)

        for layer in self._hidden_layers[::-1]:
            if next_layer := layer._next_layer:
                errors = calculate_next_errors(next_layer.weights, errors)
                layer.backpropagate_error(errors, self._lr)

        return output_errors.as_list

    def run_supervised_training(
        self,
        inputs: list[list[float]],
        expected_outputs: list[list[float]],
        epochs: int = 1,
    ) -> None:
        """
        Train the neural network using supervised learning.

        Parameters:
            inputs (list[list[float]]): List of input values
            expected_outputs (list[list[float]]): List of expected output values
            epochs (int): Number of training epochs, defaults to 1
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
        """
        Train the neural network using fitness values.

        Parameters:
            inputs (list[list[float]]): List of input values
            fitnesses (list[float]): List of fitness values for each input
            epochs (int): Number of training epochs, defaults to 1
            alpha (float): Smoothing factor for fitness values, defaults to 0.1
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
        """
        Crossover two Neural Networks by mixing their weights and biases, matching the topology of the instance of this
        class.

        Parameters:
            nn (NeuralNetwork): Neural Network to use for average weights and biases
            other_nn (NeuralNetwork): Other Neural Network to use for average weights and biases
            weights_crossover_func (Callable): Custom function for crossover operations for layer weights
            bias_crossover_func (Callable): Custom function for crossover operations for layer biases
                Should accept (element, other_element, roll) and return a float

        Returns:
            new_weights, new_biases (tuple[list[Matrix], list[Matrix]]): New Layer weights and biases
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
