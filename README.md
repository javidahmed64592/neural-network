[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=ffd343)](https://docs.python.org/3.12/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- omit from toc -->
# Neural Network
This is a neural network library in Python which can be used to feedforward arrays of inputs, generate outputs, and be trained with expected outputs.

Install this package using `pip`:

    pip install -e git+https://github.com/javidahmed64592/neural-network#egg=neural_network

To update the package:

    pip update -e git+https://github.com/javidahmed64592/neural-network#egg=neural_network

_Note: It is recommended to install this into a virtual environment._

<!-- omit from toc -->
## Table of Contents
- [Installing Dependencies](#installing-dependencies)
- [Using the Neural Network](#using-the-neural-network)
  - [Creating a Neural Network](#creating-a-neural-network)
  - [Training a Neural Network](#training-a-neural-network)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Neuroevolution](#neuroevolution)
- [Testing](#testing)
- [Linting and Formatting](#linting-and-formatting)
- [Type Checking](#type-checking)
- [License](#license)

## Installing Dependencies
Install the required dependencies using `pip`:

    pip install -e .

To install with `dev` dependencies:

    pip install -e .[dev]

## Using the Neural Network
For a complete example of how to create and train the neural network, see `example_training.ipynb` where it is trained on binary lists.

### Creating a Neural Network
The neural network can be created in the following way:

```python
from neural_network.math.activation_functions import LinearActivation, SigmoidActivation
from neural_network.neural_network import NeuralNetwork
from neural_network.layer import HiddenLayer, InputLayer, OutputLayer


input_layer = InputLayer(size=num_inputs, activation=LinearActivation)
hidden_layers = [
    HiddenLayer(size=size, activation=SigmoidActivation, weights_range=[-1, 1], bias_range=[-1, 1])
    for size in hidden_layer_sizes
]
output_layer = OutputLayer(size=num_outputs, activation=SigmoidActivation, weights_range=[-1, 1], bias_range=[-1, 1])

nn = NeuralNetwork.from_layers(layers=[input_layer, *hidden_layers, output_layer])
```

where

- `num_inputs`: Number of inputs to pass through neural network
- `num_outputs`: Number of outputs to be generated
- `hidden_layer_sizes`: List of number of nodes in each hidden layer

### Training a Neural Network
To feedforward an array of inputs:

```python
outputs = nn.feedforward([x_i, ..., x_n]) # n: Number of inputs
```

The neural network can also be trained by providing an array of inputs and expected outputs, and backpropagating the error using gradient descent.

```python
inputs = [x_i, ..., x_n] # n: Number of inputs
expected_outputs = [y_i, ..., y_m] # m: Number of outputs

errors = nn.train(inputs, expected_outputs)
```

### Saving and Loading Models
The neural network weights and biases can be saved to a `json` file:

```python
nn.save("/path/to/nn_model.json")
```

To load a neural network from a file:

```python
nn = NeuralNetwork.from_file("/path/to/nn_model.json")
```

### Neuroevolution
New weights and biases can also be calculated via crossover:

```python
nn_1 = NeuralNetwork(...)
nn_2 = NeuralNetwork(...)
nn_3 = NeuralNetwork(...)

nn_1.weights, nn_1.bias = nn_1.crossover(nn_2, nn_3, mutation_rate)
```

where

- `mutation_rate`: Percentage of weights and biases to be randomised

## Testing
This library uses Pytest for the unit tests.
These tests are located in the `tests` directory.
To run the tests:

    python -m pytest tests

## Linting and Formatting
This library uses `ruff` for linting and formatting.
This is configured in `pyproject.toml`.

To check the code for linting errors:

    python -m ruff check .

To format the code:

    python -m ruff format .

## Type Checking
This library uses `mypy` for static type checking.
This is configured in `pyproject.toml`.

To check the code for type check errors:

    python -m mypy .

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
