[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=ffd343)](https://docs.python.org/3.12/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
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
- [uv](#uv)
- [Installing Dependencies](#installing-dependencies)
- [Using the Neural Network](#using-the-neural-network)
  - [Creating a Neural Network](#creating-a-neural-network)
  - [Training a Neural Network](#training-a-neural-network)
  - [Saving and Loading Models](#saving-and-loading-models)
- [Protobuf Classes](#protobuf-classes)
- [Neuroevolution](#neuroevolution)
- [Testing, Linting, and Type Checking](#testing-linting-and-type-checking)
- [License](#license)

## uv
This repository is managed using the `uv` Python project manager: https://docs.astral.sh/uv/

To install `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh                                    # Linux/Mac
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows
```

## Installing Dependencies
Install the required dependencies using `pip`:

    uv sync

To install with `dev` dependencies:

    uv sync --extra dev

## Using the Neural Network
For a complete example of how to create and train the neural network, see the example notebooks in the `examples` directory.

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
The neural network weights and biases can be saved to a Protobuf file (`.pb`):

```python
from pathlib import Path

directory = Path("models")
filename = "neural_network_data.pb"
NeuralNetwork.save_to_file(nn, filename, directory)
```

To load a neural network from a Protobuf file:

```python
loaded_nn = NeuralNetwork.load_from_file(filepath)
```

## Protobuf Classes

This library supports saving and loading neural network models using [Protocol Buffers (Protobuf)](https://developers.google.com/protocol-buffers). The Protobuf schema is defined in [`protobuf/NeuralNetwork.proto`](protobuf/NeuralNetwork.proto) and compiled Python classes are used for serialization.

- **Saving a model:** The `NeuralNetwork.save_to_file()` method serializes the model to a `.pb` file using Protobuf.
- **Loading a model:** The `NeuralNetwork.load_from_file()` method deserializes a `.pb` file back into a neural network instance.
- **Protobuf schema:** The schema defines messages for activation functions, matrices, and the neural network structure.

This enables efficient, language-agnostic storage and transfer of neural network models.

## Neuroevolution
New weights and biases can also be calculated via crossover:

```python
def crossover_func(element: float, other_element: float, roll: float) -> float:
    if roll < mutation_rate:
        return element
    return other_element

nn_1 = NeuralNetwork(...)
nn_2 = NeuralNetwork(...)
nn_3 = NeuralNetwork(...)

nn_3.weights, nn_3.bias = NeuralNetwork.crossover(
    nn=nn_1,
    other_nn=nn_2,
    weights_crossover_func=crossover_func,
    bias_crossover_func=crossover_func
)
```

## Testing, Linting, and Type Checking

- **Run tests:** `uv run pytest`
- **Lint code:** `uv run ruff check .`
- **Format code:** `uv run ruff format .`
- **Type check:** `uv run mypy .`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
