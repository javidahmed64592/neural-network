[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=ffd343)](https://docs.python.org/3.11/)
<!-- omit from toc -->
# Neural Network
This is a neural network library in Python which can be used to feedforward arrays of inputs, generate outputs, and be trained with expected outputs.

Install this package using `pipenv`:

```
pipenv install -e git+https://github.com/javidahmed64592/neural-network#egg=neural_network
```

<!-- omit from toc -->
## Table of Contents
- [Installing Dependencies](#installing-dependencies)
- [Using the Neural Network](#using-the-neural-network)
- [Testing](#testing)
- [Formatting, Type Checking and Linting](#formatting-type-checking-and-linting)

## Installing Dependencies
Install the required dependencies using [pipenv](https://github.com/pypa/pipenv):

    pipenv install
    pipenv install --dev

## Using the Neural Network
The neural network can be created in the following way:

```
from neural_network.neural_network import NeuralNetwork

nn = NeuralNetwork(num_inputs=num_inputs, num_outputs=num_outputs, hidden_layer_sizes=hidden_layer_sizes)
```

where

- `num_inputs`: Number of inputs to pass through neural network
- `num_outputs`: Number of outputs to be generated
- `hidden_layer_sizes`: List of number of nodes in each hidden layer

To feedforward an array of inputs:

```
outputs = nn.feedforward([x_i, ..., x_n]) # n: Number of inputs
```

The neural network can also be trained by providing an array of inputs and expected outputs, and backpropagating the error using gradient descent.

```
inputs = [x_i, ..., x_n]
expected_outputs = [y_i, ..., y_m] # m: Number of outputs

errors = nn.train(inputs, expected_outputs)
```

## Testing
This library uses Pytest for the unit tests.
These tests are located in the `tests` directory.
To run the tests:

    pipenv run test

## Formatting, Type Checking and Linting
This library uses a number of tools for code formatting and linting. These tools are configured in `pyproject.toml`, `setup.cfg` and `mypy.ini`.

Black is used as a code formatter:

    black .

isort is used for tidying up imports:

    isort .

Mypy is used as a type checker:

    mypy .

Flake8 is used for linting:

    flake8
