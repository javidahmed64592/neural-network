from datetime import datetime

import numpy as np

from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.activation_functions import LinearActivation, SigmoidActivation
from neural_network.neural_network import NeuralNetwork

rng = np.random.default_rng()

NUM_BITS = 8
IN_LIMS = [0, 255]
OUT_LIMS = [0, 1]

# Constants
DATASET_SIZE = 30000
TRAIN_SIZE_RATIO = 0.8
EPOCHS = 1

HIDDEN_LAYER_SIZES = [3]
INPUT_ACTIVATION = LinearActivation
HIDDEN_ACTIVATION = SigmoidActivation
OUTPUT_ACTIVATION = SigmoidActivation
WEIGHTS_RANGE = (-1, 1)
BIAS_RANGE = (-0.3, 0.3)
LR = 0.2


# Helper methods
def generate_time_msg() -> str:
    """
    Get message prefix with current datetime.
    """
    return f"[{datetime.now().strftime('%d-%m-%G | %H:%M:%S')}]"


def print_system_msg(msg: str) -> None:
    """
    Print a message to the terminal.

    Parameters:
        msg (str): Message to print
    """
    print(f"{generate_time_msg()} {msg}")


def print_flushed_msg(msg: str) -> None:
    """
    Print a flushed message to the terminal.

    Parameters:
        msg (str): Message to print
    """
    print(f"\r{generate_time_msg()} {msg}", flush=True, end="")


# Byte list methods
def num_to_byte_list(num: int) -> list[int]:
    """
    Convert a number to a list of bits.

    Parameters:
        num (int): Number to convert

    Returns:
        byte_list (list[int]): Number represented as list of bits
    """
    _num_bin = bin(num)
    _num_bytes = _num_bin[2:]
    _padding = [0] * (NUM_BITS - len(_num_bytes))
    return _padding + [int(b) for b in _num_bytes]


def map_val(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    Map a value from an input range to an output range.

    Parameters:
        x (float): Number to map to new range
        in_min (float): Lower bound of original range
        in_max (float): Upper bound of original range
        out_min (float): Lower bound of new range
        out_max (float): Upper bound of new range

    Returns:
        y (float): Number mapped to new range
    """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# Training methods
def create_nn(
    input_size: int = NUM_BITS,
    hidden_layer_sizes: list[int] = HIDDEN_LAYER_SIZES,
    input_activation: type = INPUT_ACTIVATION,
    hidden_activation: type = HIDDEN_ACTIVATION,
    output_activation: type = OUTPUT_ACTIVATION,
    weights_range: tuple[float, float] = WEIGHTS_RANGE,
    bias_range: tuple[float, float] = BIAS_RANGE,
    lr: float = LR,
) -> NeuralNetwork:
    """Create a neural network with specified parameters."""
    input_layer = InputLayer(size=input_size, activation=input_activation)
    hidden_layers = [
        HiddenLayer(size=size, activation=hidden_activation, weights_range=weights_range, bias_range=bias_range)
        for size in hidden_layer_sizes
    ]
    output_layer = OutputLayer(size=1, activation=output_activation, weights_range=weights_range, bias_range=bias_range)

    return NeuralNetwork.from_layers(layers=[input_layer, *hidden_layers, output_layer], lr=lr)


def training_data_from_num(num: int) -> tuple[list[int], float]:
    """
    Generate byte list and mapped number from a number to use in training.

    Parameters:
        num (int): Number to use for training data

    Returns:
        training_data (tuple[list[int], float]): Input and expected output
    """
    _byte_list = np.array(num_to_byte_list(num))
    _mapped_num = map_val(num, IN_LIMS[0], IN_LIMS[1], OUT_LIMS[0], OUT_LIMS[1])
    return (_byte_list, _mapped_num)


def split_data(
    data: list[tuple[list[int], float]], train_size_ratio: float = TRAIN_SIZE_RATIO
) -> tuple[list[tuple[list[int], float]], list[tuple[list[int], float]]]:
    """
    Split the dataset into training and testing sets.

    Parameters:
        data (list[tuple[list[int], float]]): The dataset to split.
        train_size_ratio (float): The proportion of the dataset to include in the training split.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    train_size = int(len(data) * train_size_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def calculate_errors(expected_outputs: np.ndarray, actual_outputs: np.ndarray) -> np.ndarray:
    """
    Calculate the error between expected and actual outputs.

    Parameters:
        expected_outputs (np.ndarray): The expected output values.
        actual_outputs (np.ndarray): The actual output values.

    Returns:
        errors (np.ndarray): The calculated errors.
    """
    errors = expected_outputs - np.array(actual_outputs)
    return map_val(errors, OUT_LIMS[0], OUT_LIMS[1], IN_LIMS[0], IN_LIMS[1])


def calculate_percentage_error(errors: np.ndarray) -> float:
    """
    Calculate the percentage error from a list of errors.

    Parameters:
        errors (np.ndarray): The list of errors.

    Returns:
        percentage_error (float): The average error as a percentage.
    """
    avg_error = np.average(errors)
    return np.abs(avg_error) / IN_LIMS[1]


def evaluate_nn(
    nn: NeuralNetwork, data: list[tuple[list[int], float]] | list[tuple[list[int], float, float]]
) -> tuple[float, float]:
    """
    Evaluate the neural network on a dataset.

    Parameters:
        nn (NeuralNetwork): The neural network to evaluate.
        data (list[tuple[list[int], float]] | list[tuple[list[int], float, float]]): The dataset to evaluate on.

    Returns:
        errors (np.ndarray): The list of errors.
        percentage_error (float): The average error as a percentage.
    """
    dataset_size = len(data)
    outputs = []
    for i in range(dataset_size):
        inputs = data[i][0]
        output = nn.feedforward(inputs)[0]
        outputs.append(output)

    errors = calculate_errors(
        expected_outputs=np.array([data[i][1] for i in range(dataset_size)]), actual_outputs=np.array(outputs)
    )
    percentage_error = calculate_percentage_error(errors)
    return errors, percentage_error


# Supervised training
def generate_supervised_training_data(dataset_size: int) -> list[tuple[list[int], float]]:
    """
    Generate supervised training data for the neural network.

    Returns:
        training_data (tuple[list[list[int]], list[float]]): Input and expected output pairs
    """
    random_num = rng.integers(low=IN_LIMS[0], high=(IN_LIMS[1] + 1), size=dataset_size)
    return [training_data_from_num(num) for num in random_num]


def train_supervised_nn(nn: NeuralNetwork, data: list[tuple[list[int], float]], epochs: int = 1) -> None:
    """
    Train the neural network using supervised learning.

    Parameters:
        nn (NeuralNetwork): The neural network to train.
        data (list[tuple[list[int], float]]): The training data.
        epochs (int): The number of training epochs.
    """
    for _ in range(epochs):
        for input_data, expected_output in data:
            nn.train(input_data, expected_output)


# Fitness training
def calculate_fitness(expected_output: float, nn_output: float) -> float:
    """
    Calculate fitness based on the accuracy of the neural network's output.

    Parameters:
        expected_output (float): The correct output value.
        nn_output (float): The neural network's predicted output.

    Returns:
        fitness (float): A fitness value where higher is better.
    """
    return 1.0 - abs(expected_output - nn_output)


def generate_fitness_training_data(dataset_size: int, nn: NeuralNetwork) -> list[tuple[list[int], float, float]]:
    """
    Generate fitness training data for the neural network.

    Returns:
        training_data (tuple[list[list[int]], list[float], list[float]]): Input, expected output, and fitness values
    """
    data = generate_supervised_training_data(dataset_size)
    nn_outputs = [nn.feedforward(input_data) for input_data, _ in data]
    return [
        (input_data, expected_output, calculate_fitness(expected_output, nn_output))
        for (input_data, expected_output), nn_output in zip(data, nn_outputs, strict=False)
    ]


def train_fitness_nn(nn: NeuralNetwork, data: list[tuple[list[int], float, float]], epochs: int = 1) -> None:
    """
    Train the neural network using fitness-based learning.

    Parameters:
        nn (NeuralNetwork): The neural network to train.
        data (list[tuple[list[int], float]]): The training data.
        epochs (int): The number of training epochs.
    """
    for _ in range(epochs):
        for i in range(len(data) - 1):
            inputs, outputs, fitness = data[i]
            _, _, prev_fitness = data[i - 1] if i > 0 else (inputs, outputs, fitness)

            nn.train_with_fitness(inputs, outputs, fitness, prev_fitness)


def main() -> None:
    """
    Main function to run the training process.
    """
    # Supervised training
    print_system_msg("Creating neural network for supervised training...")
    nn_supervised = create_nn()
    print_system_msg("Generating supervised training data...")
    data_supervised = generate_supervised_training_data(DATASET_SIZE)
    training_data_supervised, testing_data_supervised = split_data(data_supervised)
    print_system_msg("Training neural network with supervised learning...")
    train_supervised_nn(nn_supervised, training_data_supervised, epochs=EPOCHS)

    # Testing
    print_system_msg("Testing neural network with supervised learning...")
    _, percentage_error = evaluate_nn(nn_supervised, testing_data_supervised)
    print_system_msg(f"Supervised training percentage error: {percentage_error:.4f}%")

    # Fitness training
    print_system_msg("Creating neural network for fitness training...")
    nn_fitness = create_nn()
    print_system_msg("Generating fitness training data...")
    data_fitness = generate_fitness_training_data(DATASET_SIZE, nn_fitness)
    training_data_fitness, testing_data_fitness = split_data(data_fitness)
    print_system_msg("Training neural network with fitness-based learning...")
    train_fitness_nn(nn_fitness, training_data_fitness, epochs=EPOCHS)

    # Testing
    print_system_msg("Testing neural network with fitness-based learning...")
    _, percentage_error = evaluate_nn(nn_fitness, testing_data_fitness)
    print_system_msg(f"Fitness training percentage error: {percentage_error:.4f}%")


if __name__ == "__main__":
    main()
