from datetime import datetime

import numpy as np

from neural_network.neural_network import NeuralNetwork


def main():
    num_inputs = 2
    num_hidden = [4]
    num_outputs = 1

    inputs = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    outputs = [[1.0], [1.0], [0.0], [0.0]]

    nn = NeuralNetwork(num_inputs, num_outputs, num_hidden)
    num_iters = 20000
    for i in range(num_iters):
        random_choice = np.random.randint(low=0, high=len(inputs))
        errors = nn.train(inputs[random_choice], outputs[random_choice])
        print(f"\r{i+1} / {num_iters} -> RMS: {calculate_rms(errors)}", flush=True, end="")

    print(f"\nGuessing inputs {inputs[0]}: Calculated outputs {nn.feedforward(inputs[0])} \t| Expected: {outputs[0]}")
    print(f"Guessing inputs {inputs[1]}: Calculated outputs {nn.feedforward(inputs[1])} \t| Expected: {outputs[1]}")
    print(f"Guessing inputs {inputs[2]}: Calculated outputs {nn.feedforward(inputs[2])} \t| Expected: {outputs[2]}")
    print(f"Guessing inputs {inputs[3]}: Calculated outputs {nn.feedforward(inputs[3])} \t| Expected: {outputs[3]}")

    nn2 = NeuralNetwork(num_inputs, num_outputs, num_hidden)

    nn.weights = nn2.weights
    nn.bias = nn2.bias

    print(f"\nGuessing inputs {inputs[0]}: Calculated outputs {nn.feedforward(inputs[0])} \t| Expected: {outputs[0]}")
    print(f"Guessing inputs {inputs[1]}: Calculated outputs {nn.feedforward(inputs[1])} \t| Expected: {outputs[1]}")
    print(f"Guessing inputs {inputs[2]}: Calculated outputs {nn.feedforward(inputs[2])} \t| Expected: {outputs[2]}")
    print(f"Guessing inputs {inputs[3]}: Calculated outputs {nn.feedforward(inputs[3])} \t| Expected: {outputs[3]}")


def calculate_rms(errors):
    squared = np.square(errors)
    mean = np.average(squared)
    rms = np.sqrt(mean)
    return rms


def feedforward():
    num_inputs = 16
    num_hidden = [8]
    num_outputs = 4

    nn = NeuralNetwork(num_inputs, num_outputs, num_hidden)

    num_iters = 12000 * 100

    print(f"Starting feedfoward: {num_iters} times")
    for i in range(num_iters):
        print(f"\rProgress: {i+1} / {num_iters}", flush=True, end="")
        inputs = np.random.uniform(low=-1, high=1, size=(num_inputs,))
        nn.feedforward(inputs)


def time_it(method):
    begin_time = datetime.now()

    method()

    dt = datetime.now() - begin_time
    dt_m = int(dt.total_seconds() // 60)
    dt_s = int(dt.total_seconds() - (dt_m * 60))
    print(f"\nDone! The time it took is {dt_m}m {dt_s}s.")


if __name__ == "__main__":
    time_it(feedforward)
