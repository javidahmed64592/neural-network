import datetime

import numpy as np

from src.nn.neural_network import NeuralNetwork


def main():
    num_inputs = 2
    num_hidden = 4
    num_outputs = 1

    inputs = [[0, 1], [1, 0], [1, 1], [0, 0]]
    outputs = [[1], [1], [0], [0]]

    nn = NeuralNetwork(num_inputs, num_hidden, num_outputs)

    for _ in range(20000):
        random_choice = np.random.randint(low=0, high=len(inputs))
        nn.train(inputs[random_choice], outputs[random_choice])

    print(f"Guessing inputs {inputs[0]}: Calculated outputs {nn.feedforward(inputs[0])} \t| Expected: {outputs[0]}")
    print(f"Guessing inputs {inputs[1]}: Calculated outputs {nn.feedforward(inputs[1])} \t| Expected: {outputs[1]}")
    print(f"Guessing inputs {inputs[2]}: Calculated outputs {nn.feedforward(inputs[2])} \t| Expected: {outputs[2]}")
    print(f"Guessing inputs {inputs[3]}: Calculated outputs {nn.feedforward(inputs[3])} \t| Expected: {outputs[3]}")


def time_feedforward():
    num_inputs = 2
    num_hidden = 4
    num_outputs = 1

    inputs = [[0, 1], [1, 0], [1, 1], [0, 0]]

    nn = NeuralNetwork(num_inputs, num_hidden, num_outputs)

    num_iters = 12000 * 100

    begin_time = datetime.datetime.now()
    print(f"Starting feedfoward: {num_iters} times")
    for i in range(num_iters):
        print(f"\rProgress: {i+1} / {num_iters}", flush=True, end="")
        random_choice = np.random.randint(low=0, high=len(inputs))
        nn.feedforward(inputs[random_choice])

    dt = datetime.datetime.now() - begin_time
    dt_m = int(dt.total_seconds() // 60)
    dt_s = int(dt.total_seconds() - (dt_m * 60))
    print(f"\nDone! The time it took is {dt_m}m {dt_s}s.")


if __name__ == "__main__":
    time_feedforward()
