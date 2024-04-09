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


if __name__ == "__main__":
    main()
