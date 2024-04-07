from typing import List

from src.models.point import Point
from src.nn.node import Node


def generate_training_set(num_points: int) -> List[Point]:
    return Point.training_set(num_points)


def main(num_inputs, num_points):
    node = Node(num_inputs)

    num_iters = 100
    for i in range(num_iters):
        m = -node._weights[0] / node._weights[1]
        c = -node._weights[2] / node._weights[1]
        print(f"\rGeneration {i+1} / {num_iters}: {m:.2f} x + {c:.2f}", end="", flush=True)
        points = generate_training_set(num_points)
        point_inputs = [[point.x, point.y] for point in points]
        point_targets = [point.label for point in points]
        for input, target in zip(point_inputs, point_targets):
            node.train(input, target)

    print("\nTraining complete!")
    print(f"Guess: {node.feedforward([0.3, 0.6])} Expected: -1")
    print(f"Guess: {node.feedforward([0.9, 0.1])} Expected: 1")
    print(f"Guess: {node.feedforward([0.4, -1])} Expected: 1")
    print(f"Guess: {node.feedforward([-0.4, 0.5])} Expected: -1")


if __name__ == "__main__":
    num_points = 2000
    num_inputs = 2

    main(num_inputs, num_points)
