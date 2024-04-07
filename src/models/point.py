from __future__ import annotations

from typing import List

import numpy as np


class Point:
    def __init__(self) -> None:
        self.x = np.random.randint(-1, 1)
        self.y = np.random.randint(-1, 1)
        self.label = [-1, 1][self.x > self.y]

    def __str__(self) -> str:
        return f"x: {self.x:<4} \t| y: {self.y:<4} \t| Label: {self.label}"

    @classmethod
    def training_set(cls, size: int) -> List[Point]:
        data_set = [cls() for _ in range(size)]
        return data_set
