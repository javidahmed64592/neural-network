from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from neural_network.nn.node import NodeConnection


@dataclass
class Genotype:
    genotype: list[NodeConnection]

    @property
    def innovation_nos(self) -> Generator:
        for gene in self.genotype:
            yield gene.innovation

    @property
    def ordered_genotype(self) -> list[NodeConnection]:
        innovation_nos = np.fromiter(self.innovation_nos, dtype=int)
        sorted_indices = np.argsort(innovation_nos)
        return np.array([self.genotype[i] for i in sorted_indices])

    @property
    def active_connections(self) -> list[NodeConnection]:
        return np.array([gene for gene in self.ordered_genotype if gene.is_active])

    @property
    def inactive_connections(self) -> list[NodeConnection]:
        return np.array([gene for gene in self.ordered_genotype if not gene.is_active])
