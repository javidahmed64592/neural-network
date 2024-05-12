from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from neural_network.nn.node import InputNode, Node, NodeConnection


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

    @classmethod
    def from_nodes(cls, input_nodes: list[InputNode], output_nodes: list[Node]) -> Genotype:
        connections: list[NodeConnection] = []
        innovation = 1
        for output_node in output_nodes:
            for input_node in input_nodes:
                connections.append(output_node.add_node(node=input_node, weight=0.5, innovation=innovation))
                innovation += 1

        return cls(genotype=connections)
