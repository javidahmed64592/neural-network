import pytest

from neural_network.ga.genotype import Genotype
from neural_network.nn.node import InputNode, Node, NodeConnection


@pytest.fixture
def mock_input_nodes() -> list[InputNode]:
    i1 = InputNode(index=0)
    i2 = InputNode(index=1)
    return [i1, i2]


@pytest.fixture
def mock_output_nodes(mock_bias_range: tuple[float, float]) -> list[Node]:
    o1 = Node.random_node(index=0, bias_range=mock_bias_range)
    o2 = Node.random_node(index=0, bias_range=mock_bias_range)
    o3 = Node.random_node(index=0, bias_range=mock_bias_range)
    return [o1, o2, o3]


@pytest.fixture
def mock_genotype(mock_input_nodes: list[InputNode], mock_output_nodes: list[Node]) -> Genotype:
    connections: list[NodeConnection] = []
    innovation = 1
    for output_node in mock_output_nodes:
        for input_node in mock_input_nodes:
            connections.append(output_node.add_node(node=input_node, weight=0.5, innovation=innovation))
            innovation += 1

    return Genotype(genotype=connections)


class TestGenotype:
    def test_given_genotype_when_ordering_by_innovation_then_check_correct_order_returned(
        self, mock_genotype: Genotype
    ) -> None:
        expected_order = [1, 2, 3, 4, 5, 6]
        actual_order = [gene.innovation for gene in mock_genotype.ordered_genotype]

        assert actual_order == expected_order

    def test_given_genotype_when_getting_active_connections_then_check_lists_have_correct_length(
        self, mock_genotype: Genotype
    ) -> None:
        mock_genotype.genotype[2].toggle_active()
        mock_genotype.genotype[3].toggle_active()

        expected_no_inactive = 2
        expected_no_active = len(mock_genotype.genotype) - expected_no_inactive

        actual_no_inactive = len(mock_genotype.inactive_connections)
        actual_no_active = len(mock_genotype.active_connections)

        assert actual_no_inactive == expected_no_inactive
        assert expected_no_active == actual_no_active
