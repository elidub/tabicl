import pytest

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._graph import (
    RandomCauchyDAG,
    RandomConvergingGNRDAG,
    RandomDAG,
    RandomDivergingGNRDAG,
    RandomErdosRenyiDAG,
    RandomTopologicalGNRDAG,
)


def assert_valid_parent_list(graph):
    for child, parents in enumerate(graph):
        assert all(0 <= parent < child for parent in parents)


@pytest.mark.parametrize(
    ("sampler", "p"),
    [
        (RandomErdosRenyiDAG, 0.5),
        (RandomConvergingGNRDAG, 0.5),
        (RandomDivergingGNRDAG, 0.5),
        (RandomTopologicalGNRDAG, 0.5),
    ],
)
def test_graph_samplers_return_topologically_ordered_parent_lists(sampler, p):
    graph = sampler(Context()).sample(20, p)

    assert len(graph) == 20
    assert_valid_parent_list(graph)


def test_cauchy_sampler_returns_topologically_ordered_parent_lists():
    graph = RandomCauchyDAG(Context()).sample(20)

    assert len(graph) == 20
    assert_valid_parent_list(graph)


@pytest.mark.parametrize("offset", [-10.0, 0.0, 10.0])
def test_cauchy_sampler_accepts_any_offset(offset):
    graph = RandomDAG(Context(), graph_type="cauchy", p=offset).sample(10)

    assert len(graph) == 10
    assert_valid_parent_list(graph)


def test_erdos_renyi_probability_extremes():
    sampler = RandomErdosRenyiDAG(Context())

    assert sampler.sample(5, 0.0) == [[], [], [], [], []]
    assert sampler.sample(5, 1.0) == [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
    ]


def test_gnr_returns_a_tree():
    graph = RandomConvergingGNRDAG(Context()).sample(20, 0.5)

    assert graph[0] == []
    assert sum(map(len, graph)) == 19
    parent_counts = [sum(parent in parents for parents in graph) for parent in range(20)]
    assert parent_counts == [1] * 19 + [0]


def test_diverging_gnr_is_an_out_arborescence():
    graph = RandomDivergingGNRDAG(Context()).sample(20, 0.5)

    assert graph[0] == []
    assert [len(parents) for parents in graph] == [0] + [1] * 19


def test_random_topological_gnr_has_random_acyclic_orientation():
    graph = RandomTopologicalGNRDAG(Context(seed=2)).sample(50, 0.5)

    assert sum(map(len, graph)) == 49
    indegrees = [len(parents) for parents in graph]
    outdegrees = [sum(parent in parents for parents in graph) for parent in range(50)]
    assert max(indegrees) > 1
    assert max(outdegrees) > 1


@pytest.mark.parametrize(
    "graph_type",
    ["erdos_renyi", "gnr_converging", "gnr_diverging", "grn_random"],
)
def test_random_dag_dispatches_probability_based_samplers(graph_type):
    graph = RandomDAG(Context(), graph_type=graph_type, p=0.5).sample(10)

    assert len(graph) == 10
    assert_valid_parent_list(graph)


def test_random_dag_defaults_to_cauchy():
    graph = RandomDAG(Context()).sample(10)

    assert len(graph) == 10
    assert_valid_parent_list(graph)


@pytest.mark.parametrize(
    "graph_type",
    ["erdos_renyi", "gnr_converging", "gnr_diverging", "grn_random"],
)
def test_probability_based_samplers_validate_probability(graph_type):
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        RandomDAG(Context(), graph_type=graph_type, p=1.1).sample(5)


def test_graph_samplers_validate_node_count():
    with pytest.raises(ValueError, match="n_nodes must be non-negative"):
        RandomDAG(Context()).sample(-1)
