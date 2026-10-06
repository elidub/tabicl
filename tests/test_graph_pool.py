import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._dataset import RandomDataset

from .test_graph_linear import linear_options


def sample(seed, **config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    context = Context(config=PriorConfig(min_n_nodes=6, max_n_nodes=6, **{**linear_options, **config}), seed=seed)
    properties = DatasetProperties(n_train=50, n_test=0, cat_sizes={"x": [0] * 3, "y": [0]})
    return RandomDataset(context).sample(properties)


def node_draws(graph_function):
    """The weights of every evaluated node, and its noise level."""
    return [
        (
            node.random_func_.func_.func_.fct_.matrix_.numpy().tolist() if parents else None,
            float(node.noise_level_),
        )
        for parents, node, computed in zip(graph_function.dag, graph_function.nodes_, graph_function.should_compute_)
        if computed
    ]


@pytest.mark.parametrize("hold", [False, True])
def test_hold_node_draws_keeps_noise_levels_and_root_maps(hold):
    dataset = sample(0, add_gaussian_noise=True, noise_in_columns=True, hold_node_draws=hold)
    graph_function = dataset.kwargs["graph_function"]
    roots = [node for parents, node in zip(graph_function.dag, graph_function.nodes_) if not parents]
    before = node_draws(graph_function), [node.random_points_.held_[1] if hold else None for node in roots]
    graph_function(50)
    after = node_draws(graph_function), [node.random_points_.held_[1] if hold else None for node in roots]
    # weights are fitted once either way; noise levels (and root maps) only when held
    assert [w for w, _ in before[0]] == [w for w, _ in after[0]]
    assert ([s for _, s in before[0]] == [s for _, s in after[0]]) == hold
    if hold:
        assert all(a is b for a, b in zip(before[1], after[1]))


def test_pool_of_one_scm_gives_the_same_scm_with_new_rows():
    pooled = dict(pool_size=1, pool_level="scm", hold_node_draws=True, add_gaussian_noise=True, noise_in_columns=True)
    first, second = sample(0, **pooled), sample(1, **pooled)
    assert first.kwargs["pool_index"] == second.kwargs["pool_index"] == 0
    assert first.kwargs["graph"] == second.kwargs["graph"]
    columns = [[sorted(specs) for specs in dataset.kwargs["node_feature_specs"]] for dataset in (first, second)]
    assert columns[0] == columns[1]
    assert node_draws(first.kwargs["graph_function"]) == node_draws(second.kwargs["graph_function"])
    assert not torch.equal(first.tensors["y_0"], second.tensors["y_0"])


def test_pool_does_not_change_the_callers_random_streams():
    # drawing a variant runs under its own seed and restores numpy's and torch's state
    torch.manual_seed(0)
    np.random.seed(0)
    state = np.random.get_state()[1].copy(), torch.get_rng_state().clone()
    dataset = RandomDataset(Context(config=PriorConfig(min_n_nodes=6, max_n_nodes=6, pool_size=2, **linear_options), seed=0))
    dataset._sample_pool_member(DatasetProperties(n_train=10, n_test=0, cat_sizes={"x": [0] * 3, "y": [0]}))
    assert (np.random.get_state()[1] == state[0]).all() and torch.equal(torch.get_rng_state(), state[1])


def test_pool_of_scms_needs_held_node_draws():
    with pytest.raises(ValueError, match="hold_node_draws"):
        sample(0, pool_size=2, pool_level="scm", hold_node_draws=False)
