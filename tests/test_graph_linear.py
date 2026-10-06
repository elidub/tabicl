import numpy as np
import pytest
import torch

from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._dataset import RandomDataset

# The options that strip the prior down to a linear SCM with one scalar node per column.
linear_options = dict(
    fct_types="lin",
    multi_fct_types="concat",
    random_matrix_types="gaussian",
    use_node_feature_importance=False,
    use_node_importance=False,
    use_node_l2_norm=False,
    allow_kumaraswamy_warping=False,
    subsample_feature_nodes=False,
    scalar_nodes=True,
    unique_column_nodes=True,
    root_point_types="gaussian",
    allow_categorical_features=False,
    threshold_target=True,
    remove_outliers=False,
)


def sample_all_nodes_observed(n_nodes, seed, **config):
    """One dataset with a column on every node, and the node each column is read from."""
    context = Context(config=PriorConfig(min_n_nodes=n_nodes, max_n_nodes=n_nodes, **config), seed=seed)
    properties = DatasetProperties(n_train=200, n_test=0, cat_sizes={"x": [0] * (n_nodes - 1), "y": [0]})
    ds = RandomDataset(context).sample(properties)
    column_node = {
        name: node for node, specs in enumerate(ds.kwargs["node_feature_specs"]) for name in specs
    }
    node_values = {column_node[name]: tensor[:, 0] for name, tensor in ds.tensors.items()}
    return ds.kwargs["graph"], node_values


@pytest.mark.parametrize("seed", range(5))
def test_linear_options_give_a_noiseless_linear_scm(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    graph, values = sample_all_nodes_observed(
        8, seed, graph_type="erdos_renyi", graph_edge_prob=0.5, **linear_options
    )

    assert sorted(values) == list(range(8))  # every node gives exactly one column
    for node, parents in enumerate(graph):
        if not parents:
            continue
        design = torch.stack([values[parent] for parent in parents], dim=1).double()
        target = values[node].double()
        coef = torch.linalg.lstsq(design, target[:, None]).solution
        residual = target - (design @ coef)[:, 0]
        assert residual.norm() <= 1e-4 * target.norm()


def test_noise_in_columns_makes_nodes_noisy_linear_functions_of_their_parents():
    torch.manual_seed(0)
    np.random.seed(0)
    graph, values = sample_all_nodes_observed(
        8, 0, graph_type="erdos_renyi", graph_edge_prob=1.0,
        add_gaussian_noise=True, noise_in_columns=True,
        gaussian_noise_min_sigma=0.5, gaussian_noise_max_sigma=0.5, **linear_options,
    )

    node, parents = 7, graph[7]
    design = torch.stack([values[parent] for parent in parents], dim=1).double()
    target = values[node].double()
    residual = target - (design @ torch.linalg.lstsq(design, target[:, None]).solution)[:, 0]
    assert residual.std() == pytest.approx(0.5, rel=0.2)


def test_unique_column_nodes_needs_a_node_per_column():
    context = Context(config=PriorConfig(min_n_nodes=3, max_n_nodes=3, **linear_options))
    properties = DatasetProperties(n_train=10, n_test=0, cat_sizes={"x": [0] * 3, "y": [0]})
    with pytest.raises(ValueError, match="node per column"):
        RandomDataset(context).sample(properties)


def test_threshold_target_gives_balanced_classes():
    torch.manual_seed(0)
    np.random.seed(0)
    scm = GraphSCM(
        seq_len=300, num_features=4, max_features=4, num_classes=3,
        config=PriorConfig(min_n_nodes=6, max_n_nodes=6, **linear_options),
    )
    X, y = scm()

    assert X.shape == (300, 4)
    assert torch.bincount(y.long(), minlength=3).tolist() == [100, 100, 100]
