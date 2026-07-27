from typing import Dict, List

import numpy as np
import torch

from tabicl.prior.graph_lib._base import PriorComponent, Dataset, DatasetProperties, FeatureSpec
from tabicl.prior.graph_lib._graph import RandomDAG
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction


def check_x_y_ancestors_overlap(graph: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]) -> bool:
    """
    There can only be a functional relation between x and y if one of their ancestors overlap.

    :param graph: Graph (list of parent node idxs for each node)
    :param node_feature_specs: feature specs with feature names for each node
    :return: False if x and y are completely independent according to the graph, otherwise True.
    """
    ancestors_arrs = {key: np.zeros(len(graph), dtype=np.bool_) for key in ['x', 'y']}
    for node_idx in reversed(range(len(graph))):
        for feature_spec in node_feature_specs[node_idx].values():
            if feature_spec.group in ['x', 'y']:
                # set this node as one of the ancestors from this group
                ancestors_arrs[feature_spec.group][node_idx] = True
        for group in ['x', 'y']:
            if ancestors_arrs[group][node_idx]:
                for parent_idx in graph[node_idx]:
                    ancestors_arrs[group][parent_idx] = True

    return np.any(np.logical_and(ancestors_arrs['x'], ancestors_arrs['y']))


class RandomDataset(PriorComponent):
    def _sample_scalar_spec(self, spec: float | dict | None) -> float | None:
        """Resolve a fixed value or distribution spec to one scalar."""
        if not isinstance(spec, dict):
            return spec

        dist = spec["dist"]
        if dist == "fixed":
            return spec["value"]
        if dist == "beta":
            return self.sampler.rng.beta(spec["alpha"], spec["beta"])
        raise ValueError(f"Unknown scalar distribution: {dist!r}")

    def sample_graph_edge_prob(self) -> float | None:
        """Resolve ``config.graph_edge_prob`` to a concrete graph control.

        Depending on ``graph_type`` this is either the edge probability
        (erdos_renyi/all GNR variants) or the additive DAG offset (cauchy). It accepts:

        - ``{'dist': 'fixed', 'value': 0.0}`` — a constant, equivalent to the plain float.
        - ``{'dist': 'beta', 'alpha': 2, 'beta': 6}`` — sampled from Beta(alpha, beta).
        """
        return self._sample_scalar_spec(self.config.graph_edge_prob)

    def sample(self, data_prop: DatasetProperties) -> Dataset:
        while True:
            # ----- Create computation graph -----
            n_nodes = self.sampler.randint("n_nodes", self.config.min_n_nodes, self.config.max_n_nodes + 1, use_log=True)
            graph = RandomDAG(
                self.context,
                graph_type=self.config.graph_type,
                p=self.sample_graph_edge_prob(),
            ).sample(n_nodes)

            node_feature_specs = [dict() for _ in range(n_nodes)]

            feature_groups = list(set(spec.group for spec in data_prop.feature_specs.values()))
            for feature_group in feature_groups:
                feature_specs = {
                    key: value for key, value in data_prop.feature_specs.items() if value.group == feature_group
                }
                if self.config.subsample_feature_nodes:
                    n_feature_nodes = self.sampler.randint("n_feature_nodes", 1, n_nodes + 1)
                    feature_nodes = np.random.permutation(n_nodes)[:n_feature_nodes]
                else:
                    feature_nodes = np.arange(n_nodes)
                feature_node_idxs = np.random.choice(feature_nodes, replace=True, size=len(feature_specs))

                for idx, (feature_name, feature_spec) in enumerate(feature_specs.items()):
                    node_feature_specs[feature_node_idxs[idx]][feature_name] = feature_spec

            if (not self.config.filter_unpredictable_graphs) or check_x_y_ancestors_overlap(graph, node_feature_specs):
                break  # break if predictable, otherwise try again because y and x are independent in the graph

        if self.config.graph_only:
            # Only the structure is wanted, so skip sampling and evaluating the
            # graph function, which is what dominates the runtime. Columns are
            # returned with zero rows rather than filled with dummy values, so
            # that no consumer can mistake them for data.
            tensors = {
                name: torch.zeros(0, 1, device=self.device) for name in data_prop.feature_specs
            }
        else:
            graph_func = RandomGraphFunction(self.context, dag=graph, node_feature_specs=node_feature_specs)

            # ----- Evaluate computation graph -----
            n_samples = data_prop.n_train + data_prop.n_test
            if self.config.ensure_iid:
                graph_func(n_samples)  # fit the graph function on separate data
            tensors = graph_func(n_samples)
        return Dataset(
            tensors=tensors,
            feature_specs=data_prop.feature_specs,
            graph=graph,
            node_feature_specs=node_feature_specs,
            n_train=data_prop.n_train,
        )
