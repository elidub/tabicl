from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tabicl.prior.graph_lib._base import PriorComponent, Context, Dataset, DatasetProperties, FeatureSpec, seeded
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

    def _sample_graph(self) -> List[List[int]]:
        n_nodes = self.sampler.randint("n_nodes", self.config.min_n_nodes, self.config.max_n_nodes + 1, use_log=True)
        return RandomDAG(
            self.context,
            graph_type=self.config.graph_type,
            p=self.sample_graph_edge_prob(),
        ).sample(n_nodes)

    def _assign_columns(self, n_nodes: int, data_prop: DatasetProperties) -> List[Dict[str, FeatureSpec]]:
        """The columns read from each node."""
        node_feature_specs = [dict() for _ in range(n_nodes)]

        if self.config.unique_column_nodes:
            if self.config.subsample_feature_nodes:
                raise ValueError("unique_column_nodes=True needs subsample_feature_nodes=False.")
            n_columns = len(data_prop.feature_specs)
            if n_columns > n_nodes:
                raise ValueError(
                    f"unique_column_nodes=True needs a node per column, but a graph with {n_nodes} nodes "
                    f"was sampled for {n_columns} columns: raise min_n_nodes to at least {n_columns}."
                )
            column_nodes = np.random.permutation(n_nodes)[:n_columns]
            for node_idx, (feature_name, feature_spec) in zip(column_nodes, data_prop.feature_specs.items()):
                node_feature_specs[node_idx][feature_name] = feature_spec
        else:
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
        return node_feature_specs

    def _sample_structure(
        self, data_prop: DatasetProperties, graph: Optional[List[List[int]]] = None
    ) -> Tuple[List[List[int]], List[Dict[str, FeatureSpec]]]:
        """A graph and the columns of each node; with ``graph``, only the columns are drawn.

        Redrawn until x and y share an ancestor, with ``filter_unpredictable_graphs``.
        """
        while True:
            dag = self._sample_graph() if graph is None else graph
            node_feature_specs = self._assign_columns(len(dag), data_prop)
            if (not self.config.filter_unpredictable_graphs) or check_x_y_ancestors_overlap(dag, node_feature_specs):
                return dag, node_feature_specs  # predictable, otherwise try again: y and x are independent

    def _sample_pool_member(self, data_prop: DatasetProperties):
        """One of the ``pool_size`` variants of the fixed part of the SCM (``pool_level``).

        Variant ``k`` is drawn under its own seed from ``(pool_seed, k)``, which restores the
        caller's random streams afterwards, so every process draws the same variants. Returns
        ``k``, the graph, the columns of each node (None at level 'graph') and the graph
        function, fitted on data that is discarded (None below level 'scm').
        """
        if self.config.pool_level not in ("graph", "assignment", "scm"):
            raise ValueError(f"Unknown {self.config.pool_level=}")
        fit_scm = self.config.pool_level == "scm" and not self.config.graph_only
        if fit_scm and not self.config.hold_node_draws:
            # Otherwise every call redraws the noise levels and root maps: not one SCM.
            raise ValueError("pool_level='scm' needs hold_node_draws=True.")
        pool_index = int(self.sampler.rng.integers(self.config.pool_size))
        seed = int(np.random.SeedSequence([self.config.pool_seed, pool_index]).generate_state(1)[0])
        with seeded(seed):
            member = RandomDataset(Context(device=self.device, config=self.config, seed=seed))
            graph, node_feature_specs = member._sample_structure(data_prop)
            graph_func = None
            if fit_scm:
                graph_func = RandomGraphFunction(member.context, dag=graph, node_feature_specs=node_feature_specs)
                graph_func(data_prop.n_train + data_prop.n_test)
        if self.config.pool_level == "graph":
            node_feature_specs = None
        return pool_index, graph, node_feature_specs, graph_func

    def sample(self, data_prop: DatasetProperties) -> Dataset:
        pool_index, graph, node_feature_specs, graph_func = None, None, None, None
        if self.config.pool_size:
            pool_index, graph, node_feature_specs, graph_func = self._sample_pool_member(data_prop)
        if node_feature_specs is None:
            graph, node_feature_specs = self._sample_structure(data_prop, graph)

        if self.config.graph_only:
            # Only the structure is wanted, so skip sampling and evaluating the
            # graph function, which is what dominates the runtime. Columns are
            # returned with zero rows rather than filled with dummy values, so
            # that no consumer can mistake them for data.
            tensors = {
                name: torch.zeros(0, 1, device=self.device) for name in data_prop.feature_specs
            }
        else:
            # ----- Evaluate computation graph -----
            n_samples = data_prop.n_train + data_prop.n_test
            if graph_func is None:
                graph_func = RandomGraphFunction(self.context, dag=graph, node_feature_specs=node_feature_specs)
                if self.config.ensure_iid:
                    graph_func(n_samples)  # fit the graph function on separate data
            tensors = graph_func(n_samples)
        return Dataset(
            tensors=tensors,
            feature_specs=data_prop.feature_specs,
            graph=graph,
            node_feature_specs=node_feature_specs,
            n_train=data_prop.n_train,
            graph_function=graph_func,  # the fitted functions and noise levels, for inspection
            pool_index=pool_index,
        )
