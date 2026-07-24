from typing import List, Literal, Optional

import torch
import numpy as np

from tabicl.prior.graph_lib._base import PriorComponent, Context


class RandomDAG(PriorComponent):
    """
    Random Directed Acyclic Graph (DAG) generator.

    The graph is returned as a list of parent lists. Parent indices are always
    smaller than their child index because ``RandomGraphFunction`` evaluates
    nodes in index order.
    """

    def __init__(
        self,
        context: Context,
        graph_type: Literal["erdos_renyi", "gnr", "cauchy"] = "cauchy",
        p: Optional[float] = None,
    ):
        super().__init__(context)
        self.graph_type = graph_type
        self.p = p

    def _validate_inputs(self, n_nodes: int) -> int:
        n = int(n_nodes)
        if n < 0:
            raise ValueError("n_nodes must be non-negative.")
        if self.graph_type not in ("erdos_renyi", "gnr", "cauchy"):
            raise ValueError(f"Unknown graph type: {self.graph_type!r}.")
        if self.graph_type != "cauchy" and self.p is None:
            raise ValueError(f"p must be provided for the {self.graph_type!r} sampler.")
        if (
            self.graph_type != "cauchy"
            and self.p is not None
            and not 0.0 <= self.p <= 1.0
        ):
            raise ValueError("p must be in [0, 1].")
        return n

    def sample(self, n_nodes: int) -> List[List[int]]:
        n_nodes = self._validate_inputs(n_nodes)
        samplers = {
            "erdos_renyi": RandomErdosRenyiDAG,
            "gnr": RandomGNRDAG,
            "cauchy": RandomCauchyDAG,
        }
        sampler = samplers[self.graph_type](self.context)
        if self.graph_type == "cauchy":
            # The Cauchy offset can be any real number. It has a similar role
            # to an edge probability, but is added directly to the logits.
            return sampler.sample(
                n_nodes,
                cauchy_dag_offset=0.0 if self.p is None else self.p,
            )
        return sampler.sample(n_nodes, self.p)


class RandomErdosRenyiDAG(RandomDAG):
    """
    Samples every possible forward edge independently with probability ``p``.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        adjacency_matrix = torch.rand((n_nodes, n_nodes), device=self.device) < p
        return [
            [parent for parent in range(child) if adjacency_matrix[parent, child]]
            for child in range(n_nodes)
        ]


class RandomGNRDAG(RandomDAG):
    """
    Samples a Growing Network with Redirection DAG.

    Each new node first chooses an existing node uniformly. With probability
    ``p``, the choice is redirected to that node's parent when it has one.
    As in the source GNR sampler, the edge points from the new node to the
    chosen older node. Arrival-order labels are reversed in the returned graph
    so those edges remain forward edges in TabICL's topological index order.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        arrival_targets = np.full(n_nodes, -1, dtype=np.intp)
        parents: List[List[int]] = [[] for _ in range(n_nodes)]

        for source in range(1, n_nodes):
            target = int(np.random.randint(source))
            if np.random.random() < p and arrival_targets[target] != -1:
                target = int(arrival_targets[target])
            arrival_targets[source] = target

            parent_idx = n_nodes - 1 - source
            child_idx = n_nodes - 1 - target
            parents[child_idx].append(parent_idx)

        return parents


class RandomCauchyDAG(RandomDAG):
    """
    Samples a random DAG based on Cauchy random variables modeling connectivities.
    """
    def sample(
        self,
        n_nodes: int,
        cauchy_dag_offset: float = 0.0,
    ) -> List[List[int]]:
        offset = np.random.standard_cauchy() + cauchy_dag_offset
        output_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        input_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        logits = offset + output_importances[None, :] + input_importances[:, None]
        adjacency_matrix = torch.rand_like(logits) <= torch.sigmoid(logits)
        return [[i_in for i_in in range(i_out) if adjacency_matrix[i_in, i_out]] for i_out in range(n_nodes)]
