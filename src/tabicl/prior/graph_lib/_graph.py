from typing import Iterable, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.special import expit

from tabicl.prior.graph_lib._base import PriorComponent, Context


class _DAGSampler(PriorComponent):
    """
    Base class for DAG samplers, holding the conversion to TabICL's graph representation.
    """

    def _parent_lists(self, n_nodes: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
        """
        Converts an iterable of edges into the parent-list representation.

        Every edge is oriented from its lower to its higher node index, so the identity
        permutation is always a topological order, as ``RandomGraphFunction`` requires.
        Parent lists are sorted so that a given edge set always maps to the same representation.

        :param n_nodes: Number of nodes in the graph.
        :param edges: Undirected or directed edges as ``(u, v)`` index pairs.
        :return: Graph as a list of parent node idxs for each node.
        """
        parents: List[List[int]] = [[] for _ in range(n_nodes)]
        for u, v in edges:
            parents[max(u, v)].append(min(u, v))
        for parent_list in parents:
            parent_list.sort()
        return parents


class RandomDAG(_DAGSampler):
    """
    Random Directed Acyclic Graph (DAG) generator.

    The graph is returned as a list of parent lists. Parent indices are always
    smaller than their child index because ``RandomGraphFunction`` evaluates
    nodes in index order.

    Dispatches on ``graph_type`` to one of the concrete samplers below, which are siblings
    rather than subclasses and can also be used directly.
    """

    def __init__(
        self,
        context: Context,
        graph_type: Literal[
            "erdos_renyi",
            "gnr_converging",
            "gnr_diverging",
            "grn_random",
            "cauchy",
        ] = "cauchy",
        p: Optional[float] = None,
    ):
        super().__init__(context)
        self.graph_type = graph_type
        self.p = p

    def _validate_inputs(self, n_nodes: int) -> int:
        n = int(n_nodes)
        if n < 0:
            raise ValueError("n_nodes must be non-negative.")
        if self.graph_type not in (
            "erdos_renyi",
            "gnr_converging",
            "gnr_diverging",
            "grn_random",
            "cauchy",
        ):
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
            "gnr_converging": RandomConvergingGNRDAG,
            "gnr_diverging": RandomDivergingGNRDAG,
            "grn_random": RandomTopologicalGNRDAG,
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


class RandomErdosRenyiDAG(_DAGSampler):
    """
    Samples every possible forward edge independently with probability ``p``.

    This is ``networkx``'s undirected :math:`G(n, p)` graph with each edge oriented from the
    lower to the higher node index. Note that the *directed* variant of ``fast_gnp_random_graph``
    is not equivalent: it samples both orientations of every pair and so produces cyclic digraphs.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        graph = nx.fast_gnp_random_graph(n_nodes, p, seed=self.sampler.rng)
        return self._parent_lists(n_nodes, graph.edges())


class RandomConvergingGNRDAG(_DAGSampler):
    """
    Samples a Growing Network with Redirection DAG, i.e. ``networkx``'s ``gnr_graph``.

    Each new node first chooses an existing node uniformly. With probability ``p``, the choice is
    redirected to that node's target when it has one. The edge points from the new node to the
    chosen older node, so arrival-order labels are reversed here to keep those edges forward edges
    in TabICL's topological index order. Reversing labels is an isomorphism, so the sampled graph
    is distributed exactly as ``nx.gnr_graph(n_nodes, p)``.

    The result is always a tree: it has ``n_nodes - 1`` edges, and every node except the last has
    exactly one child. Unlike the other samplers, ``p`` therefore controls the shape of the graph
    (chain-like vs. star-like) but never its density.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        if n_nodes < 2:
            # nx.gnr_graph always starts from a single node and would return it even for n_nodes=0
            return [[] for _ in range(n_nodes)]
        graph = nx.gnr_graph(n_nodes, p, seed=self.sampler.rng)
        return self._parent_lists(n_nodes, ((n_nodes - 1 - u, n_nodes - 1 - v) for u, v in graph.edges()))


class RandomDivergingGNRDAG(_DAGSampler):
    """
    Samples a GNR skeleton with every edge directed away from the oldest node.

    ``nx.gnr_graph`` points each newly added node toward an older node. Ignoring that direction and
    orienting every edge from the lower (older) arrival-order label to the higher (newer) label
    reverses the native GNR orientation. The result is an out-arborescence: the oldest node is the
    single source and every other node has exactly one parent.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        if n_nodes < 2:
            return [[] for _ in range(n_nodes)]
        graph = nx.gnr_graph(n_nodes, p, seed=self.sampler.rng)
        return self._parent_lists(n_nodes, graph.edges())


class RandomTopologicalGNRDAG(_DAGSampler):
    """
    Samples a GNR skeleton and gives it an independent random acyclic orientation.

    A uniformly random permutation defines the topological order. Each undirected skeleton edge is
    directed from the endpoint occurring earlier in that order to the endpoint occurring later.
    Relabeling nodes by their rank in the order keeps parent indices smaller than child indices while
    allowing multiple sources and sinks, forks, and colliders.
    """

    def sample(self, n_nodes: int, p: float) -> List[List[int]]:
        if n_nodes < 2:
            return [[] for _ in range(n_nodes)]
        graph = nx.gnr_graph(n_nodes, p, seed=self.sampler.rng)
        order = self.sampler.rng.permutation(n_nodes)
        topological_rank = np.empty(n_nodes, dtype=int)
        topological_rank[order] = np.arange(n_nodes)
        return self._parent_lists(
            n_nodes,
            ((topological_rank[u], topological_rank[v]) for u, v in graph.edges()),
        )


class RandomCauchyDAG(_DAGSampler):
    """
    Samples a random DAG based on Cauchy random variables modeling connectivities.

    Each node draws an input and an output importance, whose sum plus a shared offset gives the
    logit of every forward edge. There is no ``networkx`` equivalent, so the adjacency matrix is
    built directly.
    """

    def sample(
        self,
        n_nodes: int,
        cauchy_dag_offset: float = 0.0,
    ) -> List[List[int]]:
        rng = self.sampler.rng
        offset = rng.standard_cauchy() + cauchy_dag_offset
        output_importances = rng.standard_cauchy(n_nodes)
        input_importances = rng.standard_cauchy(n_nodes)
        logits = offset + output_importances[None, :] + input_importances[:, None]
        adjacency_matrix = rng.random(logits.shape) <= expit(logits)
        # only forward edges (i_in < i_out) are kept, so the identity is a topological order
        i_in, i_out = np.nonzero(np.triu(adjacency_matrix, k=1))
        return self._parent_lists(n_nodes, zip(i_in.tolist(), i_out.tolist()))
