import argparse
from dataclasses import dataclass
from typing import Literal, Optional


def str2bool(value):
    return value.lower() == "true"


@dataclass
class PriorConfig:
    """Configuration options for the prior."""

    add_gaussian_noise: bool = False
    # range of the (log-uniform, meta-sampled) noise level when add_gaussian_noise=True
    gaussian_noise_min_sigma: float = 1e-5
    gaussian_noise_max_sigma: float = 1.0
    # whether a node's noise is in its own columns, as in a textbook SCM; by default it
    # is added after the columns are read out, so only the node's children see it
    noise_in_columns: bool = False
    use_node_feature_importance: bool = True
    use_node_importance: bool = True
    use_node_l2_norm: bool = True
    allow_act_warping: bool = False
    allow_kumaraswamy_warping: bool = True
    disallow_y_warping: bool = False
    # maximum number of discretization levels in random discretization function
    max_discretization_cardinality: int = 256
    filter_unpredictable_datasets: bool = False
    filter_unpredictable_graphs: bool = False
    min_n_nodes: int = 2
    max_n_nodes: int = 32
    graph_type: Literal[
        'cauchy',
        'erdos_renyi',
        'gnr_converging',
        'gnr_diverging',
        'gnr_random',
    ] = 'cauchy'
    # Edge probability control. Either a fixed float (a fixed edge probability,
    # or the additive offset for the 'cauchy' graph type), or a distribution
    # spec dict tagged by 'dist' to sample it per dataset, e.g.
    # {'dist': 'beta', 'alpha': 2, 'beta': 6}.
    graph_edge_prob: float | dict = 0.0
    meta_sampling_mode: Literal['meta', 'local', 'global'] = 'meta'
    random_matrix_types: Literal['default', 'gaussian'] = 'default'
    fct_types: str = 'default'
    multi_fct_types: Literal['default', 'concat', 'agg'] = 'default'
    cat_modes: str = 'default'
    random_weights_types: Literal['simple', 'uniform'] = 'simple'
    subsample_feature_nodes: bool = True
    seq_lens_multiple_of: int = 1
    use_corrected_num_converters: bool = False
    use_corrected_cat_sizes: bool = False
    ensure_iid: bool = False
    remove_trivial_datasets: bool = False
    trivial_dataset_threshold: float = 0.05
    use_corrected_cat_meta_sampling: bool = False
    # Only sample the graph structure, not the data on it: skips the graph
    # function and every data-dependent validity check, and returns columns with
    # zero rows. For development/analysis of the graph prior itself.
    graph_only: bool = False
    # With return_graph_info, also return the fitted graph function (the functions
    # and noise levels the data was generated with) under "graph_function". For
    return_graph_function: bool = False
    # analysis: it is pickled back from every worker, up to a few MB per dataset.
    # Draw a node's noise level, importances and root map (the point type and the random
    # function on top of the points) once, on its first call, instead of on every call.
    # Then calling a fitted graph function again (ensure_iid, or pool_level='scm') gives
    # new rows of the same SCM; the points and the noise themselves are always new.
    hold_node_draws: bool = False
    # Keep part of the SCM fixed across datasets: each dataset takes one of pool_size
    # variants of it, variant k drawn under its own seed from (pool_seed, k), so every
    # process draws the same variants. pool_level is what is fixed: 'graph' (the DAG and
    # its node count), 'assignment' (also the node each column is read from) or 'scm'
    # (also the functions and noise levels; needs hold_node_draws). Below 'graph', a
    # variant is drawn for the dataset's columns, so keep their number and types fixed.
    # pool_size None: no pool, every dataset draws everything afresh; once it is set,
    # pool_level is required.
    pool_size: Optional[int] = None
    pool_level: Optional[Literal['graph', 'assignment', 'scm']] = None
    pool_seed: int = 0
    # Options to strip the prior down to a plain SCM on the graph (all off by default,
    # which keeps the TabICLv2 behavior):
    # every node is a single value instead of a matrix of latent dimensions
    # (needs only numerical columns: allow_categorical_features=False, and
    # threshold_target=True for classification)
    scalar_nodes: bool = False
    # every column (features and target) is read from its own node, instead of
    # nodes sampled with replacement; graphs need at least as many nodes as columns
    unique_column_nodes: bool = False
    # 'default' samples root inputs from several point distributions, 'gaussian' only
    # from a standard Gaussian
    root_point_types: Literal['default', 'gaussian'] = 'default'
    # whether some features are categorical
    allow_categorical_features: bool = True
    # classification labels from quantiles of a numerical target instead of a
    # categorical converter on its node
    threshold_target: bool = False
    # whether to clip outliers before standardizing x (and y for regression)
    remove_outliers: bool = True

    @staticmethod
    def from_args(args) -> "PriorConfig":
        return PriorConfig(add_gaussian_noise=args.graph_noise,
                           use_node_feature_importance=args.use_node_feature_importance,
                           use_node_importance=args.use_node_importance,
                           use_node_l2_norm=args.use_node_l2_norm,
                           allow_act_warping=args.allow_act_warping,
                           filter_unpredictable_datasets=args.filter_unpredictable_datasets,
                           filter_unpredictable_graphs=args.filter_unpredictable_graphs,
                           min_n_nodes=args.min_n_nodes,
                           max_n_nodes=args.max_n_nodes,
                           graph_edge_prob=args.graph_edge_prob,
                           meta_sampling_mode=args.meta_sampling_mode,
                           random_matrix_types=args.random_matrix_types,
                           fct_types=args.fct_types,
                           multi_fct_types=args.multi_fct_types,
                           cat_modes=args.cat_modes,
                           random_weights_types=args.random_weights_types,
                           subsample_feature_nodes=args.subsample_feature_nodes,
                           allow_kumaraswamy_warping=args.allow_kumaraswamy_warping,
                           disallow_y_warping=args.disallow_y_warping,
                           seq_lens_multiple_of=args.seq_lens_multiple_of,
                           use_corrected_num_converters=args.use_corrected_num_converters,
                           use_corrected_cat_sizes=args.use_corrected_cat_sizes,
                           ensure_iid=args.ensure_iid,
                           remove_trivial_datasets=args.remove_trivial_datasets,
                           trivial_dataset_threshold=args.trivial_dataset_threshold,
                           use_corrected_cat_meta_sampling=args.use_corrected_cat_meta_sampling,
                           gaussian_noise_min_sigma=args.gaussian_noise_min_sigma,
                           gaussian_noise_max_sigma=args.gaussian_noise_max_sigma,
                           noise_in_columns=args.noise_in_columns,
                           scalar_nodes=args.scalar_nodes,
                           unique_column_nodes=args.unique_column_nodes,
                           root_point_types=args.root_point_types,
                           allow_categorical_features=args.allow_categorical_features,
                           threshold_target=args.threshold_target,
                           remove_outliers=args.remove_outliers,
                           hold_node_draws=args.hold_node_draws,
                           pool_size=args.pool_size,
                           pool_level=args.pool_level,
                           pool_seed=args.pool_seed)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--graph_noise",
            default=False,
            type=str2bool,
            help="Whether to add Gaussian noise to the nodes of DAG when using graph_scm prior",
        )
        parser.add_argument(
            "--use_node_feature_importance",
            default=True,
            type=str2bool,
            help="Whether to sample random feature importances on nodes when using the graph_scm prior",
        )
        parser.add_argument(
            "--use_node_importance",
            default=True,
            type=str2bool,
            help="Whether to sample random node importances when using the graph_scm prior",
        )
        parser.add_argument(
            "--use_node_l2_norm",
            default=True,
            type=str2bool,
            help="Whether to L2-normalize the features on the node when using the graph_scm prior",
        )
        parser.add_argument(
            "--allow_act_warping",
            default=False,
            type=str2bool,
            help="Whether to include random activations as a numerical feature warping method in the graph_scm prior",
        )
        parser.add_argument(
            "--allow_kumaraswamy_warping",
            default=True,
            type=str2bool,
            help="Whether to include Kumaraswamy warping in the graph_scm prior",
        )
        parser.add_argument(
            "--disallow_y_warping",
            default=False,
            type=str2bool,
            help="Whether to exclude warping on y",
        )
        parser.add_argument(
            "--filter_unpredictable_datasets",
            default=False,
            type=str2bool,
            help="Whether to remove datasets from the prior where a simple ExtraTrees is not better than a dummy model",
        )
        parser.add_argument(
            "--filter_unpredictable_graphs",
            default=False,
            type=str2bool,
            help="Whether to filter graphs from the prior where y is independent of x",
        )
        parser.add_argument(
            "--min_n_nodes",
            default=2,
            type=int,
            help="Minimum number of nodes in the causal graph",
        )
        parser.add_argument(
            "--max_n_nodes",
            default=32,
            type=int,
            help="Maximum number of nodes in the causal graph",
        )
        parser.add_argument(
            "--graph_edge_prob",
            default=0.0,
            type=float,
            help="Fixed edge probability, or the additive Cauchy DAG offset. To sample "
                 "the probability from a distribution, set a dist-spec dict on the "
                 "PriorConfig directly (not supported via this CLI flag).",
        )
        parser.add_argument(
            "--meta_sampling_mode",
            default="meta",
            type=str,
            help="One of 'meta' (default), 'local' (will always sample iid stuff), "
                 "'global' (will always sample the same thing)"
        )
        parser.add_argument(
            "--random_matrix_types",
            default="default",
            type=str,
            help="Which types of random matrix to sample ('default' or 'gaussian' for only gaussian)"
        )
        parser.add_argument(
            "--fct_types",
            default="default",
            type=str,
            help="Which types of functions to sample ('default', 'tabpfnv2' for only NN/Tree/Discretization)"
        )
        parser.add_argument(
            "--multi_fct_types",
            default="default",
            type=str,
            help="Which types of multi-functions to sample ('default' for both, 'concat', 'agg')"
        )
        parser.add_argument(
            "--cat_modes",
            default="default",
            type=str,
            help="Which categorical discretization modes to use ('default', 'neighbor', 'softmax', 'neighbor_id', "
                 "'neighbor_disc', 'neighbor_func', 'neighbor_int', 'softmax_id', 'softmax_disc', 'softmax_int')"
        )
        parser.add_argument(
            "--random_weights_types",
            default="simple",
            type=str,
            help="Which types of random weights to sample ('simple' or 'uniform')"
        )
        parser.add_argument(
            "--subsample_feature_nodes",
            default=True,
            type=str2bool,
            help="Whether to sample a subset of nodes such that features can only be placed on those nodes",
        )
        parser.add_argument(
            "--seq_lens_multiple_of",
            default=1,
            type=int,
            help="Sequence lengths are rounded to multiples of this number",
        )
        parser.add_argument(
            "--use_corrected_num_converters",
            default=False,
            type=str2bool,
            help="Whether to apply warping to the column output instead of the propagated node values",
        )
        parser.add_argument(
            "--use_corrected_cat_sizes",
            default=False,
            type=str2bool,
            help="Whether to allow larger categorical cardinalities than 9",
        )
        parser.add_argument(
            "--ensure_iid",
            default=False,
            type=str2bool,
            help="Whether to ensure that the data is IID by first fitting all the transforms on a separate forward pass.",
        )
        parser.add_argument(
            "--remove_trivial_datasets",
            default=False,
            type=str2bool,
            help="Whether to remove datasets from the prior where a simple ExtraTrees is almost perfect",
        )
        parser.add_argument(
            "--trivial_dataset_threshold",
            default=0.05,
            type=float,
            help="If remove_trivial_dataset=True, datasets are filtered if the extratrees RMSE is below the threshold times the dummy RMSE",
        )
        parser.add_argument(
            "--use_corrected_cat_meta_sampling",
            default=False,
            type=str2bool,
            help="Whether to use the corrected meta-sampling for categoricals.",
        )
        parser.add_argument(
            "--scalar_nodes",
            default=False,
            type=str2bool,
            help="Whether every node is a single value instead of a matrix of latent dimensions",
        )
        parser.add_argument(
            "--unique_column_nodes",
            default=False,
            type=str2bool,
            help="Whether every column is read from its own node instead of nodes sampled with replacement",
        )
        parser.add_argument(
            "--root_point_types",
            default="default",
            type=str,
            help="Which point distributions root nodes sample from ('default' or 'gaussian' for only gaussian)",
        )
        parser.add_argument(
            "--allow_categorical_features",
            default=True,
            type=str2bool,
            help="Whether some features are categorical",
        )
        parser.add_argument(
            "--threshold_target",
            default=False,
            type=str2bool,
            help="Whether classification labels are quantiles of a numerical target",
        )
        parser.add_argument(
            "--remove_outliers",
            default=True,
            type=str2bool,
            help="Whether to clip outliers before standardizing",
        )
        parser.add_argument(
            "--gaussian_noise_min_sigma",
            default=1e-5,
            type=float,
            help="Lower end of the noise level range when --graph_noise=True",
        )
        parser.add_argument(
            "--gaussian_noise_max_sigma",
            default=1.0,
            type=float,
            help="Upper end of the noise level range when --graph_noise=True",
        )
        parser.add_argument(
            "--noise_in_columns",
            default=False,
            type=str2bool,
            help="Whether a node's noise is in its own columns instead of only in what its children see",
        )
        parser.add_argument(
            "--hold_node_draws",
            default=False,
            type=str2bool,
            help="Whether a node draws its noise level, importances and root map once instead of on every call",
        )
        parser.add_argument(
            "--pool_size",
            default=None,
            type=int,
            help="Number of fixed variants of part of the SCM (--pool_level) that datasets draw from; None: no pool",
        )
        parser.add_argument(
            "--pool_level",
            default=None,
            type=str,
            help="What a pool fixes: 'graph', 'assignment' (also the column nodes) or 'scm' (also the functions)",
        )
        parser.add_argument(
            "--pool_seed",
            default=0,
            type=int,
            help="Seed the pool's variants are drawn from",
        )
