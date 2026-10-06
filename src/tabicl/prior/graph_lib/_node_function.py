from typing import List, Dict, Tuple
import torch

from tabicl.prior.graph_lib._base import RandomTransformer, Context, FeatureSpec
from tabicl.prior.graph_lib._activation import Standardize, L2Normalize
from tabicl.prior.graph_lib._converter import CategoricalConverter, NumericalConverter
from tabicl.prior.graph_lib._multi_function import RandomMultiFunction
from tabicl.prior.graph_lib._points import RandomPoints
from tabicl.prior.graph_lib._weights import RandomWeights


class RandomNodeFunction(RandomTransformer):
    """
    Computes the node matrix from the parents' matrices, and optionally extracts columns for the dataset.
    """
    def __init__(self, context: Context, feature_specs: Dict[str, FeatureSpec]):
        super().__init__(context)
        self.feature_specs = feature_specs

    def _fit(self, x: List[torch.Tensor], n_samples: int):
        self.converters_ = dict()
        for feature_name, feature_spec in self.feature_specs.items():
            cat_size = feature_spec.cat_size
            converter = (
                CategoricalConverter(self.context, n_values=cat_size)
                if cat_size > 0
                else NumericalConverter(self.context,
                                        disallow_warping=self.config.disallow_y_warping and feature_spec.group == 'y')
            )
            self.converters_[feature_name] = converter

        if self.config.scalar_nodes:
            if any(converter.get_n_features() != 1 for converter in self.converters_.values()):
                raise ValueError(
                    "scalar_nodes=True needs numerical columns only: set allow_categorical_features=False, "
                    "and threshold_target=True for classification."
                )
            self.n_features_ = 1
        else:
            self.n_features_ = self.sampler.randint("node_n_latent_features", 1, 32, use_log=True)
            if len(self.converters_) > 0:
                self.n_features_ += sum(converter.get_n_features() for converter in self.converters_.values())
        self.random_points_ = RandomPoints(self.context)
        self.random_func_ = RandomMultiFunction(self.context, out_features=self.n_features_)

        self.std_ = Standardize(self.context)
        self.l2_norm_ = L2Normalize(self.context)
        self.random_weights_ = RandomWeights(self.context)

    def _transform(self, x: List[torch.Tensor], n_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(x) == 0:
            x = self.random_points_.sample(n_samples, self.n_features_)
        else:
            x = self.random_func_(x)

        if self.config.use_node_feature_importance:
            x = self.std_(x)
            weights = self._draw("feature_importance", lambda: self.random_weights_.sample(1, x.shape[1]).squeeze(0))
            weights = weights / weights.square().mean().sqrt()
            x = x * weights[None, :]
        if self.config.use_node_l2_norm:
            x = self.l2_norm_(x)

        if self.config.add_gaussian_noise and self.config.noise_in_columns:
            x = self._add_noise(x)

        out_features = dict()
        start_idx = 0
        for name, converter in self.converters_.items():
            end_idx = start_idx + converter.get_n_features()
            assert end_idx <= self.n_features_
            tens, feat = converter(x[:, start_idx:end_idx])
            x[:, start_idx:end_idx] = tens
            out_features[name] = feat
            # a scalar node gives the same value to each of its columns
            start_idx = 0 if self.config.scalar_nodes else end_idx

        if self.config.add_gaussian_noise and not self.config.noise_in_columns:
            x = self._add_noise(x)

        if self.config.use_node_importance:
            x = x * self._draw(
                "node_importance", lambda: self.sampler.numerical("node_importance", low=0.1, high=10.0, use_log=True)
            )

        return x, out_features

    def _draw(self, name: str, draw):
        """``draw()``, or with ``hold_node_draws`` the value it gave on this node's first call."""
        if not self.config.hold_node_draws:
            return draw()
        held = self.__dict__.setdefault("held_draws_", {})
        if name not in held:
            held[name] = draw()
        return held[name]

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise_level = self._draw(
            "noise_level",
            lambda: self.sampler.numerical(
                "gaussian_noise_sigma",
                low=self.config.gaussian_noise_min_sigma,
                high=self.config.gaussian_noise_max_sigma,
                use_log=True,
            ),
        )
        self.noise_level_ = noise_level  # kept for inspection; redrawn on every call unless held
        return x + noise_level * torch.randn_like(x)
