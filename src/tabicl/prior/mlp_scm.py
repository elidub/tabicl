from __future__ import annotations

import math
import random
from typing import Dict, Any, Tuple, Optional, List

import torch
from torch import nn

from .utils import GaussianNoise, XSampler


class MLPSCM(nn.Module):
    """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate hidden states of the MLP transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate MLP outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate MLP outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of layers in the MLP transformation network. Must be >= 2.
        Includes the initial linear layer and subsequent blocks of
        (Activation -> Linear -> Noise).

    hidden_dim : int, default=20
        The dimensionality of the hidden representations within the MLP layers.
        If `is_causal=True`, this is automatically increased if it's smaller than
        `num_outputs + 2 * num_features` to ensure enough intermediate variables
        are generated for sampling `X` and `y`.

    mlp_activations : default=nn.Tanh
        The activation function to be used after each linear transformation
        in the MLP layers (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the MLP's linear layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight matrix are initialized with values drawn from
          a normal distribution (scaled by `init_std` and potentially dropout),
          while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `mlp_dropout_prob`.

    mlp_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout. If `True`, `init_std` is
        divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
        initialization method.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each MLP
        layer's linear transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super(MLPSCM, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features

        assert num_layers >= 2, "Number of layers must be at least 2."
        self.num_layers = num_layers

        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            pass
            # self.hidden_dim = max(self.hidden_dim, self.num_outputs + 2 * self.num_features)
        else:
            # In non-causal mode, features are the causes
            self.num_causes = self.num_features

        # Define the input sampler
        self.xsampler = XSampler(
            self.seq_len,
            self.num_causes,
            pre_stats=self.pre_sample_cause_stats,
            sampling=self.sampling,
            device=self.device,
        )

        # Build layers
        layers = [nn.Linear(self.num_causes, self.hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

        # Initialize layers
        self.initialize_parameters()

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, linear transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        linear_layer = nn.Linear(self.hidden_dim, out_dim)

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(torch.zeros(size=(1, out_dim), device=self.device), float(self.noise_std))
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(activation, linear_layer, noise_layer)

    def initialize_parameters(self):
        """Initializes parameters using block-wise dropout or normal initialization."""
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self.initialize_with_block_dropout(param, i)
            else:
                self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        """Initializes parameters using block-wise dropout."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(slice(dim * block, dim * (block + 1)) for dim in block_size)
            nn.init.normal_(
                param[block_slice], std=self.init_std / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1)
            )

    def initialize_normally(self, param, index):
        """Initializes parameters using normal distribution."""
        if param.dim() == 2:  # Applies only to weights, not biases
            dropout_prob = self.mlp_dropout_prob if index > 0 else 0  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / ((1 - dropout_prob) ** 0.5 if self.scale_init_std_by_dropout else 1)
            nn.init.normal_(param, std=std)
            param *= torch.bernoulli(torch.full_like(param, 1 - dropout_prob))

    def forward(self):
        """Generates synthetic data by sampling input features and applying MLP transformations."""
        causes = self.xsampler.sample()  # (seq_len, num_causes)

        # Generate outputs through MLP layers
        outputs = [causes]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        outputs = outputs[2:]  # Start from 2 because the first layer is only linear without activation

        # Handle outputs based on causality
        X, y, indices = self.handle_outputs(causes, outputs)

        # Check for NaNs and handle them by setting to default values
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if self.num_outputs == 1:
            y = y.squeeze(-1)
            
        self.adj = self.get_adjacency_matrix()
        self.indices = indices

        return X, y #, adj, indices

    def handle_outputs(self, causes, outputs) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Handles outputs based on whether causal or not.

        If causal, sample inputs and target from the graph.
        If not causal, directly use causes as inputs and last output as target.

        Parameters
        ----------
        causes : torch.Tensor
            Causes of shape (seq_len, num_causes)

        outputs : list of torch.Tensor
            List of output tensors from MLP layers

        Returns
        -------
        X : torch.Tensor
            Input features (seq_len, num_features)

        y : torch.Tensor
        indices : Tuple[torch.Tensor, torch.Tensor] or None
            The indices selected for X and y from the flattened output vector.
        """
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if self.in_clique:
                # When in_clique=True, features and targets are sampled as a block, ensuring that
                # selected variables may share dense dependencies.
                start = random.randint(0, outputs_flat.shape[-1] - self.num_outputs - self.num_features)
                random_perm = start + torch.randperm(self.num_outputs + self.num_features, device=self.device)
            else:
                # Otherwise, features and targets are randomly and independently selected
                random_perm = torch.randperm(outputs_flat.shape[-1] - 1, device=self.device)

            indices_X = random_perm[self.num_outputs : self.num_outputs + self.num_features]
            if self.y_is_effect:
                # If targets are effects, take last output dims
                indices_y = torch.arange(outputs_flat.shape[-1] - self.num_outputs, outputs_flat.shape[-1], device=self.device)
            else:
                # Otherwise, take from the beginning of the permuted list
                indices_y = random_perm[: self.num_outputs]

            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)

            # Select input features and targets from outputs
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]

            # Adjust indices to account for causes and first layer at the beginning
            offset = self.num_causes + self.hidden_dim
            indices_X += offset
            indices_y += offset
            
            return X, y, (indices_X, indices_y)
        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            y = outputs[-1]
            return X, y, None

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Constructs the adjacency matrix of the COMPLETE MLP graph.
        
        Nodes are ordered as:
        [Causes (Inputs) | Output of Layer 0 | Output of Layer 1 | ... | Output of Layer N]

        Returns
        -------
        adj : torch.Tensor
            A square matrix of shape (total_dim, total_dim).
        """
        if not self.is_causal:
            return None
        else:
            assert self.is_causal is True
        

        # Calculate total dimension:
        # 1. num_causes (Inputs)
        # 2. num_layers * hidden_dim (Outputs of all layers)
        total_dim = self.num_causes + (self.num_layers * self.hidden_dim)
        
        adj = torch.zeros((total_dim, total_dim), device=self.device)

        # --- 1. Handle the first layer (Causes -> Layer 0) ---
        # self.layers[0] is just nn.Linear(num_causes, hidden_dim)
        # It connects the 'Causes' block to the 'Layer 0 Output' block.
        
        weight_0 = self.layers[0].weight.detach().T  # Shape: (num_causes, hidden_dim)
        
        # Source: The first 'num_causes' indices
        src_start = 0
        src_end = self.num_causes
        
        # Dest: The block immediately following the causes
        dst_start = self.num_causes
        dst_end = self.num_causes + self.hidden_dim
        
        adj[src_start:src_end, dst_start:dst_end] = weight_0

        # --- 2. Handle subsequent layers (Layer i -> Layer i+1) ---
        # self.layers[1:] are Sequential(Activation, Linear, Noise)
        # They connect 'Layer i Output' to 'Layer i+1 Output'
        
        # The hidden blocks start after the causes
        hidden_offset = self.num_causes
        
        for i in range(1, self.num_layers):
            # The linear layer is at index 1 in the Sequential block
            weight = self.layers[i][1].weight.detach().T # Shape: (hidden_dim, hidden_dim)
            
            # Source Block: Output of Layer (i-1)
            # For i=1, this is Layer 0 (which starts at hidden_offset + 0)
            src_s = hidden_offset + (i - 1) * self.hidden_dim
            src_e = src_s + self.hidden_dim
            
            # Dest Block: Output of Layer i
            dst_s = hidden_offset + i * self.hidden_dim
            dst_e = dst_s + self.hidden_dim
            
            adj[src_s:src_e, dst_s:dst_e] = weight
            
        return adj

    # def get_adjacency_matrix(self) -> torch.Tensor:
    #     """
    #     Constructs the adjacency matrix of the internal MLP graph that corresponds
    #     to the flattened outputs used for X and y selection.

    #     Returns
    #     -------
    #     adj : torch.Tensor
    #         A square matrix where adj[i, j] != 0 implies a directed edge from node i to node j.
    #         The indices correspond to the flattened feature vector `outputs[2:]`.
    #     """
    #     if not self.is_causal:
    #         return None

    #     # We need to map the layers to the blocks in the flattened output.
    #     # outputs[2] corresponds to block 0
    #     # outputs[3] corresponds to block 1
    #     # ...
    #     # The connection between outputs[k] (block k-2) and outputs[k+1] (block k-1)
    #     # is determined by self.layers[k].
        
    #     # Calculate total dimension of the adjacency matrix
    #     # In is_causal=True, all intermediate layers have size hidden_dim.
    #     # We start slicing outputs from index 2 up to the end.
    #     num_blocks = len(self.layers) - 1 # Corresponds to outputs[2:]
    #     total_dim = num_blocks * self.hidden_dim
        
    #     adj = torch.zeros((total_dim, total_dim), device=self.device)

    #     # Iterate through the layers that connect the blocks within our scope
    #     # We start from layer 2 because layer 0 connects causes->L0, layer 1 connects L0->L1.
    #     # outputs[2] is L1 output. outputs[3] is L2 output.
    #     # So we look at connection L1 -> L2, which is layers[2].
    #     for k in range(2, len(self.layers)):
    #         # Determine source and target block indices in the flattened vector
    #         src_block_idx = k - 2
    #         tgt_block_idx = k - 1
            
    #         # Get the weight matrix. Layer structure: Sequential(Activation, Linear, Noise)
    #         # The Linear layer is at index 1.
    #         # PyTorch Linear weights are (out_features, in_features).
    #         # Adjacency matrix usually expects (source, target) -> we need Transpose.
    #         weight = self.layers[k][1].weight.detach()
            
    #         start_src = src_block_idx * self.hidden_dim
    #         end_src = start_src + self.hidden_dim
            
    #         start_dst = tgt_block_idx * self.hidden_dim
    #         end_dst = start_dst + self.hidden_dim
            
    #         adj[start_src:end_src, start_dst:end_dst] = weight.T
            
    #     return adj
        
    


    # def _get_full_connectivity(self) -> torch.Tensor:
    #     """
    #     Computes the reachability matrix for all hidden neurons in the flattened output.
    #     Cached because it's computationally expensive and only changes if weights change.
    #     """

    #     # Gather weights corresponding to outputs_flat (Layer 1 onward)
    #     # Note: self.layers[0] is input->L1 (linear only).
    #     # self.layers[1:] are blocks (Act->Lin->Noise).
        
    #     # We need the weights connecting the layers present in `outputs`
    #     # `outputs` in forward starts after layers[0] is applied.
    #     # It contains output of layers[0] (which is inputs to layers[1]), 
    #     # then output of layers[1], etc.
    #     # Wait, forward logic: outputs = outputs[2:].
    #     # outputs[0] = causes
    #     # outputs[1] = layers[0](causes) -> This is "Layer 1 Pre-activation"
    #     # outputs[2] = layers[1](outputs[1]) -> This is "Layer 1 Post-activation" (roughly)
        
    #     weights = []
    #     for i in range(1, len(self.layers)):
    #         # Each block is Activation -> Linear -> Noise
    #         # We want the Linear weight.
    #         block = self.layers[i]
    #         weights.append(block[1].weight.T) # (in_features, out_features)

    #     total_neurons = sum(w.shape[1] for w in weights)
        
    #     # Create the full adjacency of the hidden neurons
    #     # Since it's feedforward, we can fill it block by block
    #     full_adj = torch.zeros((total_neurons, total_neurons), device=self.device)
        
    #     current_idx = 0
    #     layer_start_indices = [0]
        
    #     # Fill direct connections (A)
    #     for i, w in enumerate(weights):
    #         if i < len(weights) - 1:
    #             this_layer_size = w.shape[1]
    #             next_layer_size = weights[i+1].shape[1]
    #             w_next = weights[i+1] # Connects current to next
                
    #             row_start = current_idx
    #             row_end = current_idx + this_layer_size
    #             col_start = row_end
    #             col_end = col_start + next_layer_size
                
    #             full_adj[row_start:row_end, col_start:col_end] = w_next
                
    #             current_idx += this_layer_size
    #             layer_start_indices.append(current_idx)

    #     # Compute Reachability / Path Weights (A + A^2 + A^3...)
    #     # Since it's a DAG, we can just perform matrix multiplications for the depth
    #     path_matrix = full_adj.clone()
    #     temp_adj = full_adj.clone()
        
    #     for _ in range(len(weights) - 1):
    #         temp_adj = torch.matmul(temp_adj, full_adj)
    #         path_matrix += temp_adj
            
    #     return path_matrix

    # def get_adjacency_matrix_v2(self, idx_X: Optional[torch.Tensor], idx_y: Optional[torch.Tensor]) -> torch.Tensor:
    #     """Computes the subgraph adjacency matrix for the specific X and y indices."""
        
    #     # Case 1: Non-Causal (Direct Mapping)
    #     if not self.is_causal:
    #         adj = torch.zeros((self.num_features + self.num_outputs, self.num_features + self.num_outputs), device=self.device)
            
    #         # Chain multiply all weights to get effective X -> y influence
    #         w_total = self.layers[0].weight.T
    #         for i in range(1, len(self.layers)):
    #             w_total = torch.matmul(w_total, self.layers[i][1].weight.T)
                
    #         adj[:self.num_features, self.num_features:] = w_total
    #         return adj

    #     # Case 2: Causal (Subgraph selection)
    #     path_matrix = self._get_full_connectivity()
        
    #     # Combined indices: [X_indices, y_indices]
    #     all_indices = torch.cat([idx_X, idx_y])
    #     n_total = len(all_indices)
        
    #     final_adj = torch.zeros((n_total, n_total), device=self.device)
        
    #     # Extract sub-matrix
    #     # We want final_adj[i, j] = path_matrix[all_indices[i], all_indices[j]]
    #     # But we must ensure i != j (no self loops in DAG definition usually)
        
    #     # Vectorized extraction
    #     # meshgrid to get all pairs (i, j)
    #     grid_x, grid_y = torch.meshgrid(all_indices, all_indices, indexing="ij")
        
    #     # Extract weights from path matrix
    #     extracted_weights = path_matrix[grid_x, grid_y]
        
    #     # Remove self-loops (diagonal)
    #     mask = torch.eye(n_total, device=self.device).bool()
    #     extracted_weights.masked_fill_(mask, 0.0)
        
    #     final_adj = extracted_weights
        
    #     return final_adj
