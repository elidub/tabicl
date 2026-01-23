from typing import List, Dict
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


def _marginalize_nodes_array(adj_matrix: np.ndarray, indices_to_remove: np.ndarray | List[int]) -> np.ndarray:
    """
    Marginalizes (removes) nodes from a DAG adjacency matrix while preserving 
    path connectivity via the outer product shortcut.

    Args:
        adj_matrix (np.ndarray): Shape (N, N). Adjacency matrix (0/1 or bool).
                                 adj[i, j] = 1 means edge i -> j.
        indices_to_remove (list or np.ndarray): Indices of nodes to drop.

    Returns:
        np.ndarray: The reduced adjacency matrix.
    """
    if isinstance(indices_to_remove, list): indices_to_remove = np.array(indices_to_remove)
    
    # 1. Work on a boolean copy. 
    # Boolean logic is faster and cleaner for connectivity checks.
    m = adj_matrix.copy().astype(bool)
    
    # 2. Phase 1: The "Fill-in" Loop
    # We process the graph BEFORE deleting any rows/cols. 
    # This prevents index shifting issues (e.g., removing index 2 doesn't make index 3 become 2).
    for k in indices_to_remove:
        # Get all parents of node k (Incoming edges: Column k)
        parents = m[:, k]
        
        # Get all children of node k (Outgoing edges: Row k)
        children = m[k, :]
        
        # === THE SHORTCUT: Rank-One Update / Outer Product ===
        # If Parent(i) -> k AND k -> Child(j), create edge Parent(i) -> Child(j).
        # np.outer creates a matrix where entry (i, j) is True only if 
        # parents[i] is True AND children[j] is True.
        fill_in = np.outer(parents, children)
        
        # Update the matrix using Logical OR (Accumulate the new paths)
        m |= fill_in

    # 3. Phase 2: "Soft" Removal (Zeroing out)
    # Instead of slicing the matrix, we set the rows and columns 
    # of the marginalized nodes to False (0).
    
    # Remove all outgoing edges from the marginalized nodes
    m[indices_to_remove, :] = False
    
    # Remove all incoming edges to the marginalized nodes
    m[:, indices_to_remove] = False
    
    # Note: We zero out at the END. This ensures that if you remove a chain 
    # like A->B->C (removing B and C), the logic still sees the connections 
    # needed to bridge A->C before B and C are wiped.

    return m.astype(int)

        
    # 3. Phase 2: Removal
    # Now that all necessary bridges are built, we safely remove the rows/cols.
    n_feat = adj_matrix.shape[0]
    keep_mask = np.ones(n_feat, dtype=bool)
    keep_mask[indices_to_remove] = False
    
    # Slice the matrix: First filter rows, then filter columns
    m_reduced = m[keep_mask][:, keep_mask]
    
    # Return as integer (0/1) to match standard adjacency format
    return m_reduced.astype(int)

# def marginalize_nodes(adj_matrix: torch.Tensor, indices_to_remove: torch.Tensor) -> torch.Tensor:
#     """
#     Marginalizes (removes) nodes from a DAG adjacency matrix while preserving 
#     path connectivity via the outer product shortcut.

#     Args:
#         adj_matrix (torch.Tensor): Shape (N, N). Adjacency matrix (0/1 or bool).
#                                  adj[i, j] = 1 means edge i -> j.
#         indices_to_remove (torch.Tensor): Indices of nodes to drop.

#     Returns:
#         torch.Tensor: The reduced adjacency matrix.
#     """
#     # 1. Work on a boolean copy. 
#     # Boolean logic is faster and cleaner for connectivity checks.
#     m = adj_matrix.clone().bool()
    
#     # 2. Phase 1: The "Fill-in" Loop
#     # We process the graph BEFORE deleting any rows/cols. 
#     # This prevents index shifting issues (e.g., removing index 2 doesn't make index 3 become 2).
#     for k in indices_to_remove:
#         # Get all parents of node k (Incoming edges: Column k)
#         parents = m[:, k]
        
#         # Get all children of node k (Outgoing edges: Row k)
#         children = m[k, :]
        
#         # === THE SHORTCUT: Rank-One Update / Outer Product ===
#         # If Parent(i) -> k AND k -> Child(j), create edge Parent(i) -> Child(j).
#         # torch.outer creates a matrix where entry (i, j) is True only if 
#         # parents[i] is True AND children[j] is True.
#         fill_in = torch.outer(parents, children)
        
#         # Update the matrix using Logical OR (Accumulate the new paths)
#         m |= fill_in

#     # 3. Phase 2: "Soft" Removal (Zeroing out)
#     # Instead of slicing the matrix, we set the rows and columns 
#     # of the marginalized nodes to False (0).
    
#     # Remove all outgoing edges from the marginalized nodes
#     m[indices_to_remove, :] = False
    
#     # Remove all incoming edges to the marginalized nodes
#     m[:, indices_to_remove] = False
    
#     # Note: We zero out at the END. This ensures that if you remove a chain 
#     # like A->B->C (removing B and C), the logic still sees the connections 
#     # needed to bridge A->C before B and C are wiped.

#     return m.to(adj_matrix.dtype)

        
#     # 3. Phase 2: Removal
#     # Now that all necessary bridges are built, we safely remove the rows/cols.
#     n_feat = adj_matrix.shape[0]
#     keep_mask = torch.ones(n_feat, dtype=bool)
#     keep_mask[indices_to_remove] = False
    
#     # Slice the matrix: First filter rows, then filter columns
#     m_reduced = m[keep_mask][:, keep_mask]
    
#     # Return as integer (0/1) to match standard adjacency format
#     return m_reduced.astype(int)

# def marginalize_2hop_iterative(adj_matrix: torch.Tensor, indices_to_remove: torch.Tensor):
#     """
#     Marginalizes nodes by iteratively adding the specific 2-hop neighborhood 
#     generated by the removed node.
#     """
#     # 1. Work on a copy (boolean is best for connectivity)
#     m = adj_matrix.clone().bool()
    
#     # 2. Iteratively process ONLY the nodes we want to remove
#     for k in indices_to_remove:
#         # --- The "2-Hop" Logic ---
        
#         # Step A: Find incoming paths (1st hop): i -> k
#         # We perform a slice k:k+1 to keep dimensions (N, 1)
#         incoming_paths = m[:, k:k+1] 
        
#         # Step B: Find outgoing paths (2nd hop): k -> j
#         # We perform a slice k:k+1 to keep dimensions (1, N)
#         outgoing_paths = m[k:k+1, :]
        
#         # Step C: Combine to find the 2-hop shortcuts: i -> j
#         # Matrix Multiplication: (N, 1) @ (1, N) = (N, N)
#         # This calculates exactly the paths that go i -> k -> j
#         two_hop_shortcuts = incoming_paths.to(torch.int) @ outgoing_paths.to(torch.int)
        
#         # Step D: Add these 2-hop shortcuts to our graph
#         # This effectively "bridges" the gap left by node k
#         m |= two_hop_shortcuts.to(torch.bool)

#     # 3. Apply the "Only for nodes we keep" constraint
#     # We zero out the rows/cols of removed nodes at the VERY END.
#     # Why? Because a node we plan to remove later might be a necessary 
#     # bridge for a node we are removing now.
    
#     # Remove outgoing edges from marginalized nodes
#     m[indices_to_remove, :] = 0
#     # Remove incoming edges to marginalized nodes
#     m[:, indices_to_remove] = 0
    
#     return m.to(adj_matrix.dtype)

