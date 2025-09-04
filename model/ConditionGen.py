import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from torch_geometric.utils import (to_dense_adj, degree, to_undirected, remove_self_loops, 
                                   homophily, k_hop_subgraph, subgraph)
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as sp


def normalize_l2(x):
    """L2 normalization without sklearn dependency."""
    if isinstance(x, np.ndarray):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-8)
    else:  # torch tensor
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + 1e-8)


class ZeroMLP(nn.Module):
    """
    ZeroMLP for ControlNet functionality - generates FiLM parameters.
    Outputs [gamma, beta] for Feature-wise Linear Modulation.
    """
    def __init__(self, in_dim, hidden=128, out_dim=None, coeff_dim=None):
        super(ZeroMLP, self).__init__()
        if out_dim is None and coeff_dim is not None:
            out_dim = 2 * coeff_dim  # gamma and beta
        elif out_dim is None:
            out_dim = 2 * in_dim  # default behavior
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        
        # Initialize with zeros for stable training (Zero initialization)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize last layer bias to gamma=1, beta=0 for identity modulation at start
        # This ensures FiLM path is active from the beginning
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        if linear_layers:
            last_layer = linear_layers[-1]
            half = last_layer.bias.shape[0] // 2
            with torch.no_grad():
                last_layer.bias[:half].fill_(1.0)  # gamma initialized to 1
                last_layer.bias[half:].fill_(0.0)  # beta initialized to 0
    
    def forward(self, P_cond):
        """
        Args:
            P_cond: Condition positional embedding (N x d_pe) or graph-level representation
        Returns:
            gamma, beta: FiLM parameters for modulation
        """
        if P_cond.dim() == 2:
            # Average over nodes to get graph-level representation
            P_cond = P_cond.mean(dim=0, keepdim=True)  # (1, d_pe)
        
        out = self.mlp(P_cond)  # (1, out_dim)
        mid = out.shape[1] // 2
        gamma = out[:, :mid]  # (1, coeff_dim)
        beta = out[:, mid:]   # (1, coeff_dim)
        return gamma, beta


def compute_laplacian_pe(edge_index, num_nodes, k_eigs=32, normalization='sym'):
    """
    Compute Laplacian positional encoding with sign alignment.
    
    Args:
        edge_index: Edge indices
        num_nodes: Number of nodes
        k_eigs: Number of eigenvectors to compute
        normalization: Type of Laplacian normalization ('sym' or 'rw')
    
    Returns:
        pe: Positional embedding (num_nodes, k_eigs)
    """
    # Convert to scipy sparse matrix
    adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    
    # Add self-loops if not present
    adj_scipy = adj_scipy + sp.eye(num_nodes)
    
    # Compute degree matrix
    degrees = np.array(adj_scipy.sum(axis=1)).flatten()
    
    if normalization == 'sym':
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = sp.diags(np.power(degrees, -0.5))
        laplacian = sp.eye(num_nodes) - deg_inv_sqrt @ adj_scipy @ deg_inv_sqrt
    else:  # 'rw'
        # Random walk normalization: I - D^{-1} A
        deg_inv = sp.diags(np.power(degrees, -1.0))
        laplacian = sp.eye(num_nodes) - deg_inv @ adj_scipy
    
    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(
            laplacian, k=min(k_eigs, num_nodes-1), which='SM', return_eigenvectors=True
        )
    except:
        # Fallback to dense computation for small graphs
        laplacian_dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
        # Sort by eigenvalue
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx[:k_eigs]]
        eigenvectors = eigenvectors[:, idx[:k_eigs]]
    
    # Sign alignment to ensure consistency
    eigenvectors = _sign_align_pe(eigenvectors)
    
    # L2 normalize
    eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-8)
    
    return torch.FloatTensor(eigenvectors)


def _sign_align_pe(eigenvectors):
    """
    Align eigenvector signs for consistency.
    """
    # Align based on the largest absolute value element
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        max_idx = np.argmax(np.abs(vec))
        if vec[max_idx] < 0:
            eigenvectors[:, i] = -vec
    return eigenvectors


def similarity(x1, x2):
    """Compute cosine similarity matrix."""
    return F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2)


def get_laplacian_matrix(adj_matrix):
    """Convert adjacency matrix to normalized Laplacian."""
    # Compute degree matrix
    degree_vector = adj_matrix.sum(dim=1)
    degree_matrix = torch.diag(degree_vector)
    
    # Compute Laplacian
    laplacian = degree_matrix - adj_matrix
    
    # Normalized Laplacian: D^{-1/2} L D^{-1/2}
    deg_inv_sqrt = torch.diag(torch.pow(degree_vector + 1e-8, -0.5))
    normalized_laplacian = deg_inv_sqrt @ laplacian @ deg_inv_sqrt
    
    return normalized_laplacian


def build_condition_pe(data, condition_type, num_dim=32, **cfg):
    """
    Build condition positional embedding using GraphControl methods.
    
    Args:
        data: PyTorch Geometric data object
        condition_type: 'role', 'inverse', or 'ppr'
        num_dim: Number of dimensions for PE
        **cfg: Additional configuration parameters
    
    Returns:
        P_cond: Condition positional embedding (N x num_dim)
    """
    device = data.x.device
    
    if condition_type == 'role':
        P_cond = obtain_role_attributes(data, num_dim=num_dim, **cfg)
    elif condition_type == 'inverse':
        P_cond = obtain_inverse_attributes(data, num_dim=num_dim, **cfg)
    elif condition_type == 'ppr':
        P_cond = obtain_disassortative_ppr_attributes(data, num_dim=num_dim, **cfg)
    else:
        raise ValueError(f"Unknown condition_type: {condition_type}")
    
    return P_cond.to(device)


def compute_degree_sequence(edge_index, k_hop=3):
    """Compute k-hop degree sequence for structural role embedding."""
    num_nodes = edge_index.max().item() + 1
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    degree_sequences = []
    current_adj = adj
    
    for k in range(k_hop):
        degrees = current_adj.sum(dim=1)
        degree_sequences.append(degrees.unsqueeze(1))
        if k < k_hop - 1:
            current_adj = torch.matmul(current_adj, adj)
            current_adj = torch.where(current_adj > 0, 1.0, 0.0)
    
    return torch.cat(degree_sequences, dim=1)


def obtain_role_attributes(data, method='struc2vec', k_hop=3, num_dim=32, percentile=0.85):
    """
    Generate role-based positional embeddings for heterophilic graphs.
    Based on GraphControl implementation.
    """
    save_node_border = 30000
    
    if method == 'struc2vec' or method == 'degree':
        # Compute structural signatures based on degree sequences
        role_emb = compute_degree_sequence(data.edge_index, k_hop)
        # Normalize role embeddings
        role_emb = F.normalize(role_emb, p=2, dim=1)
    else:
        raise ValueError(f"Unknown role method: {method}")
    
    num_nodes = role_emb.shape[0]
    
    # Memory-efficient approach for large graphs
    if num_nodes > 3000:
        # Use sampling-based approach for large graphs
        sample_size = min(1000000, num_nodes * 100)  # Sample for threshold estimation
        idx1 = torch.randint(0, num_nodes, (sample_size,))
        idx2 = torch.randint(0, num_nodes, (sample_size,))
        sample_sim = F.cosine_similarity(role_emb[idx1], role_emb[idx2], dim=1)
        q_threshold = torch.quantile(sample_sim, q=percentile)
        
        # Create sparse adjacency matrix
        A_role = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        batch_size = 100
        for i in range(0, num_nodes, batch_size):
            end_i = min(i + batch_size, num_nodes)
            batch_sim = F.cosine_similarity(role_emb[i:end_i].unsqueeze(1), 
                                           role_emb.unsqueeze(0), dim=2)
            A_role[i:end_i] = torch.where(batch_sim > q_threshold, 1.0, 0.0)
    else:
        # Original method for smaller graphs
        role_sim = F.cosine_similarity(role_emb.unsqueeze(1), 
                                       role_emb.unsqueeze(0), dim=2)
        q_threshold = torch.quantile(role_sim.flatten(), q=percentile)
        A_role = torch.where(role_sim > q_threshold, 1.0, 0.0)
    
    # Eigendecomposition for positional embedding
    Lap = get_laplacian_matrix(A_role)
    
    if Lap.shape[0] > save_node_border:
        L, V = scipy.linalg.eigh(Lap.cpu().numpy())
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(Lap)
    
    x = V[:, :num_dim].float()
    x = normalize_l2(x.cpu())
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    
    return x


def obtain_inverse_attributes(data, percentile=0.15, num_dim=32, combine_with_role=False):
    """
    Generate inverse similarity conditions for heterophilic graphs.
    Based on GraphControl implementation.
    """
    save_node_border = 30000
    n_nodes = data.x.shape[0]
    
    # Compute feature similarity
    feat_sim = similarity(data.x, data.x)
    
    # Memory-efficient quantile computation for large graphs
    if n_nodes > 3000:  # Use sampling for large graphs
        # Sample up to 10M elements (instead of all n² elements)
        max_samples = min(10_000_000, feat_sim.numel())
        
        # Random sampling for approximate quantile
        indices = torch.randperm(feat_sim.numel(), device=feat_sim.device)[:max_samples]
        sampled_values = feat_sim.flatten()[indices]
        threshold = torch.quantile(sampled_values, q=percentile)
    else:
        # Small graph - use original method
        threshold = torch.quantile(feat_sim.flatten(), q=percentile)
    
    # Connect dissimilar nodes (bottom percentile)
    A_inverse = torch.where(feat_sim < threshold, 1.0, 0.0)
    
    # Optional: Combine with role similarity for noise reduction
    if combine_with_role:
        role_emb = compute_degree_sequence(data.edge_index, k_hop=2)
        role_emb = F.normalize(role_emb, p=2, dim=1)
        role_sim = F.cosine_similarity(role_emb.unsqueeze(1), 
                                       role_emb.unsqueeze(0), dim=2)
        # Mask with role similarity to reduce noise
        A_inverse = A_inverse * (role_sim > 0.5).float()
    
    # Eigendecomposition for positional embedding
    Lap = get_laplacian_matrix(A_inverse)
    
    if Lap.shape[0] > save_node_border:
        L, V = scipy.linalg.eigh(Lap.cpu().numpy())
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(Lap)
    
    x = V[:, :num_dim].float()
    x = normalize_l2(x.cpu())
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    
    return x


def compute_ppr(edge_index, alpha=0.15, num_nodes=None):
    """Compute Personalized PageRank matrix."""
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    # Create adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    
    # Random walk matrix
    rw_matrix = adj * deg_inv.unsqueeze(1)
    
    # PPR: (I - (1-alpha)*P)^{-1} * alpha
    I = torch.eye(num_nodes, device=edge_index.device)
    ppr_matrix = torch.inverse(I - (1 - alpha) * rw_matrix) * alpha
    
    return ppr_matrix


def obtain_disassortative_ppr_attributes(data, alpha=0.15, num_dim=32, percentile=0.85):
    """
    Generate PPR-based disassortative conditions.
    Based on GraphControl implementation.
    """
    save_node_border = 30000
    num_nodes = data.x.shape[0]
    
    # Compute PPR matrix
    ppr_matrix = compute_ppr(data.edge_index, alpha, num_nodes)
    
    # Compute feature distance
    feat_dist = torch.cdist(data.x, data.x, p=2)
    
    # Normalize feature distance
    feat_dist = (feat_dist - feat_dist.min()) / (feat_dist.max() - feat_dist.min() + 1e-8)
    
    # Weight PPR by feature distance (disassortative)
    A_disassort = ppr_matrix * feat_dist
    
    # Memory-efficient quantile computation for large graphs
    if num_nodes > 3000:  # Use sampling for large graphs
        # Sample up to 10M elements
        max_samples = min(10_000_000, A_disassort.numel())
        
        # Random sampling for approximate quantile
        indices = torch.randperm(A_disassort.numel(), device=A_disassort.device)[:max_samples]
        sampled_values = A_disassort.flatten()[indices]
        threshold = torch.quantile(sampled_values, q=percentile)
    else:
        # Small graph - use original method
        threshold = torch.quantile(A_disassort.flatten(), q=percentile)
    
    # Threshold to create adjacency
    A_disassort = torch.where(A_disassort > threshold, 1.0, 0.0)
    
    # Eigendecomposition for positional embedding
    Lap = get_laplacian_matrix(A_disassort)
    
    if Lap.shape[0] > save_node_border:
        L, V = scipy.linalg.eigh(Lap.cpu().numpy())
        L = torch.from_numpy(L)
        V = torch.from_numpy(V)
    else:
        L, V = torch.linalg.eigh(Lap)
    
    x = V[:, :num_dim].float()
    x = normalize_l2(x.cpu())
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    
    return x


def select_condition_type(data, auto_detect=True, condition_type=None, heterophily_threshold=0.4):
    """
    Adaptively select condition type based on heterophily ratio.
    Based on GraphControl implementation.
    """
    CONDITION_FUNCTIONS = {
        'homophilic': 'obtain_attributes',
        'role': 'obtain_role_attributes', 
        'inverse': 'obtain_inverse_attributes',
        'ppr': 'obtain_disassortative_ppr_attributes'
    }
    
    if not auto_detect and condition_type:
        return CONDITION_FUNCTIONS.get(condition_type, 'obtain_attributes')
    
    # Measure heterophily ratio
    if hasattr(data, 'y') and data.y is not None:
        h_ratio = homophily(data.edge_index, data.y)
        
        if h_ratio > 0.7:  # Homophilic
            return 'obtain_attributes'
        elif h_ratio > heterophily_threshold:  # Mixed
            return 'obtain_role_attributes'
        else:  # Strongly heterophilic
            return 'obtain_inverse_attributes'
    else:
        # Default to role-based when no labels available
        return 'obtain_role_attributes'