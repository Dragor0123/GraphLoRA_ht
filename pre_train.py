from model.GNN_model import GNN
from model.GRACE_model import GRACE
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from util import get_dataset, act, mkdir
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import scipy.sparse as sp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_laplacian_pe(edge_index, num_nodes, k_eigs=32, normalization='sym'):
    """
    Compute Laplacian positional encoding with sign alignment.
    """
    adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj_scipy = adj_scipy + sp.eye(num_nodes)
    
    degrees = np.array(adj_scipy.sum(axis=1)).flatten()
    
    if normalization == 'sym':
        deg_inv_sqrt = sp.diags(np.power(degrees, -0.5))
        laplacian = sp.eye(num_nodes) - deg_inv_sqrt @ adj_scipy @ deg_inv_sqrt
    else:
        deg_inv = sp.diags(np.power(degrees, -1.0))
        laplacian = sp.eye(num_nodes) - deg_inv @ adj_scipy
    
    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(
            laplacian, k=min(k_eigs, num_nodes-1), which='SM', return_eigenvectors=True
        )
    except:
        laplacian_dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx[:k_eigs]]
        eigenvectors = eigenvectors[:, idx[:k_eigs]]
    
    # Sign alignment
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        max_idx = np.argmax(np.abs(vec))
        if vec[max_idx] < 0:
            eigenvectors[:, i] = -vec
    
    # L2 normalize
    eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-8)
    
    return torch.FloatTensor(eigenvectors)


def sample_rwr_subgraph(edge_index, start_node, walk_length, restart_prob, num_nodes, adj_list=None):
    """
    Optimized subgraph sampling using Random Walk with Restart.
    """
    # Use pre-built adjacency list if provided
    if adj_list is None:
        adj_list = {}
        for src, dst in edge_index.t():
            src_item, dst_item = src.item(), dst.item()
            if src_item not in adj_list:
                adj_list[src_item] = []
            adj_list[src_item].append(dst_item)
    
    visited = set()
    current_node = start_node
    
    # Pre-generate random numbers for better performance
    rand_vals = torch.rand(walk_length)
    
    for i in range(walk_length):
        visited.add(current_node)
        
        if current_node not in adj_list or len(adj_list[current_node]) == 0 or rand_vals[i].item() < restart_prob:
            current_node = start_node
        else:
            neighbors = adj_list[current_node] 
            current_node = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
        
        # Early stopping if subgraph is large enough
        if len(visited) > walk_length // 4:  # Stop if we have enough nodes
            break
    
    return torch.tensor(list(visited), dtype=torch.long)


class StructuralContrastiveModel(nn.Module):
    """
    Structural contrastive model for feature-free pretraining.
    """
    def __init__(self, gnn, num_hidden, num_proj_hidden, tau=0.5):
        super(StructuralContrastiveModel, self).__init__()
        self.gnn = gnn
        self.tau = tau
        
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)
    
    def forward(self, pe, edge_index):
        """Forward pass with positional encoding as input."""
        return self.gnn(pe, edge_index)
    
    def projection(self, z):
        """Projection head for contrastive learning."""
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def contrastive_loss(self, z1, z2, batch_z1=None, batch_z2=None):
        """
        Enhanced InfoNCE contrastive loss with better GPU utilization.
        Supports both single-pair and batch contrastive learning.
        """
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        
        if batch_z1 is not None and batch_z2 is not None:
            # Batch contrastive learning for better GPU utilization
            batch_h1 = self.projection(batch_z1)  # (batch_size, hidden_dim)
            batch_h2 = self.projection(batch_z2)  # (batch_size, hidden_dim)
            
            # Pool each subgraph to graph-level
            h1 = h1.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            h2 = h2.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            batch_h1_pooled = batch_h1.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
            batch_h2_pooled = batch_h2.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
            
            # Normalize
            h1_norm = F.normalize(h1, dim=1)
            h2_norm = F.normalize(h2, dim=1)
            batch_h1_norm = F.normalize(batch_h1_pooled.squeeze(1), dim=1)
            batch_h2_norm = F.normalize(batch_h2_pooled.squeeze(1), dim=1)
            
            # Positive similarity
            pos_sim = torch.exp(torch.sum(h1_norm * h2_norm, dim=1) / self.tau)
            
            # Negative similarities (against batch)
            neg_sim1 = torch.exp(torch.mm(h1_norm, batch_h1_norm.t()) / self.tau).sum(dim=1)
            neg_sim2 = torch.exp(torch.mm(h1_norm, batch_h2_norm.t()) / self.tau).sum(dim=1)
            
            loss = -torch.log(pos_sim / (neg_sim1 + neg_sim2 + pos_sim)).mean()
        else:
            # Simple pair-wise contrastive loss
            h1 = h1.mean(dim=0, keepdim=True)
            h2 = h2.mean(dim=0, keepdim=True)
            
            h1_norm = F.normalize(h1, dim=1)
            h2_norm = F.normalize(h2, dim=1)
            
            pos_sim = torch.exp(torch.sum(h1_norm * h2_norm, dim=1) / self.tau)
            neg_sim = torch.exp(torch.mm(h1_norm, h2_norm.t()) / self.tau).sum(dim=1)
            
            loss = -torch.log(pos_sim / neg_sim).mean()
        
        return loss


def pretrain_structural(dataname, config, gpu, is_reduction=False, seed=42):
    """
    Structural pretraining using Laplacian PE and subgraph contrastive learning.
    Feature-free pretraining (GraphControl style).
    
    Returns:
        str: Path to the saved structural encoder checkpoint
    """
    set_seed(seed)
    print(f"Starting structural pretraining for {dataname}")
    
    path = os.path.join('./datasets', dataname)
    dataset = get_dataset(path, dataname)
    data = dataset[0]
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Get struct config if available, otherwise use default config 
    struct_config = config.get('struct', config)
    
    # Compute Laplacian PE
    k_eigs = struct_config.get('k_eigs', 32)
    pe = compute_laplacian_pe(data.edge_index, data.x.shape[0], k_eigs)
    pe = pe.to(device)
    
    pre_trained_model_path = './pre_trained_gnn/'
    mkdir(pre_trained_model_path)
    
    # Model parameters with synonym mapping for config compatibility
    input_dim = k_eigs  # Use PE dimension as input
    output_dim = struct_config.get('output_dim', struct_config.get('out_dim', 256))
    num_proj_dim = struct_config.get('num_proj_dim', output_dim)
    activation = act(struct_config.get('activation', 'relu'))
    learning_rate = struct_config.get('learning_rate', struct_config.get('lr', 0.001))
    weight_decay = struct_config.get('weight_decay', struct_config.get('wd', 0.0))
    num_epochs = struct_config.get('num_epochs', struct_config.get('epochs', 1000))
    tau = struct_config.get('tau', 0.5)
    gnn_type = struct_config.get('gnn_type', 'GAT')
    num_layers = struct_config.get('num_layers', 2)
    
    # Subgraph sampling parameters
    num_subgraphs = struct_config.get('num_subgraphs', 1024)
    walk_length = struct_config.get('walk_length', 256)
    restart_prob = struct_config.get('restart_prob', 0.8)
    batch_size = struct_config.get('batch_size', 256)
    
    # Create GNN for structural encoder (takes PE as input)
    gnn = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
    
    # Use GRACE model structure for contrastive learning
    pretrain_model = StructuralContrastiveModel(gnn, output_dim, num_proj_dim, tau)
    pretrain_model.to(device)
    
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start = time()
    prev = start
    pretrain_model.train()
    min_loss = float('inf')
    
    model_path = pre_trained_model_path + f"{dataname}.STRUCT.GAT.{is_reduction}.pth"
    
    print(f"Structural pretraining with {num_subgraphs} subgraphs per epoch...")
    
    # Pre-build adjacency list once for efficiency  
    print("Building adjacency list...")
    adj_list = {}
    for src, dst in data.edge_index.t():
        src_item, dst_item = src.item(), dst.item()
        if src_item not in adj_list:
            adj_list[src_item] = []
        adj_list[src_item].append(dst_item)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = (num_subgraphs - 1) // batch_size + 1
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_subgraphs)
            current_batch_size = end_idx - start_idx
            
            # Sample subgraphs using pre-built adjacency list
            subgraph_pairs = []
            for _ in range(current_batch_size):
                # Sample two views of the same subgraph
                center_node = torch.randint(0, data.x.shape[0], (1,)).item()
                sg1 = sample_rwr_subgraph(data.edge_index, center_node, walk_length, restart_prob, data.x.shape[0], adj_list)
                sg2 = sample_rwr_subgraph(data.edge_index, center_node, walk_length, restart_prob, data.x.shape[0], adj_list)
                subgraph_pairs.append((sg1, sg2))
            
            # Compute contrastive loss for this batch with better GPU utilization
            optimizer.zero_grad()
            
            # Collect all subgraph embeddings in batch for efficient GPU computation
            batch_z1_list = []
            batch_z2_list = []
            
            for sg1, sg2 in subgraph_pairs:
                # Ensure subgraph indices are on the same device as data
                sg1 = sg1.to(device)
                sg2 = sg2.to(device)
                
                # Get PE for subgraph nodes
                pe1 = pe[sg1]
                pe2 = pe[sg2]
                
                # Get subgraph edges
                sg_edges1, _ = subgraph(sg1, data.edge_index, relabel_nodes=True, num_nodes=data.x.shape[0])
                sg_edges2, _ = subgraph(sg2, data.edge_index, relabel_nodes=True, num_nodes=data.x.shape[0])
                
                # Forward pass - accumulate embeddings
                z1 = pretrain_model.forward(pe1, sg_edges1)  
                z2 = pretrain_model.forward(pe2, sg_edges2)
                
                batch_z1_list.append(z1)
                batch_z2_list.append(z2)
            
            # Simplified contrastive loss computation for better GPU utilization
            if len(batch_z1_list) >= 2:
                # Use the first pair as anchor, others as negatives
                anchor_z1, anchor_z2 = batch_z1_list[0], batch_z2_list[0]
                
                # Process batch embeddings with proper padding
                if len(batch_z1_list) > 2:
                    # Get all embeddings except the anchor
                    neg_z1_list = batch_z1_list[1:]
                    neg_z2_list = batch_z2_list[1:]
                    
                    # Find maximum size for padding
                    max_nodes = max(max(z.shape[0] for z in neg_z1_list), 
                                   max(z.shape[0] for z in neg_z2_list))
                    
                    padded_z1 = []
                    padded_z2 = []
                    
                    for z1, z2 in zip(neg_z1_list, neg_z2_list):
                        # Ensure both tensors have the same target size
                        target_size = max_nodes
                        
                        # Pad z1
                        if z1.shape[0] < target_size:
                            padding1 = torch.zeros(target_size - z1.shape[0], z1.shape[1], device=device)
                            z1_padded = torch.cat([z1, padding1], dim=0)
                        else:
                            z1_padded = z1[:target_size]
                            
                        # Pad z2
                        if z2.shape[0] < target_size:
                            padding2 = torch.zeros(target_size - z2.shape[0], z2.shape[1], device=device)
                            z2_padded = torch.cat([z2, padding2], dim=0)
                        else:
                            z2_padded = z2[:target_size]
                        
                        padded_z1.append(z1_padded)
                        padded_z2.append(z2_padded)
                    
                    if padded_z1:  # If we have negative samples
                        batch_z1 = torch.stack(padded_z1, dim=0)
                        batch_z2 = torch.stack(padded_z2, dim=0)
                        batch_loss = pretrain_model.contrastive_loss(anchor_z1, anchor_z2, batch_z1, batch_z2)
                    else:
                        batch_loss = pretrain_model.contrastive_loss(anchor_z1, anchor_z2)
                else:
                    # Only two samples, use simple pairwise loss
                    batch_loss = pretrain_model.contrastive_loss(anchor_z1, anchor_z2)
            else:
                # Single pair fallback
                batch_loss = pretrain_model.contrastive_loss(batch_z1_list[0], batch_z2_list[0])
            
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        epoch_loss = epoch_loss / num_batches
        now = time()
        print(f'(STRUCT) Epoch={epoch:03d}, loss={epoch_loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(pretrain_model.gnn.state_dict(), model_path)
            print(f"+++structural model saved! {model_path}")
    
    print("=== Structural Pretraining Complete ===")
    return model_path


def pretrain(dataname, pretext, config, gpu, is_reduction=False, seed=42):
    set_seed(seed)
    
    if pretext == 'STRUCT':
        return pretrain_structural(dataname, config, gpu, is_reduction, seed)
    
    print(os.getcwd())
    path = os.path.join('./datasets', dataname)
    dataset = get_dataset(path, dataname)
    data = dataset[0]
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    pre_trained_model_path = './pre_trained_gnn/'
    mkdir(pre_trained_model_path)
    print("create PreTrain instance...")
    input_dim = data.x.shape[1]
    # Model parameters with synonym mapping for config compatibility
    output_dim = config.get('output_dim', config.get('out_dim', 256))
    num_proj_dim = config.get('num_proj_dim', output_dim)
    activation = act(config.get('activation', 'relu'))
    learning_rate = config.get('learning_rate', config.get('lr', 0.001))
    weight_decay = config.get('weight_decay', config.get('wd', 0.0))
    num_epochs = config.get('num_epochs', config.get('epochs', 1000))
    tau = config.get('tau', 0.5)
    gnn_type = config.get('gnn_type', 'GAT')
    num_layers = config.get('num_layers', 2)
    drop_edge_rate = config.get('drop_edge_rate', 0.4)
    drop_feature_rate = config.get('drop_feature_rate', 0.1)
    gnn = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
    if pretext == 'GRACE':
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    else:
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    pretrain_model.to(device)
    print("pre-training...")
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time()
    prev = start
    pretrain_model.train()
    min_loss = 100000
    # Consistent naming convention: STRUCT for structural, FEAT for feature encoders
    if pretext == 'GRACE':
        model_path = pre_trained_model_path + f"{dataname}.FEAT.GAT.{is_reduction}.pth"
    else:
        model_path = pre_trained_model_path + "{}.{}.{}.{}.pth".format(dataname, pretext, gnn_type, is_reduction)
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = pretrain_model.compute_loss(data.x, data.edge_index)
        loss.backward()
        optimizer.step()
        now = time()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        if min_loss > loss:
            min_loss = loss
            torch.save(pretrain_model.gnn.state_dict(), model_path)
            print("+++model saved ! {}.{}.{}.pth".format(dataname, pretext, gnn_type))
    print("=== Final ===")
    
    # Return model path for consistency
    return model_path