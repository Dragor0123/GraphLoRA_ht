import torch
import numpy as np
from torch_geometric.utils import degree, to_dense_adj
from typing import Optional, Tuple
import random


class RobustHomophilyDetector:
    """
    Lightweight homophily detection with edge density correction.
    Complexity: O(k × samples) << O(N³)
    
    This detector uses random walk sampling to efficiently estimate homophily
    without computing the full graph statistics, making it scalable to large graphs.
    """
    
    def __init__(self, k_hops: int = 3, n_samples: int = 100):
        """
        Initialize the homophily detector.
        
        Args:
            k_hops: Number of hops to consider in random walks
            n_samples: Number of random walk samples to take
        """
        self.k_hops = k_hops
        self.n_samples = n_samples
    
    def detect(self, graph, labels: Optional[torch.Tensor] = None) -> float:
        """
        Detect homophily ratio with corrections for sparse graphs.
        
        Args:
            graph: PyTorch Geometric Data object with edge_index
            labels: Node labels tensor. If None, uses graph.y
        
        Returns:
            Homophily ratio in [0, 1]
        """
        if labels is None:
            if hasattr(graph, 'y'):
                labels = graph.y
            else:
                raise ValueError("Labels must be provided either directly or in graph.y")
        
        # Handle different graph input types
        if hasattr(graph, 'edge_index'):
            edge_index = graph.edge_index
            num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else labels.size(0)
        else:
            raise ValueError("Graph must have edge_index attribute")
        
        # Check if graph is too small for sampling
        if num_nodes < 10:
            # For very small graphs, compute exact homophily
            return self._exact_homophily(edge_index, labels, num_nodes)
        
        # Sample-based homophily estimation
        h_sample = self.sample_based_homophily(edge_index, labels, self.k_hops, self.n_samples)
        
        # Apply edge density correction for very sparse graphs
        edge_density = edge_index.size(1) / (num_nodes * (num_nodes - 1))
        if edge_density < 0.01:
            h_corrected = self.edge_density_correction(h_sample, edge_density)
        else:
            h_corrected = h_sample
        
        return h_corrected
    
    def sample_based_homophily(self, edge_index: torch.Tensor, labels: torch.Tensor, 
                               k: int, samples: int) -> float:
        """
        Core sampling logic using random walks.
        
        Args:
            edge_index: Edge index tensor
            labels: Node labels
            k: Number of hops in random walk
            samples: Number of samples to take
        
        Returns:
            Estimated homophily ratio
        """
        num_nodes = labels.size(0)
        device = edge_index.device
        
        # Build adjacency list for efficient neighbor sampling
        adj_list = self._build_adj_list(edge_index, num_nodes)
        
        homophily_scores = []
        
        for _ in range(samples):
            # Random starting node
            start_node = random.randint(0, num_nodes - 1)
            
            # Perform k-hop random walk
            current_node = start_node
            same_label_count = 0
            total_count = 0
            
            for hop in range(k):
                neighbors = adj_list.get(current_node, [])
                if len(neighbors) == 0:
                    break
                
                # Random neighbor selection
                next_node = random.choice(neighbors)
                
                # Check label agreement
                if labels[current_node] == labels[next_node]:
                    same_label_count += 1
                total_count += 1
                
                current_node = next_node
            
            if total_count > 0:
                homophily_scores.append(same_label_count / total_count)
        
        if len(homophily_scores) == 0:
            return 0.5  # Default to neutral if no valid walks
        
        return np.mean(homophily_scores)
    
    def edge_density_correction(self, h_sample: float, edge_density: float) -> float:
        """
        Correct for sparse graphs where edge_density < 0.01.
        
        Very sparse graphs can have artificially high homophily due to
        disconnected components. This correction adjusts for that bias.
        
        Args:
            h_sample: Sample-based homophily estimate
            edge_density: Graph edge density
        
        Returns:
            Corrected homophily ratio
        """
        # Apply logarithmic correction for very sparse graphs
        # As density approaches 0, correction factor approaches 0.5 (neutral)
        correction_factor = 0.5 + 0.5 * np.log10(max(edge_density, 1e-4) / 0.01)
        correction_factor = np.clip(correction_factor, 0.5, 1.0)
        
        # Blend original estimate with neutral value based on sparsity
        h_corrected = h_sample * correction_factor + 0.5 * (1 - correction_factor)
        
        return h_corrected
    
    def _build_adj_list(self, edge_index: torch.Tensor, num_nodes: int) -> dict:
        """
        Build adjacency list from edge index for efficient neighbor access.
        
        Args:
            edge_index: Edge index tensor
            num_nodes: Number of nodes in graph
        
        Returns:
            Dictionary mapping node index to list of neighbors
        """
        # Convert to CPU for processing if needed
        edge_index_cpu = edge_index.cpu().numpy()
        
        # Ensure adjacency list covers all nodes that appear in edges
        max_node_idx = edge_index_cpu.max() if edge_index_cpu.size > 0 else num_nodes - 1
        actual_num_nodes = max(max_node_idx + 1, num_nodes)
        adj_list = {i: [] for i in range(actual_num_nodes)}
        
        for i in range(edge_index_cpu.shape[1]):
            src = edge_index_cpu[0, i]
            dst = edge_index_cpu[1, i]
            adj_list[src].append(dst)
        
        return adj_list
    
    def _exact_homophily(self, edge_index: torch.Tensor, labels: torch.Tensor, 
                        num_nodes: int) -> float:
        """
        Compute exact homophily for small graphs.
        
        Args:
            edge_index: Edge index tensor
            labels: Node labels
            num_nodes: Number of nodes
        
        Returns:
            Exact homophily ratio
        """
        if edge_index.size(1) == 0:
            return 0.5  # No edges, return neutral
        
        # Count edges between same-label nodes
        same_label_edges = 0
        total_edges = edge_index.size(1)
        
        for i in range(total_edges):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            if labels[src] == labels[dst]:
                same_label_edges += 1
        
        return same_label_edges / total_edges
    
    def get_homophily_level(self, h_ratio: float) -> str:
        """
        Categorize homophily level based on ratio.
        
        Args:
            h_ratio: Homophily ratio
        
        Returns:
            String describing homophily level
        """
        if h_ratio > 0.6:
            return "homophilic"
        elif h_ratio < 0.4:
            return "heterophilic"
        else:
            return "mixed"