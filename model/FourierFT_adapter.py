import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FourierFTAdapter(nn.Module):
    """
    FourierFT Adapter for parameter-efficient fine-tuning using Fourier Transform.
    
    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension  
        n (int): Number of trainable spectral coefficients
        alpha (float): Scaling factor
        base_layer (nn.Module, optional): Base layer to wrap
    """
    
    def __init__(self, d_in: int, d_out: int, n: int = 1000, alpha: float = 1.0, base_layer: Optional[nn.Module] = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n = n
        self.alpha = alpha
        self.base_layer = base_layer
        
        # Randomly select spectral entries (frozen across all layers)
        # Store as (2, n) tensor where first row is i-indices, second row is j-indices
        total_entries = d_in * d_out
        selected_indices = torch.randperm(total_entries)[:n]
        self.register_buffer('E', torch.stack([
            selected_indices // d_out,  # i-indices
            selected_indices % d_out     # j-indices
        ]))
        
        # Trainable spectral coefficients 
        self.c = nn.Parameter(torch.randn(n), requires_grad=True)
        
        # Initialize coefficients with small values
        nn.init.normal_(self.c, mean=0.0, std=0.01)
        
        # Cache for Delta_W computation
        self._delta_w_cache = None
        self._dirty = True
        
        # Register hook to invalidate cache when coefficients change
        self.c.register_hook(lambda grad: setattr(self, '_dirty', True))
    
    def _compute_delta_w(self, device):
        """
        Compute and cache Delta_W matrix.
        Only recomputes when coefficients have changed.
        """
        if self._dirty or self._delta_w_cache is None:
            # Create dense spectral matrix F
            F = torch.zeros(self.d_in, self.d_out, dtype=torch.complex64, device=device)
            F[self.E[0], self.E[1]] = self.c.to(torch.complex64)
            
            # Compute Delta_W via inverse DFT and cache it
            self._delta_w_cache = torch.fft.ifft2(F).real * self.alpha
            self._dirty = False
        
        return self._delta_w_cache
    
    def forward(self, x, edge_index):
        """
        Forward pass through FourierFT adapter.
        
        Args:
            x: Input node features
            edge_index: Graph edge indices
            
        Returns:
            Updated node embeddings
        """
        # Get cached Delta_W (recomputes only if coefficients changed)
        Delta_W = self._compute_delta_w(x.device)
        
        if self.base_layer is not None:
            # If base layer exists, apply it and add FourierFT adaptation
            h_base = self.base_layer(x, edge_index)
            x_transformed = torch.matmul(x, Delta_W)
            return h_base + x_transformed
        else:
            # If no base layer, just apply FourierFT transformation
            return torch.matmul(x, Delta_W)


class GATConv_FourierFT(nn.Module):
    """
    GAT layer with FourierFT adapter integration.
    """
    
    def __init__(self, base_gat, d_in: int, d_out: int, n: int = 1000, alpha: float = 1.0):
        super().__init__()
        self.base_gat = base_gat
        self.fourier_adapter = FourierFTAdapter(d_in, d_out, n, alpha)
        
    def forward(self, x, edge_index):
        # Base GAT forward pass
        base_output = self.base_gat(x, edge_index)
        
        # FourierFT adaptation using cached Delta_W
        Delta_W = self.fourier_adapter._compute_delta_w(x.device)
        fourier_delta = torch.matmul(x, Delta_W)
        
        return base_output + fourier_delta


class TransformerConv_FourierFT(nn.Module):
    """
    TransformerConv layer with FourierFT adapter integration.
    """
    
    def __init__(self, base_transformer, d_in: int, d_out: int, n: int = 1000, alpha: float = 1.0):
        super().__init__()
        self.base_transformer = base_transformer
        self.fourier_adapter = FourierFTAdapter(d_in, d_out, n, alpha)
        
    def forward(self, x, edge_index):
        # Base TransformerConv forward pass
        base_output = self.base_transformer(x, edge_index)
        
        # FourierFT adaptation using cached Delta_W
        Delta_W = self.fourier_adapter._compute_delta_w(x.device)
        fourier_delta = torch.matmul(x, Delta_W)
        
        return base_output + fourier_delta