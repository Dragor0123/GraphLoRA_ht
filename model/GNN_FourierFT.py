import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from .FourierFT_adapter import FourierFTAdapter, GATConv_FourierFT, TransformerConv_FourierFT


class GNNFourierFT(nn.Module):
    """
    GNN with FourierFT adapters for parameter-efficient fine-tuning.
    
    Args:
        gnn: Pretrained GNN model (frozen)
        gnn_type: Type of GNN ('GAT', 'GCN', 'TransformerConv')
        gnn_layer_num: Number of GNN layers
        d_in: Input dimension
        d_out: Output dimension  
        n: Number of spectral coefficients for FourierFT
        alpha: Scaling factor for FourierFT
    """
    
    def __init__(self, gnn, gnn_type='GAT', gnn_layer_num=2, d_in=512, d_out=256, n=1000, alpha=1.0):
        super().__init__()
        
        # Freeze the pretrained GNN
        self.gnn = gnn
        for param in self.gnn.parameters():
            param.requires_grad = False
            
        self.gnn_type = gnn_type
        self.gnn_layer_num = gnn_layer_num
        self.n = n
        self.alpha = alpha
        
        # Create FourierFT adapters for each layer
        self.fourier_adapters = nn.ModuleList()
        
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            # Single layer case
            if gnn_type == 'GAT':
                adapter = GATConv_FourierFT(
                    self.gnn.conv[0], d_in, d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter = TransformerConv_FourierFT(
                    self.gnn.conv[0], d_in, d_out, n, alpha
                )
            else:  # GCN or other types
                adapter = FourierFTAdapter(d_in, d_out, n, alpha, self.gnn.conv[0])
            self.fourier_adapters.append(adapter)
            
        elif gnn_layer_num == 2:
            # Two layer case
            # First layer: d_in -> 2*d_out
            if gnn_type == 'GAT':
                adapter1 = GATConv_FourierFT(
                    self.gnn.conv[0], d_in, 2*d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter1 = TransformerConv_FourierFT(
                    self.gnn.conv[0], d_in, 2*d_out, n, alpha
                )
            else:
                adapter1 = FourierFTAdapter(d_in, 2*d_out, n, alpha, self.gnn.conv[0])
            self.fourier_adapters.append(adapter1)
            
            # Second layer: 2*d_out -> d_out  
            if gnn_type == 'GAT':
                adapter2 = GATConv_FourierFT(
                    self.gnn.conv[1], 2*d_out, d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter2 = TransformerConv_FourierFT(
                    self.gnn.conv[1], 2*d_out, d_out, n, alpha
                )
            else:
                adapter2 = FourierFTAdapter(2*d_out, d_out, n, alpha, self.gnn.conv[1])
            self.fourier_adapters.append(adapter2)
            
        else:
            # Multi-layer case
            # First layer: d_in -> 2*d_out
            if gnn_type == 'GAT':
                adapter = GATConv_FourierFT(
                    self.gnn.conv[0], d_in, 2*d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter = TransformerConv_FourierFT(
                    self.gnn.conv[0], d_in, 2*d_out, n, alpha
                )
            else:
                adapter = FourierFTAdapter(d_in, 2*d_out, n, alpha, self.gnn.conv[0])
            self.fourier_adapters.append(adapter)
            
            # Middle layers: 2*d_out -> 2*d_out
            for i in range(1, gnn_layer_num - 1):
                if gnn_type == 'GAT':
                    adapter = GATConv_FourierFT(
                        self.gnn.conv[i], 2*d_out, 2*d_out, n, alpha
                    )
                elif gnn_type == 'TransformerConv':
                    adapter = TransformerConv_FourierFT(
                        self.gnn.conv[i], 2*d_out, 2*d_out, n, alpha
                    )
                else:
                    adapter = FourierFTAdapter(2*d_out, 2*d_out, n, alpha, self.gnn.conv[i])
                self.fourier_adapters.append(adapter)
                
            # Last layer: 2*d_out -> d_out
            if gnn_type == 'GAT':
                adapter = GATConv_FourierFT(
                    self.gnn.conv[-1], 2*d_out, d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter = TransformerConv_FourierFT(
                    self.gnn.conv[-1], 2*d_out, d_out, n, alpha
                )
            else:
                adapter = FourierFTAdapter(2*d_out, d_out, n, alpha, self.gnn.conv[-1])
            self.fourier_adapters.append(adapter)

    def forward(self, x, edge_index):
        """
        Forward pass with base GNN + FourierFT adapters.
        
        Returns:
            Tuple of (total_embedding, base_embedding, fourier_embedding)
        """
        # Base GNN forward pass (frozen)
        x_base = x
        for i in range(self.gnn_layer_num - 1):
            x_base = self.gnn.conv[i](x_base, edge_index)
            x_base = self.gnn.activation(x_base)
        emb_base = self.gnn.conv[-1](x_base, edge_index)
        
        # FourierFT adapter forward pass - parallel path
        x_fourier = x
        
        for i in range(self.gnn_layer_num - 1):
            # Apply FourierFT transformation
            F = torch.zeros(x_fourier.shape[1], x_fourier.shape[1] if i == 0 else 2*self.d_out, 
                           dtype=torch.complex64, device=x.device)
            if i == 0:
                target_dim = 2 * self.d_out
            else:
                target_dim = 2 * self.d_out
                
            # Resize F if necessary
            if i == 0:
                F = torch.zeros(x_fourier.shape[1], target_dim, dtype=torch.complex64, device=x.device)
            else:
                F = torch.zeros(x_fourier.shape[1], target_dim, dtype=torch.complex64, device=x.device)
                
            # Use adapter coefficients
            adapter = self.fourier_adapters[i]
            if hasattr(adapter, 'fourier_adapter'):
                E = adapter.fourier_adapter.E
                c = adapter.fourier_adapter.c
                alpha = adapter.fourier_adapter.alpha
            else:
                E = adapter.E
                c = adapter.c
                alpha = adapter.alpha
                
            # Ensure E indices are within bounds
            valid_mask = (E[0] < F.shape[0]) & (E[1] < F.shape[1])
            E_valid = E[:, valid_mask]
            c_valid = c[valid_mask]
            
            F[E_valid[0], E_valid[1]] = c_valid.to(torch.complex64)
            Delta_W = torch.fft.ifft2(F).real * alpha
            x_fourier = torch.matmul(x_fourier, Delta_W)
            x_fourier = self.gnn.activation(x_fourier)
        
        # Final layer
        final_adapter = self.fourier_adapters[-1]
        F_final = torch.zeros(x_fourier.shape[1], self.d_out, dtype=torch.complex64, device=x.device)
        
        if hasattr(final_adapter, 'fourier_adapter'):
            E = final_adapter.fourier_adapter.E
            c = final_adapter.fourier_adapter.c
            alpha = final_adapter.fourier_adapter.alpha
        else:
            E = final_adapter.E
            c = final_adapter.c
            alpha = final_adapter.alpha
            
        # Ensure E indices are within bounds
        valid_mask = (E[0] < F_final.shape[0]) & (E[1] < F_final.shape[1])
        E_valid = E[:, valid_mask]
        c_valid = c[valid_mask]
        
        F_final[E_valid[0], E_valid[1]] = c_valid.to(torch.complex64)
        Delta_W_final = torch.fft.ifft2(F_final).real * alpha
        emb_fourier = torch.matmul(x_fourier, Delta_W_final)
        
        # Combined output
        emb_total = emb_base + emb_fourier
        
        return emb_total, emb_base, emb_fourier