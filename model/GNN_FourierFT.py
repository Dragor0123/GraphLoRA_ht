import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from .FourierFT_adapter import FourierFTAdapter, GATConv_FourierFT, TransformerConv_FourierFT


def infer_layer_dimensions(layer):
    """
    Infer input and output dimensions from a GNN layer.
    
    Args:
        layer: GNN layer (GATConv, TransformerConv, GCNConv, etc.)
        
    Returns:
        tuple: (input_dim, output_dim)
    """
    if hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
        # Handle tuple input channels (for heterogeneous graphs)
        in_channels = layer.in_channels
        if isinstance(in_channels, tuple):
            in_channels = in_channels[0]  # Use source node channels
            
        out_channels = layer.out_channels
        
        # For GAT with heads, adjust output dimension
        if hasattr(layer, 'heads') and hasattr(layer, 'concat'):
            if layer.concat:
                out_channels = out_channels * layer.heads
            # If not concat, out_channels remains the same (average pooling)
                
        return in_channels, out_channels
    
    elif hasattr(layer, 'weight'):
        # Fallback: infer from weight matrix shape
        weight_shape = layer.weight.shape
        return weight_shape[1], weight_shape[0]  # (in_features, out_features)
    
    else:
        raise ValueError(f"Cannot infer dimensions from layer type: {type(layer)}")


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
            # Single layer case - infer dimensions from base layer
            layer_d_in, layer_d_out = infer_layer_dimensions(self.gnn.conv[0])
            
            if gnn_type == 'GAT':
                adapter = GATConv_FourierFT(
                    self.gnn.conv[0], layer_d_in, layer_d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter = TransformerConv_FourierFT(
                    self.gnn.conv[0], layer_d_in, layer_d_out, n, alpha
                )
            else:  # GCN or other types
                adapter = FourierFTAdapter(layer_d_in, layer_d_out, n, alpha, self.gnn.conv[0])
            self.fourier_adapters.append(adapter)
            
        elif gnn_layer_num == 2:
            # Two layer case - infer dimensions from base layers
            layer1_d_in, layer1_d_out = infer_layer_dimensions(self.gnn.conv[0])
            layer2_d_in, layer2_d_out = infer_layer_dimensions(self.gnn.conv[1])
            
            # First layer
            if gnn_type == 'GAT':
                adapter1 = GATConv_FourierFT(
                    self.gnn.conv[0], layer1_d_in, layer1_d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter1 = TransformerConv_FourierFT(
                    self.gnn.conv[0], layer1_d_in, layer1_d_out, n, alpha
                )
            else:
                adapter1 = FourierFTAdapter(layer1_d_in, layer1_d_out, n, alpha, self.gnn.conv[0])
            self.fourier_adapters.append(adapter1)
            
            # Second layer
            if gnn_type == 'GAT':
                adapter2 = GATConv_FourierFT(
                    self.gnn.conv[1], layer2_d_in, layer2_d_out, n, alpha
                )
            elif gnn_type == 'TransformerConv':
                adapter2 = TransformerConv_FourierFT(
                    self.gnn.conv[1], layer2_d_in, layer2_d_out, n, alpha
                )
            else:
                adapter2 = FourierFTAdapter(layer2_d_in, layer2_d_out, n, alpha, self.gnn.conv[1])
            self.fourier_adapters.append(adapter2)
            
        else:
            # Multi-layer case - infer dimensions from each base layer
            for i in range(gnn_layer_num):
                layer_d_in, layer_d_out = infer_layer_dimensions(self.gnn.conv[i])
                
                if gnn_type == 'GAT':
                    adapter = GATConv_FourierFT(
                        self.gnn.conv[i], layer_d_in, layer_d_out, n, alpha
                    )
                elif gnn_type == 'TransformerConv':
                    adapter = TransformerConv_FourierFT(
                        self.gnn.conv[i], layer_d_in, layer_d_out, n, alpha
                    )
                else:
                    adapter = FourierFTAdapter(layer_d_in, layer_d_out, n, alpha, self.gnn.conv[i])
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
            # Use adapter's cached Delta_W
            adapter = self.fourier_adapters[i]
            if hasattr(adapter, 'fourier_adapter'):
                Delta_W = adapter.fourier_adapter._compute_delta_w(x.device)
            elif hasattr(adapter, '_compute_delta_w'):
                Delta_W = adapter._compute_delta_w(x.device)
            else:
                # Fallback for direct adapters
                Delta_W = adapter._compute_delta_w(x.device)
                
            x_fourier = torch.matmul(x_fourier, Delta_W)
            x_fourier = self.gnn.activation(x_fourier)
        
        # Final layer using cached Delta_W
        final_adapter = self.fourier_adapters[-1]
        if hasattr(final_adapter, 'fourier_adapter'):
            Delta_W_final = final_adapter.fourier_adapter._compute_delta_w(x.device)
        elif hasattr(final_adapter, '_compute_delta_w'):
            Delta_W_final = final_adapter._compute_delta_w(x.device)
        else:
            # Fallback for direct adapters
            Delta_W_final = final_adapter._compute_delta_w(x.device)
            
        emb_fourier = torch.matmul(x_fourier, Delta_W_final)
        
        # Combined output
        emb_total = emb_base + emb_fourier
        
        return emb_total, emb_base, emb_fourier