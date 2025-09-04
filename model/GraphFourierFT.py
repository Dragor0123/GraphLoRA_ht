from model.GNN_model import GNN
from model.GNN_FourierFT import GNNFourierFT
from model.ConditionGen import build_condition_pe, ZeroMLP, compute_laplacian_pe
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import (get_dataset, act, SMMDLoss, mkdir, get_ppr_weight, print_trainable_parameters,
                  get_few_shot_mask, batched_smmd_loss, batched_gct_loss, batched_mmd_loss, 
                  load_dual_encoders, print_memory_usage)
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def calculate_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, batch_size=128):
    """
    Calculate SMMD loss term
    
    Args:
        feature_map: Projected features from test dataset
        pretrain_graph_loader: DataLoader for pretrain dataset features
        SMMD: SMMD loss function
        ppr_weight: PPR weight matrix
        batch_size: Batch size for SMMD loss calculation
    
    Returns:
        smmd_loss: Computed SMMD loss value
    """
    return batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, batch_size)


def calculate_mmd_loss(feature_map, pretrain_graph_loader, MMD, batch_size=128):
    return batched_mmd_loss(feature_map, pretrain_graph_loader, MMD, batch_size)


def calculate_reg_loss(logits, target_adj, device):
    """
    Calculate regularization loss term
    
    Args:
        logits: Output logits from the model
        target_adj: Target adjacency matrix
        device: Device to run computations on
    
    Returns:
        loss_reg: Computed regularization loss
    """
    pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    
    reg_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
    loss_reg = F.binary_cross_entropy(reg_adj.view(-1), target_adj.view(-1), weight=weight_tensor)
    
    return loss_reg


def log_output(message, args):
    """Helper function to handle output based on logging preference"""
    if hasattr(args, 'use_logging') and args.use_logging and hasattr(args, 'log_file'):
        with open(args.log_file, 'a') as f:
            f.write(message + '\n')
    else:
        print(message)


def print_memory_usage(model, adapter_module_names=None):
    """
    Print memory usage for trainable parameters and adapters.
    
    Args:
        model: The model to analyze
        adapter_module_names: List of adapter module names to filter
    """
    total_params = 0
    adapter_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            # Check if this is an adapter parameter
            if adapter_module_names:
                for adapter_name in adapter_module_names:
                    if adapter_name in name:
                        adapter_params += param_count
                        break
            elif 'fourier' in name.lower() or 'adapter' in name.lower():
                adapter_params += param_count
    
    # Convert to MB (4 bytes per float32 parameter)
    total_mb = total_params * 4 / (1024 * 1024)
    adapter_mb = adapter_params * 4 / (1024 * 1024)
    
    print(f"Adapter params (FourierFT): {adapter_params//1000}K ({adapter_mb:.2f} MB)")
    print(f"Total trainable params: {total_params/1000000:.1f}M ({total_mb:.1f} MB)")


def transfer_fourier(args, config, gpu_id, is_reduction):
    """
    FourierFT-based transfer learning with Dual Encoders + ControlNet.
    
    Architecture:
        Target Data → [Feature Encoder + FourierFT Adapter] → Embeddings
                                          ↑
                    Condition PE → ZeroMLP → FiLM(γ,β) modulation
                                          ↑
                    Target Data → Structural Encoder (frozen)
    
    Args:
        args: Command line arguments with encoder paths and condition type
        config: Configuration dictionary
        gpu_id: GPU device ID
        is_reduction: Whether to apply feature reduction
    """
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    log_output(f"🚀 Starting FourierFT + Heterophilic GraphControl Transfer", args)

    # Load datasets
    pretrain_datapath = os.path.join('./datasets', args.pretrain_dataset)
    test_datapath = os.path.join('./datasets', args.test_dataset)
    pretrain_dataset = get_dataset(pretrain_datapath, args.pretrain_dataset)[0]
    test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
    
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_dataset = feature_reduce(pretrain_dataset)
        test_dataset = feature_reduce(test_dataset)
    
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset = test_dataset.to(device)

    # Target adjacency for structural regularization
    target_adj = to_dense_adj(test_dataset.edge_index)[0]

    # 🔑 STEP 1: Load Dual Encoders (Structural + Feature)
    k_eigs = 32  # PE dimension for structural encoder
    encoder_S, encoder_F = load_dual_encoders(
        encoder_S_path=args.encoder_S_ckpt,
        encoder_F_path=args.encoder_F_ckpt, 
        device=device,
        input_dim_pe=k_eigs,
        feature_dim=test_dataset.x.shape[1],  # Use target feature dim
        output_dim=config['output_dim'],
        gnn_type=config['gnn_type'],
        num_layers=config['num_layers'],
        activation_fn=act(config['activation'])
    )

    # 🔑 STEP 2: Generate Condition PE using Heterophilic GraphControl
    log_output(f"🎯 Building condition PE with type: {args.condition_type}", args)
    P_cond = build_condition_pe(
        test_dataset, 
        condition_type=args.condition_type,
        num_dim=k_eigs,
        percentile=0.85 if args.condition_type in ['role', 'ppr'] else 0.15,
        k_hop=3,
        alpha=0.15  # for PPR
    )

    # 🔑 STEP 3: Create FourierFT with Adapters on Feature Encoder
    gnn_fourier = GNNFourierFT(
        encoder_F,  # Attach adapters to frozen feature encoder
        config['gnn_type'],
        config['num_layers'],
        d_in=test_dataset.x.shape[1],
        d_out=config['output_dim'],
        n=args.n,  # Number of spectral coefficients
        alpha=args.alpha  # Scaling factor
    )
    gnn_fourier.to(device)
    gnn_fourier.train()

    # 🔑 STEP 4: Create ZeroMLP for ControlNet (FiLM modulation)
    zero_mlp = ZeroMLP(
        in_dim=k_eigs,  # Condition PE dimension
        hidden=128,
        coeff_dim=args.n  # Number of FourierFT coefficients to modulate
    )
    zero_mlp.to(device)
    zero_mlp.train()

    # Loss functions
    SMMD = SMMDLoss().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Optional projector for domain alignment
    projector = Projector(test_dataset.x.shape[1], test_dataset.x.shape[1])  # Identity mapping by default
    projector = projector.to(device)
    projector.train()

    # Classifier
    logreg = LogReg(config['output_dim'], max(test_dataset.y) + 1)
    logreg = logreg.to(device)
    logreg.train()

    # 🔑 STEP 5: Setup train/val/test masks
    if args.test_dataset in ['PubMed', 'CiteSeer', 'Cora']:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            train_mask = test_dataset.train_mask
            val_mask = test_dataset.val_mask
            test_mask = test_dataset.test_mask
    else:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            index = np.arange(test_dataset.x.shape[0])
            np.random.shuffle(index)
            train_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            val_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            test_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            train_mask[index[:int(len(index) * 0.1)]] = True
            val_mask[index[int(len(index) * 0.1):int(len(index) * 0.2)]] = True
            test_mask[index[int(len(index) * 0.2):]] = True
    
    # Contrastive learning mask for consistency loss
    mask = torch.zeros((test_dataset.x.shape[0], test_dataset.x.shape[0])).to(device)
    ppr_weight = get_ppr_weight(test_dataset)
    idx_a = torch.empty(0, dtype=torch.long, device=device)
    idx_b = torch.empty(0, dtype=torch.long, device=device)
    for i in range(max(test_dataset.y) + 1):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        train_label = test_dataset.y[train_idx]
        idx_a = torch.concat((idx_a, train_idx[train_label == i].repeat_interleave(len(train_idx[train_label == i]))))
        idx_b = torch.concat((idx_b, train_idx[train_label == i].repeat(len(train_idx[train_label == i]))))
    mask = torch.sparse_coo_tensor(indices=torch.stack((idx_a, idx_b)), values=torch.ones(len(idx_a)).to(device), size=[test_dataset.x.shape[0], test_dataset.x.shape[0]]).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(test_dataset.x.shape[0]).to(device)
    
    # 🔑 STEP 6: Setup 3-group optimizer (adapter + controlnet + classifier)
    optimizer = torch.optim.Adam([
        {"params": gnn_fourier.parameters(), 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},  # FourierFT adapters
        {"params": zero_mlp.parameters(), 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},   # ControlNet
        {"params": list(projector.parameters()) + list(logreg.parameters()), 'lr': config['lr2'], 'weight_decay': config['wd2']}  # Projector + Classifier
    ])

    # Labels for evaluation
    train_labels = test_dataset.y[train_mask]
    val_labels = test_dataset.y[val_mask]
    test_labels = test_dataset.y[test_mask]
    
    # Pretrain dataset loader for SMMD loss
    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)
    
    # Tracking variables
    max_acc = 0
    max_test_acc = 0
    max_epoch = 0
    
    # 🔑 STEP 7: Print experiment setup and memory usage
    log_output('='*80, args)
    log_output(f'🚀 FourierFT + Heterophilic GraphControl Transfer', args)
    log_output(f'   Condition Type: {args.condition_type}', args)
    log_output(f'   {args.pretrain_dataset} → {args.test_dataset}', args)
    log_output(f'   FourierFT: n={args.n}, α={args.alpha}', args)
    log_output(f'   Few-shot: {args.few}, Shot: {args.shot if args.few else "N/A"}', args)
    
    # Print memory usage with new function
    all_modules = [gnn_fourier, zero_mlp, projector, logreg]
    print_memory_usage(all_modules, adapter_module_names=['fourier', 'adapter', 'zero_mlp'], method_name="FourierFT+ControlNet")
    
    log_output('='*80, args)

    # 🔑 STEP 8: Training loop with Dual Encoders + ControlNet
    for epoch in range(config.get('epoch1', 100)):
        # Set training mode for trainable modules
        gnn_fourier.train()
        zero_mlp.train()
        projector.train()
        logreg.train()
        
        optimizer.zero_grad()

        # Forward pass with FiLM modulation
        # 1. Get FiLM parameters from structural encoder + condition PE
        with torch.no_grad():
            P_cond_processed = encoder_S(P_cond, test_dataset.edge_index)  # Process condition PE through structural encoder
        gamma, beta = zero_mlp(P_cond_processed.mean(dim=0, keepdim=True))  # Global condition
        
        # 2. Forward through feature encoder + modulated FourierFT adapter
        feature_map = projector(test_dataset.x)
        
        # Apply FiLM modulation to FourierFT coefficients
        # This requires modifying the GNNFourierFT forward pass to accept FiLM params
        emb, emb1, emb2 = gnn_fourier(feature_map, test_dataset.edge_index, film_gamma=gamma, film_beta=beta)
        
        # 3. Classification
        logits = logreg(emb)
        train_logits = logits[train_mask]
        
        # 🔑 STEP 9: Compute 4-loss (cls + smmd + contrastive + struct_reg)
        # Loss 1: Classification loss
        cls_loss = loss_fn(train_logits, train_labels)
        
        # Loss 2: SMMD loss (domain alignment)
        smmd_loss_f = 0
        if config.get('lambda_map', 0) > 0:
            smmd_loss_f = calculate_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)
        
        # Loss 3: Contrastive loss (consistency between views)
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + 
                        batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()
        
        # Loss 4: Structural regularization loss
        loss_reg = 0
        if config.get('lambda_str', 0) > 0:
            loss_reg = calculate_reg_loss(logits, target_adj, device)

        # Combined loss
        loss = (config.get('lambda_cyc', 1) * cls_loss + 
                config.get('lambda_map', 5) * smmd_loss_f + 
                config.get('lambda_cos', 0) * ct_loss + 
                config.get('lambda_str', 0) * loss_reg)
        
        loss.backward()
        optimizer.step()

        # Training accuracy
        preds = torch.argmax(train_logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        
        # 🔑 STEP 10: Evaluation
        gnn_fourier.eval()
        zero_mlp.eval()
        projector.eval()
        logreg.eval()
        
        with torch.no_grad():
            # Re-compute with FiLM modulation
            P_cond_eval = encoder_S(P_cond, test_dataset.edge_index)
            gamma_eval, beta_eval = zero_mlp(P_cond_eval.mean(dim=0, keepdim=True))
            
            feature_map_eval = projector(test_dataset.x)
            emb_eval, _, _ = gnn_fourier(feature_map_eval, test_dataset.edge_index, 
                                       film_gamma=gamma_eval, film_beta=beta_eval)
            logits_eval = logreg(emb_eval)
            
            val_logits = logits_eval[val_mask]
            test_logits = logits_eval[test_mask]
            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]
            
            log_output('Epoch: {:03d}, train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}'.format(epoch, train_acc, val_acc, test_acc), args)
            
            if max_acc < val_acc:
                max_acc = val_acc
                max_test_acc = test_acc
                max_epoch = epoch + 1
                
    # 🔑 STEP 11: Final results and logging
    log_output('\n' + '='*80, args)
    log_output('🏆 FourierFT + Heterophilic GraphControl Results:', args)
    log_output(f'   Condition Type: {args.condition_type}', args)
    log_output(f'   Best Epoch: {max_epoch}, Val Acc: {max_acc:.4f}, Test Acc: {max_test_acc:.4f}', args)
    log_output(f'   Final Epoch: {epoch + 1}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}', args)
    
    # Print final memory usage summary
    all_modules = [gnn_fourier, zero_mlp, projector, logreg]
    memory_stats = print_memory_usage(all_modules, adapter_module_names=['fourier', 'adapter', 'zero_mlp'], method_name="FourierFT+ControlNet")
    
    log_output('='*80, args)
    
    # Save results
    result_path = './result'
    mkdir(result_path)
    
    # Enhanced result logging with condition type
    result_filename = f'{result_path}/GraphFourierFT_DualEncoder_{args.condition_type.upper()}.txt'
    if args.few:
        with open(result_filename, 'a') as f:
            f.write(f'Few: True, n: {args.n}, Shot: {args.shot}, Condition: {args.condition_type}, '
                   f'{args.pretrain_dataset} → {args.test_dataset}: '
                   f'BEST val_acc: {max_acc:.4f}, test_acc: {max_test_acc:.4f} | '
                   f'FINAL val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f} | '
                   f'Adapter Params: {memory_stats["adapter_params"]:,} ({memory_stats["adapter_mb"]:.2f}MB)\n')
    else:
        with open(result_filename, 'a') as f:
            f.write(f'Few: False, n: {args.n}, Condition: {args.condition_type}, '
                   f'{args.pretrain_dataset} → {args.test_dataset}: '
                   f'BEST val_acc: {max_acc:.4f}, test_acc: {max_test_acc:.4f} | '
                   f'FINAL val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f} | '
                   f'Adapter Params: {memory_stats["adapter_params"]:,} ({memory_stats["adapter_mb"]:.2f}MB)\n')
    
    # Log completion
    if hasattr(args, 'use_logging') and args.use_logging and hasattr(args, 'log_file'):
        log_output(f'\n✓ Results saved to: {result_filename}', args)
        log_output(f'✓ Log file saved to: {args.log_file}', args)
    
    return max_test_acc