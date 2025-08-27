from model.GNN_model import GNN, GNNLoRA
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss
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


def transfer(args, config, gpu_id, is_reduction):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # load data
    pretrian_datapath = os.path.join('./datasets', args.pretrain_dataset)
    test_datapath = os.path.join('./datasets', args.test_dataset)
    pretrain_dataset = get_dataset(pretrian_datapath, args.pretrain_dataset)[0]
    test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_dataset = feature_reduce(pretrain_dataset)
        test_dataset = feature_reduce(test_dataset)
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset = test_dataset.to(device)

    # target adj
    target_adj = to_dense_adj(test_dataset.edge_index)[0]
    # pos_weight = float(test_dataset.x.shape[0] * test_dataset.x.shape[0] - test_dataset.edge_index.shape[1]) / test_dataset.edge_index.shape[1]
    # weight_mask = target_adj.view(-1) == 1
    # weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    # weight_tensor[weight_mask] = pos_weight / 10

    gnn = GNN(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    model_path = "./pre_trained_gnn/{}.{}.{}.{}.pth".format(args.pretrain_dataset, args.pretext, config['gnn_type'], args.is_reduction)
    gnn.load_state_dict(torch.load(model_path))
    gnn.to(device)
    gnn.eval()
    for param in gnn.conv.parameters():
        param.requires_grad = False

    gnn2 = GNNLoRA(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']), gnn, config['gnn_type'], config['num_layers'], r=args.r)
    gnn2.to(device)
    gnn2.train()

    SMMD = SMMDLoss().to(device)

    projector = Projector(test_dataset.x.shape[1], pretrain_dataset.x.shape[1])
    projector = projector.to(device)
    projector.train()

    # optimizer
    logreg = LogReg(config['output_dim'], max(test_dataset.y) + 1)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

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
    mask = torch.zeros((test_dataset.x.shape[0], test_dataset.x.shape[0])).to(device)
    ppr_weight = get_ppr_weight(test_dataset)
    idx_a = torch.tensor([]).to(device)
    idx_b = torch.tensor([]).to(device)
    for i in range(max(test_dataset.y) + 1):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        train_label = test_dataset.y[train_idx]
        idx_a = torch.concat((idx_a, train_idx[train_label == i].repeat_interleave(len(train_idx[train_label == i]))))
        idx_b = torch.concat((idx_b, train_idx[train_label == i].repeat(len(train_idx[train_label == i]))))
    mask = torch.sparse_coo_tensor(indices=torch.stack((idx_a, idx_b)), values=torch.ones(len(idx_a)).to(device), size=[test_dataset.x.shape[0], test_dataset.x.shape[0]]).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(test_dataset.x.shape[0]).to(device)
    optimizer = torch.optim.Adam([{"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1}, {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, {"params": gnn2.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}])

    test_dataset.train_mask = train_mask
    test_dataset.val_mask = val_mask
    test_dataset.test_mask = test_mask

    train_labels = test_dataset.y[train_mask]
    val_labels = test_dataset.y[val_mask]
    test_labels = test_dataset.y[test_mask]

    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)
    max_acc = 0
    max_test_acc = 0
    max_epoch = 0

    for epoch in range(0, args.num_epochs):
        logreg.train()
        projector.train()
  
        pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
        weight_mask = target_adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight

        feature_map = projector(test_dataset.x)
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)
        train_labels = test_dataset.y[train_mask]
        optimizer.zero_grad()

        smmd_loss_f = 0.0# batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)      
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()
        logits = logreg(emb)
        train_logits = logits[train_mask]
        
        # Debug: Check devices and intermediate values
        if epoch >= 0 and epoch < 5:
            print(f"Epoch {epoch} Root Cause Analysis:")
            print(f"logits device: {logits.device}")
            print(f"logits shape: {logits.shape}")
            print(f"logits min/max: {logits.min().item():.6f} / {logits.max().item():.6f}")
            print(f"logits has NaN: {torch.isnan(logits).any().item()}")
            print(f"logits has Inf: {torch.isinf(logits).any().item()}")
            print(f"target_adj device: {target_adj.device}")
            print(f"emb device: {emb.device}")
            print(f"emb min/max: {emb.min().item():.6f} / {emb.max().item():.6f}")
            print(f"emb has NaN: {torch.isnan(emb).any().item()}")
            print(f"emb has Inf: {torch.isinf(emb).any().item()}")
        
        # Check softmax computation step by step
        softmax_logits = torch.softmax(logits, dim=1)
        
        if epoch >= 0 and epoch < 5:
            print(f"softmax_logits min/max: {softmax_logits.min().item():.6f} / {softmax_logits.max().item():.6f}")
            print(f"softmax_logits has NaN: {torch.isnan(softmax_logits).any().item()}")
            print(f"softmax_logits has Inf: {torch.isinf(softmax_logits).any().item()}")
        
        # Check if dataset is heterophilic - modify adjacency reconstruction accordingly
        is_heterophilic = args.test_dataset in ['Squirrel', 'Chameleon', 'Actor', 'Cornell', 'Texas', 'Wisconsin']
        
        if is_heterophilic:
            # For heterophilic graphs: use 2-hop homophilic assumption
            # Step 1: Compute 2-hop adjacency matrix
            A1 = to_dense_adj(test_dataset.edge_index)[0]  # 1-hop adjacency
            A2 = torch.matmul(A1, A1)  # 2-hop adjacency matrix
            I = torch.eye(test_dataset.x.shape[0], device=device)  # Identity matrix (self-loops)
            A2_pure = A2 - A1  # Remove 1-hop connections, keep 2-hop + self-loops from A2
            
            # Binarize 2-hop connections (any path count > 0 becomes 1)
            A2_binary = (A2_pure > 0).float()
            
            # Add self-loops explicitly
            A2_with_self = A2_binary + I
            A2_with_self = torch.clamp(A2_with_self, max=1.0)  # Ensure binary values
            
            # Step 2: Mixed reconstruction strategy
            similarity_matrix = torch.matmul(softmax_logits, softmax_logits.T)
            
            # For 2-hop + self: use similarity (homophilic)
            # For others: use dissimilarity (heterophilic) 
            matmul_result = A2_with_self * similarity_matrix + (1.0 - A2_with_self) * (1.0 - similarity_matrix)
        else:
            # For homophilic graphs: nodes with SIMILAR predictions should be connected
            matmul_result = torch.matmul(softmax_logits, softmax_logits.T)
        
        if epoch >= 0 and epoch < 5:
            print(f"Dataset type: {'Heterophilic' if is_heterophilic else 'Homophilic'}")
            if is_heterophilic:
                print(f"2-hop adjacency stats: edges={A2_with_self.sum().item():.0f}, density={A2_with_self.mean().item():.4f}")
                print(f"similarity_matrix min/max: {similarity_matrix.min().item():.6f} / {similarity_matrix.max().item():.6f}")
            print(f"matmul_result min/max: {matmul_result.min().item():.6f} / {matmul_result.max().item():.6f}")
            print(f"matmul_result has NaN: {torch.isnan(matmul_result).any().item()}")
            print(f"matmul_result has Inf: {torch.isinf(matmul_result).any().item()}")
        
        rec_adj = torch.sigmoid(matmul_result)
        
        if epoch >= 0 and epoch < 5:
            print(f"rec_adj min/max: {rec_adj.min().item():.6f} / {rec_adj.max().item():.6f}")
            print(f"rec_adj has NaN: {torch.isnan(rec_adj).any().item()}")
            print(f"rec_adj has Inf: {torch.isinf(rec_adj).any().item()}")
        
        # Check if rec_adj goes outside [0,1] range (which should be impossible after sigmoid)
        rec_adj_flat = rec_adj.view(-1)
        outside_range_mask = (rec_adj_flat < 0) | (rec_adj_flat > 1)
        if outside_range_mask.any():
            print(f"CRITICAL: rec_adj contains values outside [0,1] at epoch {epoch}!")
            outside_values = rec_adj_flat[outside_range_mask]
            print(f"Number of values outside [0,1]: {outside_range_mask.sum().item()}")
            print(f"Outside values (first 10): {outside_values[:10].tolist()}")
            print(f"Min outside value: {outside_values.min().item()}")
            print(f"Max outside value: {outside_values.max().item()}")
            
            # Check the input to sigmoid as well
            matmul_flat = matmul_result.view(-1)
            extreme_inputs = matmul_flat[outside_range_mask]
            print(f"Corresponding matmul_result values (first 10): {extreme_inputs[:10].tolist()}")
            print(f"Min matmul input: {extreme_inputs.min().item()}")
            print(f"Max matmul input: {extreme_inputs.max().item()}")
        
        # Also check for any values that are exactly 0 or 1 (edge cases for BCE)
        exact_zero_mask = (rec_adj_flat == 0)
        exact_one_mask = (rec_adj_flat == 1)
        if exact_zero_mask.any() or exact_one_mask.any():
            print(f"Warning at epoch {epoch}: rec_adj contains exact 0s: {exact_zero_mask.sum().item()}, exact 1s: {exact_one_mask.sum().item()}")
        
        # Temporarily disable loss_rec calculation to test
        
        target_adj_clamped = torch.clamp(target_adj, min=0.0, max=1.0)
        
        # Add epsilon for numerical stability in BCE
        epsilon = 1e-7
        rec_adj_stable = torch.clamp(rec_adj, min=epsilon, max=1-epsilon)
        loss_rec = F.binary_cross_entropy(rec_adj_stable.view(-1), target_adj_clamped.view(-1), weight=weight_tensor)
        
        preds = torch.argmax(train_logits, dim=1)
        cls_loss = loss_fn(train_logits, train_labels)
        # Use a small coefficient for reconstruction loss to avoid overwhelming the classification loss
        loss_rec_coeff = 0.01 if is_heterophilic else 0.1
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + loss_rec_coeff * loss_rec
        # loss = args.l1 * cls_loss + args.l2 * 0.0 +  args.l3 * ct_loss + loss_rec_coeff * loss_rec
        
        # Check for NaN in loss before backward pass
        if torch.isnan(loss):
            print(f"Warning: NaN detected in loss at epoch {epoch}")
            print(f"cls_loss: {cls_loss}, smmd_loss_f: {smmd_loss_f}, ct_loss: {ct_loss}, loss_rec: {loss_rec}")
            optimizer.zero_grad()
            continue
            
        loss.backward()
        
        # Add gradient clipping to prevent gradient explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(logreg.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(gnn2.parameters(), max_norm=1.0)
        
        # Check gradient norm
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"Warning: Invalid gradient norm {grad_norm} at epoch {epoch}")
            optimizer.zero_grad()
            continue
        
        optimizer.step()

        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        logreg.eval()
        projector.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            test_logits = logits[test_mask]
            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]
            print('Epoch: {}, train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(epoch, train_acc, val_acc, test_acc))
            if max_acc < val_acc:
                max_acc = val_acc
                max_test_acc = test_acc
                max_epoch = epoch + 1
    print('epoch: {}, val_acc: {:4f}, test_acc: {:4f}'.format(max_epoch, max_acc, max_test_acc))
    result_path = './result'
    mkdir(result_path)
    if args.few:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: True, r: %d, Shot: %d, %s to %s: val_acc: %f, test_acc: %f\n'%(args.r, args.shot, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))
    else:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: False, r: %d, %s to %s: val_acc: %f, test_acc: %f\n'%(args.r, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))