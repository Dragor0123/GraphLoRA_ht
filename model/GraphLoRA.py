from model.GNN_model import GNN, GNNLoRA
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss, batched_mmd_loss
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
    idx_a = torch.empty(0, dtype=torch.long, device=device)
    idx_b = torch.empty(0, dtype=torch.long, device=device)
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
    
    # Log experiment details
    log_output('='*80, args)
    log_output(f'Transfer Learning: {args.pretrain_dataset} -> {args.test_dataset}', args)
    log_output(f'Seed: {args.seed}, LoRA rank: {args.r}', args)
    log_output(f'Few-shot: {args.few}, Shot: {args.shot if args.few else "N/A"}', args)
    log_output(f'Hyperparameters: l1={args.l1}, l2={args.l2}, l3={args.l3}, l4={args.l4}', args)
    log_output('='*80, args)

    for epoch in range(0, args.num_epochs):
        logreg.train()
        projector.train()

        feature_map = projector(test_dataset.x)
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)
        train_labels = test_dataset.y[train_mask]
        optimizer.zero_grad()

        # Calculate SMMD loss if enabled (args.l2 > 0)
        smmd_loss_f = 0
        if args.l2 > 0:
            smmd_loss_f = calculate_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)
            # smmd_loss_f = calculate_mmd_loss(feature_map, pretrain_graph_loader, SMMD, 128)
        
        # Calculate contrastive loss
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()
        
        logits = logreg(emb)
        train_logits = logits[train_mask]

        # Calculate regularization loss if enabled (args.l4 > 0)
        loss_reg = 0
        if args.l4 > 0:
            loss_reg = calculate_reg_loss(logits, target_adj, device)

        preds = torch.argmax(train_logits, dim=1)
        cls_loss = loss_fn(train_logits, train_labels)
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_reg
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        
        # Evaluation with updated parameters
        logreg.eval()
        projector.eval()
        gnn2.eval()
        with torch.no_grad():
            # Re-compute embeddings and logits with updated parameters
            feature_map_eval = projector(test_dataset.x)
            emb_eval, _, _ = gnn2(feature_map_eval, test_dataset.edge_index)
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
        gnn2.train()
    log_output('\n' + '='*80, args)
    log_output('Best Results:', args)
    log_output('Best epoch: {}, val_acc: {:.4f}, test_acc: {:.4f}'.format(max_epoch, max_acc, max_test_acc), args)
    log_output('Final Results:', args)
    log_output('Final epoch: {}, val_acc: {:.4f}, test_acc: {:.4f}'.format(epoch + 1, val_acc, test_acc), args)
    log_output('='*80, args)
    result_path = './result'
    mkdir(result_path)
    if args.few:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: True, r: %d, Shot: %d, %s to %s: BEST val_acc: %f, test_acc: %f | FINAL val_acc: %f, test_acc: %f\n'%(args.r, args.shot, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc, val_acc, test_acc))
    else:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: False, r: %d, %s to %s: BEST val_acc: %f, test_acc: %f | FINAL val_acc: %f, test_acc: %f\n'%(args.r, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc, val_acc, test_acc))
    
    # Also save summary to log file if logging is enabled
    if hasattr(args, 'use_logging') and args.use_logging and hasattr(args, 'log_file'):
        log_output('\nResults saved to: ' + result_path + '/GraphLoRA.txt', args)
        log_output('Log file saved to: ' + args.log_file, args)