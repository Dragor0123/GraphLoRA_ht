import yaml
from yaml import SafeLoader
import argparse
import torch
import random
import numpy as np
import os
from datetime import datetime
from pre_train import pretrain
from model.GraphLoRA import transfer
from model.GraphFourierFT import transfer_fourier
from util import get_parameter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dataset', type=str, default='PubMed')
    parser.add_argument('--test_dataset', type=str, default='CiteSeer')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--pretext', type=str, default='GRACE')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--is_pretrain', type=bool, default=False)
    parser.add_argument('--is_transfer', type=bool, default=True)
    parser.add_argument('--is_reduction', type=bool, default=True)
    parser.add_argument('--few', type=bool, default=False)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--sup_weight', type=float, default=0.2)
    parser.add_argument('--method', type=str, default='lora', choices=['lora', 'fourier'], help='Choose fine-tuning method: lora or fourier')
    parser.add_argument('--r', type=int, default=32, help='LoRA rank (only used when method=lora)')
    parser.add_argument('--n', type=int, default=1000, help='Number of spectral coefficients for FourierFT (only used when method=fourier)')
    parser.add_argument('--alpha', type=float, default=300.0, help='Scaling factor for FourierFT (only used when method=fourier)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_logging', type=bool, default=False, help='Save results to log file instead of printing')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save log files')
    parser.add_argument('--condition_type', type=str, default='role', choices=['role', 'inverse', 'ppr'], 
                        help='Condition type for Heterophilic GraphControl')
    parser.add_argument('--encoder_S_ckpt', type=str, default=None, 
                        help='Path to structural encoder checkpoint')
    parser.add_argument('--encoder_F_ckpt', type=str, default=None, 
                        help='Path to feature encoder checkpoint')
    args = parser.parse_args()
    args = get_parameter(args)

    set_seed(args.seed)
    
    # Setup logging
    log_file = None
    if args.use_logging:
        os.makedirs(args.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{args.pretrain_dataset}_{args.test_dataset}_{timestamp}_seed{args.seed}.log"
        log_file = os.path.join(args.log_dir, log_filename)
        args.log_file = log_file
    
    assert args.gpu_id in range(0, 2)
    torch.cuda.set_device(args.gpu_id)

    if args.is_pretrain:
        if args.pretext == 'STRUCT':
            # Structural pretraining - use struct config
            config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)['pretrain']['struct']
            path_S = pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction, args.seed)
            print(f"✓ Structural encoder saved: {path_S}")
        elif args.pretext == 'GRACE':
            # Feature pretraining - use dataset-specific config
            config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)[args.pretrain_dataset]
            path_F = pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction, args.seed)
            print(f"✓ Feature encoder saved: {path_F}")
        else:
            raise ValueError(f"Unknown pretext: {args.pretext}")
    
    if args.is_transfer:
        # Auto-set encoder checkpoint paths based on naming convention
        # Format: {dataset}.{STRUCT|FEAT}.GAT.{is_reduction}.pth
        if not args.encoder_S_ckpt:
            args.encoder_S_ckpt = f'./pre_trained_gnn/{args.pretrain_dataset}.STRUCT.GAT.{args.is_reduction}.pth'
        if not args.encoder_F_ckpt:
            args.encoder_F_ckpt = f'./pre_trained_gnn/{args.pretrain_dataset}.FEAT.GAT.{args.is_reduction}.pth'
        
        config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']
        
        print(f"🔄 Transfer Learning Setup:")
        print(f"   • Structural Encoder: {args.encoder_S_ckpt}")
        print(f"   • Feature Encoder: {args.encoder_F_ckpt}")
        print(f"   • Condition Type: {args.condition_type}")
        print(f"   • Method: {args.method}")
        
        if args.method == 'lora':
            print(f"   • LoRA rank: {args.r}")
            transfer(args, config_transfer, args.gpu_id, args.is_reduction)
        elif args.method == 'fourier':
            print(f"   • FourierFT coefficients: {args.n}, alpha: {args.alpha}")
            transfer_fourier(args, config_transfer, args.gpu_id, args.is_reduction)
        else:
            raise ValueError(f"Unknown method: {args.method}")
