import yaml
from yaml import SafeLoader
import argparse
import torch
import random
import numpy as np
from pre_train import pretrain
from model.GraphLoRA import transfer
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
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args = get_parameter(args)

    set_seed(args.seed)
    
    assert args.gpu_id in range(0, 2)
    torch.cuda.set_device(args.gpu_id)

    if args.is_pretrain:
        config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)[args.pretrain_dataset]
        pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction, args.seed)
    
    if args.is_transfer:
        config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']
        transfer(args, config_transfer, args.gpu_id, args.is_reduction)
