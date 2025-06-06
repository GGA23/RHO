import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
#from polt import visualize_dual_embeddings_3d
from model import OC_Auto
from utils import count_parameters, init_params, get_split
from dataset import Dataset
from similarity import get_similarity,draw_homo

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_center_c(adj, inputs, net, device, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c_global = torch.zeros(net.rep_dim).to(device)
        c_local = torch.zeros(net.rep_dim).to(device)
        net.eval()
        with torch.no_grad():
            outputs_global, outputs_local,  _ = net(adj, inputs)

            n_samples = outputs_global.shape[0]
            c_global =torch.sum(outputs_global, dim=0)
            c_local =torch.sum(outputs_local, dim=0)

        c_global /= n_samples
        c_local /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c_local[(abs(c_local) < eps) & (c_local < 0)] = -eps
        c_local[(abs(c_local) < eps) & (c_local > 0)] = eps

        c_global[(abs(c_global) < eps) & (c_global < 0)] = -eps
        c_global[(abs(c_global) < eps) & (c_global > 0)] = eps

        return c_local, c_global


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='tfinance',
                        choices=['amazon','tfinance','reddit','photo','elliptic','tolokers','questions'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument("--train_ratio", type=float, default=0.3, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio")
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--hidden1', type=int, default=1024)
    parser.add_argument('--hidden2', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=0) #1024 for 'tfinance','elliptic','questions', 32 for dgraph
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0) #0.1 for reddit and photo
    parser.add_argument('--tau', type=float, default=0.2)
    

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

    graph = Dataset(args.dataset).graph
    adj = Dataset(args.dataset).adj
    labels = graph.ndata['label']
    features = graph.ndata['feature']
    in_feats = graph.ndata['feature'].shape[1]

    num_node = features.shape[0]
    idx_train, idx_val, idx_test = get_split(num_node, labels, args.train_ratio)

    net = OC_Auto(in_feats, args.hidden1, args.hidden2, args.nlayers, args.batch_size, args.tau).to(device) 
    #graph = graph.to(device)
    adj = adj.to(device)
    features = features.to(device) 
    labels = labels.to(device) 
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(count_parameters(net))
    net.load_state_dict(torch.load('checkpoint/{}_test'.format(args.dataset)))
    net.eval()
    outputs_global, outputs_local, nce_loss = net(adj,features)
    center_local, center_global = init_center_c(adj, features, net, device, eps=0.1)

    scores = ((torch.sum((outputs_global[idx_test] - center_global) ** 2, dim=1))+\
                      (torch.sum((outputs_local[idx_test] - center_local) ** 2, dim=1)))/2
                       
            
    labels = np.array(labels.cpu().data.numpy())
    scores = np.array(scores.cpu().data.numpy())

    precision, recall, _ = precision_recall_curve(labels[idx_test], scores)
    
    # 计算 AUROC
    auroc = roc_auc_score(labels[idx_test], scores)
    print(" Test set  AUROC: {:.4f}".format(100. * auroc))
    auprc = auc(recall, precision)
    print(" Test set  AUPRC: {:.4f}".format(100. * auprc))


