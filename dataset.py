from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import scipy.io as sio
from torch_geometric.datasets import HeterophilousGraphDataset
import scipy.sparse as sp
import networkx as nx
from numpy import inf
import pickle
#device = torch.device('cuda:'+str('1') if torch.cuda.is_available() else 'cpu')
def load_mat(dataset):
    """Load .mat dataset."""
    data = sio.loadmat("/home/guoguoai/code/data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    attr = sp.lil_matrix(attr)
    
    return label, attr, adj

def convert_to_dgl_graph(label, attr, adj):
    """Convert attributes, network, and label to DGL graph."""
    # Convert the adjacency matrix to a DGL graph
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    graph = dgl.from_networkx(nx_graph)
    #print(label.shape)
    # Add node features to the graph
    
    attr_dense = attr.toarray()
        # If attr is already dense, convert directly to tensor
    graph.ndata['feature'] = torch.tensor(attr_dense, dtype=torch.float32)

    # Add graph label (for graph classification tasks)
    graph.ndata['label'] = torch.tensor(label.squeeze(), dtype=torch.long)
    
    return graph

class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('/home/guoguoai/code/data/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs('/home/guoguoai/code/data/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('/home/guoguoai/code/data/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == 'tsocial':
            graph, label_dict = load_graphs('/home/guoguoai/code/data/tsocial')
            graph = graph[0]
            #adj = graph.adjacency_matrix()
            #print("Sparse adjacency matrix:\n", adj)

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]           
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                #graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            #print(graph.edge_index)
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                #graph = dgl.add_self_loop(graph)
        elif name in ['reddit','photo','elliptic']:
            label, feat, adj = load_mat(name)
            graph = convert_to_dgl_graph(label, feat, adj)
        elif name in ["tolokers", "questions"]:
            root = "./home/guoguoai/code/data/"
            dataset = HeterophilousGraphDataset(root=root, name=name)
            graph_dgl = dataset[0]
        elif name in ['dgraphfin']:
            graph = dgl.load_graphs('dataset/'+name)[0][0]
        else:
            print('no such dataset')
            exit(1)
        if name in ["tolokers", "questions"]:
            graph = dgl.DGLGraph()
            graph.add_nodes(graph_dgl.num_nodes)
            graph.add_edges(graph_dgl.edge_index[0], graph_dgl.edge_index[1])

            graph.ndata['label'] = graph_dgl.y.data.long().squeeze(-1)
            graph.ndata['feature'] = graph_dgl.x.float()
        else:
            graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
            graph.ndata['feature'] = graph.ndata['feature'].float()
        
        adj = sp.load_npz('adj_matrix_{}.npz'.format(name))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        # 保存邻接矩阵
        
        self.graph = graph
        self.adj = adj

def get_sp_adj(graph,name):
    #print(graph)
    #src, dst = graph.edges()
    #edge_index = torch.stack([src, dst], dim=0)
    #self.edge_index = edge_index
    graph =dgl.remove_self_loop(graph)
    #adj = nx.adjacency_matrix(graph)
    # 将 DGLHeteroGraph 转换为 NetworkX 图对象
    nx_graph = dgl.to_networkx(graph)

    # 提取邻接矩阵
    adj = nx.adjacency_matrix(nx_graph)
    # 对称化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 添加自环
    adj = adj + sp.eye(adj.shape[0])

    # 计算度矩阵 D
    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = D - adj
    with np.errstate(divide='ignore'):
        D_norm = D ** (-0.5)
    D_norm[D_norm == inf] = 0
    adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))
    sp.save_npz('adj_matrix_{}.npz'.format(name), adj)
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


