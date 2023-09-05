# coding=utf-8
# Author: Jung
# Time: 2022/8/15 10:14

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import dgl
from dgl.nn import GraphConv
import torch.nn as nn
import argparse
from sklearn import metrics
import random
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy import sparse
import pickle as pkl
import community

random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

def modularity(adj: np.array, pred: np.array):
    """
    非重叠模块度
    :param adj: 邻接矩阵
    :param pred: 预测社区标签
    :return:
    """
    graph = nx.from_numpy_matrix(adj)
    part = pred.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur


def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_ac(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_f1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def calculate_entropy(k, pred_labels, feat):
    """
    :param k: 社区个数
    :param pred_labels: 预测社区
    :param num_nodes: 节点的个数
    :param feat: 节点属性
    :return:
    """
    # 初始化两个矩阵

    num_nodes = feat.shape[0]

    label_assemble = np.zeros(shape=(num_nodes, k))
    label_atts = np.zeros(shape=(k, feat.shape[1]))

    label_assemble[range(num_nodes), pred_labels] = 1
    label_assemble = label_assemble.T

    # 遍历每个社区
    for i in range(k):
        # 如果社区中的值大于0，则获得索引
        node_indx = np.where(label_assemble[i] > 0)
        # 获得索引下的所有属性
        node_feat = feat[node_indx]
        label_atts[i] = node_feat.sum(axis=0) # 向下加和

    __count_attrs = label_atts.sum(axis=1)
    __count_attrs = __count_attrs[:,np.newaxis]
    _tmp = label_atts / (__count_attrs + 1e-10)
    p = (_tmp) * - (np.log2(_tmp + 1e-10))

    p = p.sum(axis=1)
    label_assemble = label_assemble.sum(axis=1)
    __entropy = (label_assemble / num_nodes) * p
    return __entropy.sum()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='the number of epochs')


    return parser.parse_known_args()

def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

class GCN(nn.Module):
    def __init__(self, feat_dim, hid_dim, k):
        super(GCN, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.conv = GraphConv(feat_dim, hid_dim)
        self.conv2 = GraphConv(hid_dim, k)
        self.act = nn.ReLU()
    def forward(self, graph, feat):
        """ Notice: the final h need to use activation function """
        h = self.conv(graph, feat)
        h = self.act(h)
        h = self.conv2(graph, h)
        return h

def knn_graph(feat, topk, weight = False, loop = True):
    sim_feat = cosine_similarity(feat)
    sim_matrix = np.zeros(shape=(feat.shape[0], feat.shape[0]))

    inds = []
    for i in range(sim_feat.shape[0]):
        ind = np.argpartition(sim_feat[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    for i, vs in enumerate(inds):
        for v in vs:
            if v == i:
                pass
            else:
                if weight is True:
                    sim_matrix[i][v] = sim_feat[i][v]
                    sim_matrix[v][i] = sim_feat[v][i]
                else:
                    sim_matrix[i][v] = 1
                    sim_matrix[v][i] = 1

    sp_matrix = sparse.csr_matrix(sim_matrix)
    dgl_matrix = dgl.from_scipy(sp_matrix)
    if loop is True:
        dgl_matrix = dgl.add_self_loop(dgl_matrix)
    return dgl_matrix


def load_data(name: str):
    with open("datasets/"+name+".pkl", 'rb') as f:
        data = pkl.load(f)
    graph = dgl.from_scipy(data['topo'])  # 数据集中自带自环
    return graph, data['attr'].toarray(), data['label']

class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size= 16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)

class Ays(nn.Module):
    def __init__(self, graph, kgraph, feat, label):
        super(Ays, self).__init__()
        self.graph = graph
        self.kgraph = kgraph
        self.adj = graph.adjacency_matrix().to_dense()
        self.kadj = kgraph.adjacency_matrix().to_dense()
        self.feat = torch.from_numpy(feat).to(torch.float)
        self.labels = label
        self.num_nodes, self.feat_dim = self.feat.shape
        self.k = len(np.unique(self.labels))
        self.atten = Attention(self.k)
        self.t = self.k # number of topics equal number of communities

        """
            hid dim 
            Cora: 200
            Citeseer: 90
            uai2010: 90
            pubmed 128
            blogcatalog 128
            flickr 128
            webkb 32
            
        """
        hid = 200
        self.gcn = GCN(self.feat_dim, hid, self.k)
        self.knn_gcn = GCN(self.feat_dim, hid, self.k)
        self.B = self.get_b_of_modularity(self.adj)

        """ activate functions """
        self.relu = nn.ReLU()
        self.log_sig = nn.LogSigmoid()
        self.soft = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.bce_with_log = nn.BCEWithLogitsLoss()


        """ parameters """
        self.topo_par = nn.Parameter(torch.FloatTensor(self.k, self.k))
        self.feat_par = nn.Parameter(torch.FloatTensor(self.t, self.feat_dim))
        self.topic = nn.Parameter(torch.FloatTensor(self.k, self.t))
        torch.nn.init.xavier_uniform_(self.topo_par)
        torch.nn.init.xavier_uniform_(self.feat_par)
        torch.nn.init.xavier_uniform_(self.topic)

    def get_b_of_modularity(self, A):
        K = 1 / (A.sum().item()) * (A.sum(axis=1).reshape(A.shape[0], 1) @ A.sum(axis=1).reshape(1, A.shape[0]))
        return A - K

    def constraint(self):
        w = self.topo_par.data.clamp(0, 1)
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.topo_par.data = w

        w = self.topic.data.clamp(0, 1)
        col_sums = w.sum(dim=0)
        w = torch.divide(w.t(), torch.reshape(col_sums, (-1, 1))).t()
        self.topic.data = w

    def forward(self):

        h = self.gcn(self.graph, self.feat)

        h_knn = self.gcn(self.kgraph, self.feat) # embedding of KNN Graph

        h_att = self.atten(torch.stack([h, h_knn], dim=1)) # fusion these two representations

        emb_feat = h_att @ self.topic @ self.feat_par # reconstruction (attribute)

        emb_topo = h_att @ self.topo_par @ h_att.t() # reconstruction (topology)

        return self.soft(h_att), self.soft(emb_feat), self.soft(emb_topo)


if __name__ == "__main__":
    args, _ = parse_args()
    printConfig(args)
    graph, feat, label = load_data(args.dataset)

    kgraph = knn_graph(feat, 20) # top-k number

    model = Ays(graph, kgraph, feat, label)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        stmp = epoch

        model.train()
        optimizer.zero_grad()

        emb, emb_feat, emb_topo = model()
        feat_loss = F.mse_loss(model.feat, emb_feat) # loss value of attribute graph

        topo_loss = F.mse_loss(model.adj, emb_topo) # loss value of topology graph

        modu_loss = torch.trace(emb.t() @ model.B @ emb) # modularity

        loss = 1 * feat_loss + 10* topo_loss + - modu_loss*0.01

        model.eval()
        pred = emb.argmax(dim=1)
        nmi = compute_nmi(pred, model.labels)
        ari = computer_ari(model.labels, pred)
        Q = modularity(model.adj.numpy(), pred)
        etp = calculate_entropy(model.k, pred.detach().numpy(),model.feat.numpy())
        print(
            'epoch={},  nmi: {:.3f},  ari= {:.3f},  Q = {:.3f},  AE = {:.3f}'.format(
                epoch,
                nmi,
                ari,
                Q,
                etp
            ))
        loss.backward()
        optimizer.step()
        model.constraint()



