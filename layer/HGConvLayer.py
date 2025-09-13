
import torch
from torch import nn
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import dgl
import numpy as np
import os
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["TORCH_USE_CUDA_DSA"] = "1"


torch.backends.cudnn.benchmark = True

def expand_as_pair(input):
    if isinstance(input, (tuple, list)):
        return input
    else:
        return input, input

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x





class HypergraphConvLayer(nn.Module):
    """
    标准超图卷积层
    输入：
        node_feats: [num_nodes, in_dim]
        edge_feats: [num_hyperedges, edge_dim] 可选
        H: [num_nodes, num_hyperedges] incidence matrix (0/1)
    输出：
        updated_nodes: [num_nodes, out_dim]
    """
    def __init__(self, in_dim, out_dim, edge_dim=None, use_edge_feat=False, feat_drop=0., residual=True, batch_norm=True):
        super(HypergraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_feat = use_edge_feat
        self.feat_drop = feat_drop
        self.residual = residual
        self.batch_norm = batch_norm

        # 节点特征线性变换
        self.node_fc = nn.Linear(in_dim, out_dim, bias=False)
        # 可选：超边特征映射
        if use_edge_feat and edge_dim is not None:
            self.edge_fc = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.edge_fc = None

        # Dropout
        self.feat_drop_layer = nn.Dropout(feat_drop)

        # BatchNorm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)

        # 残差
        if residual:
            
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = Identity()
        
        self.activation = F.relu
                


    def forward(self, node_feats, H, edge_feats=None):
        """
        node_feats: [N, in_dim]
        H: [N, E] 0/1 incidence matrix
        edge_feats: [E, edge_dim] 可选
        """
        device = node_feats.device
        H = H.to(device)
        n_nodes, n_edges = H.shape

        # Node and hyperedge degrees
        dv = H.sum(dim=1)  # [n_nodes]
        de = H.sum(dim=0)  # [n_edges]

        # Avoid division by zero
        dv = dv.clamp(min=1)
        de = de.clamp(min=1)

        dv_inv_sqrt = dv.pow(-0.5)  # D_v^-1/2
        de_inv = de.pow(-1.0)       # D_e^-1

        # Normalize H
        H_norm = dv_inv_sqrt.view(-1, 1) * H * de_inv.view(1, -1)  # [n_nodes, n_edges]

        # Node -> Hyperedge aggregation
        hyper_edge_feat = H_norm.T @ node_feats  # [n_edges, in_feats]

        # Hyperedge -> Node aggregation
        out = H_norm @ hyper_edge_feat     # [n_nodes, in_feats]

        out = self.node_fc(out)
        
        # Residual connection
        if self.res_fc is not None:
            resval = self.res_fc(node_feats)
            out += resval



        # Activation
        if self.activation:
            out = self.activation(out)

        return out
