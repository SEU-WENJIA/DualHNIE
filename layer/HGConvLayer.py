
import torch
from torch import nn
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import os
import torch.nn.functional as F
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
    Standard hypergraph convolution layer.

    This layer performs:
        1. Node -> Hyperedge aggregation
        2. Hyperedge -> Node aggregation
        3. Linear transformation + optional residual connection
        4. Optional activation (ReLU)

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    edge_dim : int, optional
        Hyperedge feature dimension, required if use_edge_feat=True.
    use_edge_feat : bool, default=False
        Whether to incorporate hyperedge features.
    feat_drop : float, default=0.
        Dropout rate on node features.
    residual : bool, default=True
        Whether to use residual connections.
    batch_norm : bool, default=True
        Whether to apply batch normalization.

    Forward Inputs
    --------------
    node_feats : torch.Tensor, shape [num_nodes, in_dim]
        Node feature matrix.
    H : torch.Tensor, shape [num_nodes, num_hyperedges]
        Node-hyperedge incidence matrix (0/1).
    edge_feats : torch.Tensor, shape [num_hyperedges, edge_dim], optional
        Hyperedge features (only used if use_edge_feat=True).

    Returns
    -------
    torch.Tensor, shape [num_nodes, out_dim]
        Updated node features.
    """
    def __init__(self, in_dim, out_dim, edge_dim=None, use_edge_feat=False, 
                 feat_drop=0., residual=True, batch_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_edge_feat = use_edge_feat
        self.feat_drop = feat_drop
        self.residual = residual
        self.batch_norm = batch_norm

        # Node feature linear transformation
        self.node_fc = nn.Linear(in_dim, out_dim, bias=False)

        # Optional hyperedge feature linear transformation
        if use_edge_feat and edge_dim is not None:
            self.edge_fc = nn.Linear(edge_dim, out_dim, bias=False)
        else:
            self.edge_fc = None

        # Dropout layer
        self.feat_drop_layer = nn.Dropout(feat_drop)

        # Batch normalization
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)

        # Residual connection
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = Identity()

        self.activation = F.relu

    def forward(self, node_feats, H, edge_feats=None):
        """
        Forward pass for hypergraph convolution.

        Parameters
        ----------
        node_feats : torch.Tensor, shape [N, in_dim]
            Node feature matrix.
        H : torch.Tensor, shape [N, E]
            Node-hyperedge incidence matrix (0/1).
        edge_feats : torch.Tensor, shape [E, edge_dim], optional
            Hyperedge features (used only if use_edge_feat=True).

        Returns
        -------
        torch.Tensor, shape [N, out_dim]
            Updated node features.
        """
        device = node_feats.device
        H = H.to(device)
        n_nodes, n_edges = H.shape

        # Node and hyperedge degrees
        dv = H.sum(dim=1).clamp(min=1)  # node degrees
        de = H.sum(dim=0).clamp(min=1)  # hyperedge degrees

        dv_inv_sqrt = dv.pow(-0.5)      # D_v^-1/2
        de_inv = de.pow(-1.0)           # D_e^-1

        # Normalize incidence matrix
        H_norm = dv_inv_sqrt.view(-1, 1) * H * de_inv.view(1, -1)  # [N, E]

        # Node -> Hyperedge aggregation
        hyper_edge_feat = H_norm.T @ node_feats  # [E, in_dim]

        # Optionally add hyperedge features
        if self.use_edge_feat and edge_feats is not None:
            hyper_edge_feat += self.edge_fc(edge_feats)

        # Hyperedge -> Node aggregation
        out = H_norm @ hyper_edge_feat  # [N, in_dim]

        # Linear transformation
        out = self.node_fc(out)

        # Residual connection
        if self.residual and hasattr(self, 'res_fc'):
            out += self.res_fc(node_feats)

        # Activation
        if self.activation is not None:
            out = self.activation(out)

        # Optional batch normalization
        if self.batch_norm:
            out = self.bn(out)

        return out
