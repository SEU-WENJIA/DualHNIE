# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl.function as fn
# from dgl.nn.pytorch.utils import Identity
from torch_scatter import scatter_softmax, scatter_add, scatter_max
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   


import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class SCAHGTLayer(nn.Module):
    """
    Structure- and Content-Aware Hypergraph Transformer Layer (SCA-HGT).

    This layer implements a hypergraph attention mechanism with:
        1. Node-to-hyperedge attention
        2. Hyperedge-to-node attention
        3. Feed-forward network with residual connections and batch normalization

    Parameters
    ----------
    in_dim : int
        Dimension of input node features.
    out_dim : int
        Dimension of output node features per head.
    edge_dim : int
        Dimension of hyperedge features.
    n_heads : int
        Number of attention heads.
    feat_drop : float, default=0.
        Dropout rate applied to node features.
    attn_drop : float, default=0.
        Dropout rate applied to attention weights.
    residual : bool, default=True
        Whether to use residual connections.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    edge_mode : str, optional
        Type of edge feature interaction (unused in this implementation).
    chunked_size : int, default=10000
        Row chunk size for processing large incidence matrices to avoid memory limits.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 edge_dim,
                 n_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=True,
                 batch_norm=True,
                 edge_mode=None,
                 chunked_size=10000):
        super(SCAHGTLayer, self).__init__()

        # Initialize parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.residual = residual
        self.batch_norm = batch_norm
        self.edge_mode = edge_mode
        self.out_channel = out_dim * n_heads
        self.sqrt_dim = out_dim ** 0.5
        self.edge_dim = edge_dim
        self.chunk_size = chunked_size 

        # Node-to-hyperedge attention parameters
        self.n2h_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.n2h_k = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.n2h_v = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.n2h_o = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)
        self.edge_to_input = nn.Linear(edge_dim, in_dim, bias=False)
        
        # Hyperedge-to-node attention parameters
        self.h2n_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.h2n_k = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)
        self.h2n_v = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)
        self.h2n_o = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)
        
        # Output linear layer
        self.o_linear = nn.Linear(self.out_channel, self.out_channel, bias=False)
        
        # Feed-forward network
        self.FFN_h = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel * 4),
            nn.GELU(),
            nn.Linear(self.out_channel * 4, self.out_channel),
            nn.Dropout(feat_drop),
        )
        
        # Batch normalization layers
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.out_channel)
            self.batch_norm2 = nn.BatchNorm1d(self.out_channel)
        
        # Dropout layers
        self.feat_drop_layer = nn.Dropout(feat_drop)
        self.attn_drop_layer = nn.Dropout(attn_drop)
        
        # Residual connection
        if residual:
            if in_dim != self.out_channel:
                self.res_fc = nn.Linear(in_dim, self.out_channel, bias=False)
            else:
                self.res_fc = Identity()

    def get_nonzero_indices_chunked(self, H):
        """
        Get non-zero indices of a large tensor H in row-wise chunks to avoid memory overflow.

        Parameters
        ----------
        H : torch.Tensor [N, D]
            The incidence matrix.

        Returns
        -------
        torch.LongTensor of shape [num_nonzero, 2]
            Indices of non-zero elements in H.
        """
        nonzero_indices = []
        total_rows = H.shape[0]
        row_chunk_size = self.chunk_size

        for row_start in range(0, total_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, total_rows)
            chunk = H[row_start:row_end]
            chunk_sparse = chunk.to_sparse()
            chunk_idx = chunk_sparse.indices().t()
            chunk_idx[:, 0] += row_start
            nonzero_indices.append(chunk_idx)

        return torch.cat(nonzero_indices, dim=0)

    def node_to_hyperedge_attention(self, node_feats, hyperedge_feats, H, logger=None):
        """
        Node-to-hyperedge attention mechanism.

        Parameters
        ----------
        node_feats : torch.Tensor [num_nodes, in_dim]
        hyperedge_feats : torch.Tensor [num_hyperedges, edge_dim]
        H : torch.Tensor [num_nodes, num_hyperedges]

        Returns
        -------
        torch.Tensor [num_hyperedges, n_heads*out_dim]
        """
        device = node_feats.device
        num_nodes, num_hyperedges = H.shape

        edge_feats = self.edge_to_input(hyperedge_feats)
        Q = self.n2h_q(node_feats).view(num_nodes, self.n_heads, self.out_dim)
        K = self.n2h_k(edge_feats).view(num_hyperedges, self.n_heads, self.out_dim)
        V = self.n2h_v(node_feats).view(num_nodes, self.n_heads, self.out_dim)

        indices = self.get_nonzero_indices_chunked(H)
        hyperedge_ids = indices[:, 1]
        node_ids = indices[:, 0]

        Q_nodes = Q[node_ids]
        K_hyperedges = K[hyperedge_ids]
        attn_scores = (Q_nodes * K_hyperedges).sum(dim=-1) / self.sqrt_dim
        attn_weights = scatter_softmax(attn_scores, hyperedge_ids, dim=0)
        weighted_V = V[node_ids] * attn_weights.unsqueeze(-1)
        hyperedge_updates = scatter_add(weighted_V, hyperedge_ids, dim=0, dim_size=num_hyperedges)
        hyperedge_updates = hyperedge_updates.view(num_hyperedges, self.n_heads * self.out_dim)
        return self.n2h_o(hyperedge_updates)

    def hyperedge_to_node_attention(self, node_feats, hyperedge_feats, H, logger=None):
        """
        Hyperedge-to-node attention mechanism.

        Parameters
        ----------
        node_feats : torch.Tensor [num_nodes, in_dim]
        hyperedge_feats : torch.Tensor [num_hyperedges, n_heads*out_dim]
        H : torch.Tensor [num_nodes, num_hyperedges]

        Returns
        -------
        torch.Tensor [num_nodes, n_heads*out_dim]
        """
        device = node_feats.device
        num_nodes, num_hyperedges = H.shape

        Q = self.h2n_q(node_feats).view(num_nodes, self.n_heads, self.out_dim)
        K = self.h2n_k(hyperedge_feats).view(num_hyperedges, self.n_heads, self.out_dim)
        V = self.h2n_v(hyperedge_feats).view(num_hyperedges, self.n_heads, self.out_dim)

        indices = self.get_nonzero_indices_chunked(H)
        hyperedge_ids = indices[:, 1]
        node_ids = indices[:, 0]

        Q_nodes = Q[node_ids]
        K_hyperedges = K[hyperedge_ids]
        attn_scores = (Q_nodes * K_hyperedges).sum(dim=-1) / self.sqrt_dim
        attn_weights = scatter_softmax(attn_scores, node_ids, dim=0)
        weighted_V = V[hyperedge_ids] * attn_weights.unsqueeze(-1)
        node_updates = scatter_add(weighted_V, node_ids, dim=0, dim_size=num_nodes)
        node_updates = node_updates.view(num_nodes, self.n_heads * self.out_dim)
        return self.h2n_o(node_updates)

    def forward(self, graph, q, k, v, edge_feat, H, logger=None):
        """
        Forward pass for SCA-HGT Layer.

        Steps:
            1. Node-to-hyperedge attention
            2. Hyperedge-to-node attention
            3. Residual connection + batch normalization
            4. Feed-forward network + final normalization
        """
        device = q.device
        H = H.to(device)
        num_nodes = H.size(0)

        q_in1 = self.feat_drop_layer(q)
        k_src = self.feat_drop_layer(k)
        v_src = self.feat_drop_layer(v)

        # Node -> Hyperedge attention
        new_hyperedge_feats = self.node_to_hyperedge_attention(
            node_feats=k_src,
            hyperedge_feats=edge_feat,
            H=H,
            logger=logger
        )

        # Hyperedge -> Node attention
        node_messages = self.hyperedge_to_node_attention(
            node_feats=q_in1,
            hyperedge_feats=new_hyperedge_feats,
            H=H,
            logger=logger
        )

        # Residual connection
        if self.residual:
            resval = self.res_fc(q_in1)
            updated_nodes = node_messages + resval
        else:
            updated_nodes = node_messages

        # Batch normalization
        if self.batch_norm:
            updated_nodes = self.batch_norm1(updated_nodes)

        # Feed-forward network
        ffn_out = self.FFN_h(updated_nodes)

        # Residual connection
        if self.residual:
            ffn_out = updated_nodes + ffn_out

        # Final batch normalization
        if self.batch_norm:
            ffn_out = self.batch_norm2(ffn_out)

        return ffn_out
