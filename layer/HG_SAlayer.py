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


class HGSA_layer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim, num_heads, feat_drop=0, attn_drop=0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False):
        super(HGSA_layer, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._edge_dim = edge_dim

        # Attention parameters initialization
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))  # Source node attention
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))  # Destination node attention
        self.attn_m = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_dim)))     # Edge attention

        # Dropout and activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_m, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, hypergraph, feat, edge_feat, H, logger=None):
        '''
        hypergraph: hypergraph 
        feat: nodes'features [node_num, attn_num]
        edge_feat: hyperedges' feature [node_num, feat_dim] need to convert before
        H: adjacency matrix of hypergraph [n, m]
        '''

        # Stage 1: Node to Hyperedge Attention
        with hypergraph.local_scope():
            # Ensure no zero in-degree nodes if not allowed
            if not self._allow_zero_in_degree:
                if (hypergraph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid.')
            device = edge_feat.device
            H = H.to(device)
            # Node features
            h_src = h_dst = feat  # [num_nodes, num_features_node]
            feat_src = feat_dst = h_src.view(-1, self._num_heads, self._out_feats)  # Reshape

            # Edge features
            # edge_feat = torch.mm(H.T,edge_feat)
            feat_e = edge_feat.unsqueeze(1)  # [num_edges, 1, edge_dim]

            # Compute attention scores for source nodes to edges
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # [num_nodes, num_heads, 1]

            # Save node features and attention scores to edges
            hypergraph.srcdata.update({'ft': feat_src, 'el': el})

            # Update edge features using source attention scores
            hypergraph.apply_edges(lambda edges: {
                'e': edges.src['el'] + (feat_e * self.attn_m).sum(dim=-1).unsqueeze(-1)
            })

            # Apply Leaky ReLU activation
            e = self.leaky_relu(hypergraph.edata.pop('e'))

            # Compute softmax for edge attention
            hypergraph.edata['a'] = self.attn_drop(edge_softmax(hypergraph, e))  # [num_edges, num_heads, 1]

            # Stage 2: Hyperedge to Node Attention
            hypergraph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = hypergraph.dstdata['ft']  # Updated node features
            H = H.to(rst.device)
            rst = torch.einsum('nm,mdk->ndk', H.float(), rst)
            # Residual connection if defined
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # Apply activation function if defined
            if self.activation:
                rst = self.activation(rst)  # [num_nodes, num_heads, 1]



            return rst.reshape(rst.size(0), -1)  # Flatten the output


class HGSALayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim, num_heads, feat_drop=0, 
                 attn_drop=0, negative_slope=0.2, residual=None, activation=None, 
                 allow_zero_in_degree=False):
        super(HGSALayer, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_dim = edge_dim
        self._allow_zero_in_degree = allow_zero_in_degree

        # Feature transformation parameters
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        
        # Attention parameters (优化为更高效的参数形式)
        self.attn_src = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.attn_edge = nn.Parameter(torch.Tensor(1, num_heads, edge_dim))
        
        # Dropout and activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # Residual connection
        if residual:
            if in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.res_fc = None
        
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Xavier初始化保证训练稳定性"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain)
        nn.init.xavier_uniform_(self.attn_src, gain)
        nn.init.xavier_uniform_(self.attn_edge, gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_uniform_(self.res_fc.weight, gain)


    def get_nonzero_indices_chunked(self, H, row_chunk_size=10000):
        """
        从大张量 H 中获取非零元素的索引（避免 INT_MAX 限制），按行分块处理。
        参数：
            H: torch.Tensor [N, D]
            row_chunk_size: 每次处理的行数（根据显存和效率调节）
        返回：
            torch.LongTensor of shape [num_nonzero, 2]
        """
        nonzero_indices = []
        total_rows = H.shape[0]

        for row_start in range(0, total_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, total_rows)
            chunk = H[row_start:row_end]  # shape: [chunk_rows, D]

            # 将 chunk 转为 sparse COO 表示，获取非零索引
            chunk_sparse = chunk.to_sparse()
            chunk_idx = chunk_sparse.indices().t()  # shape: [num_nonzero_in_chunk, 2]

            # 行索引加上偏移
            chunk_idx[:, 0] += row_start

            nonzero_indices.append(chunk_idx)

        # 合并所有索引
        all_indices = torch.cat(nonzero_indices, dim=0)
        return all_indices


    def forward(self, hypergraph, feat, edge_feat, H, logger=None):
        device = feat.device
        H = H.to(device)
        n_nodes, n_edges = H.shape
        
        # 特征变换
        h_src = self.feat_drop(feat)
        feat_src = self.fc(h_src).view(n_nodes, self._num_heads, self._out_feats)
        
        # 获取所有有效的节点-超边对 (修复关键错误)
        # H_sparse = H.to_sparse()
        # indices = H_sparse.indices().t() 
        # indices = H.nonzero(as_tuple=False)  # [E, 2] with (node_idx, edge_idx)

        indices = self.get_nonzero_indices_chunked(H, row_chunk_size=10000)
        if indices.size(0) == 0:
            return torch.zeros(n_nodes, self._num_heads * self._out_feats, device=device)
        
        # 准备注意力计算
        src_indices, edge_indices = indices[:, 0], indices[:, 1]
        
        # 计算注意力分数 (包含源节点和目标超边特征)
        src_feat = feat_src[src_indices]  # [E, h, d]
        edge_feat_expanded = edge_feat[edge_indices].unsqueeze(1)  # [E, 1, edge_dim]
        
        # 注意力分数计算: 源节点特征 + 边特征交互
        attn_score = (src_feat * self.attn_src).sum(dim=-1)  # [E, h]
        edge_contrib = (edge_feat_expanded * self.attn_edge).sum(dim=-1).squeeze(1)  # [E, h]
        e = attn_score + edge_contrib
        e = self.leaky_relu(e)  # [E, h]
        
        # 超边内注意力归一化 (使用高效分组softmax)
        max_val = torch.zeros(n_edges, self._num_heads, device=device).to(dtype=torch.bfloat16)
        max_val.scatter_reduce_(0, edge_indices.unsqueeze(1).expand(-1, self._num_heads), 
                                e, reduce="amax", include_self=False)
        
        e_exp = torch.exp(e - max_val[edge_indices])
        sum_exp = torch.zeros_like(max_val)
        sum_exp.scatter_add_(0, edge_indices.unsqueeze(1).expand(-1, self._num_heads), e_exp)
        
        attention = e_exp / (sum_exp[edge_indices] + 1e-9)  # [E, h]
        attention = self.attn_drop(attention)
        
        # 节点到超边聚合
        messages = src_feat * attention.unsqueeze(-1)  # [E, h, d]
        hyper_edge_feats = torch.zeros(n_edges, self._num_heads, self._out_feats, device=device).to(dtype=torch.bfloat16)
        hyper_edge_feats.index_add_(0, edge_indices, messages)
        
        # 超边到节点聚合 (矩阵乘法优化)
        rst = torch.einsum('ne,ehd->nhd', H, hyper_edge_feats)  # [n, h, d]
        
        # 残差连接
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(n_nodes, self._num_heads, self._out_feats)
            rst += resval
        
        # 激活函数
        if self.activation:
            rst = self.activation(rst)

        # mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        # logger.info("Hyperattention GPU: allocated {:.2f} MB".format(mem_allocated))
    
        return rst.view(n_nodes, -1)
    


class UniGSA_layer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_dim, num_heads, feat_drop=0, attn_drop=0, 
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False):
        super(UniGSA_layer, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._edge_dim = edge_dim

        # Attention parameters initialization
        self.attn_l = Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))  # Source node attention
        self.attn_r = Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))  # Destination node attention
        self.attn_m = Parameter(torch.FloatTensor(size=(1, num_heads, edge_dim)))    # Edge attention

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_m, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat, edge_feat, H):
        '''
        feat: nodes' features [node_num, in_feats]
        edge_feat: hyperedges' feature [edge_num, edge_dim]
        H: adjacency matrix of hypergraph [node_num, edge_num]
        '''
        device = edge_feat.device
        H = H.to(device)
        node_num, edge_num  = H.size() 

            # Pre-compute frequently used shapes
        nh = self._num_heads  # num_heads
        nf = self._out_feats  # out_feats
 
        edge_feat = torch.einsum('ij,ik->jk',H , edge_feat)        
        # Dropout for features
        feat = self.feat_drop(feat)
        edge_feat = self.feat_drop(edge_feat)
        
        # Reshape features
        h_src = h_dst = feat
        feat_src = feat_dst = h_src.view(node_num, self._num_heads, self._out_feats)
        
        # Stage 1: Node to Hyperedge Attention
        # Compute attention scores for source nodes to edges
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # [node_num, num_heads, 1]
        
        # Prepare edge features
        feat_e = edge_feat.unsqueeze(1)  # [edge_num, 1, edge_dim]
        em = (feat_e * self.attn_m).sum(dim=-1).unsqueeze(-1)  # [edge_num, num_heads, 1]       
        # Broadcast and combine attention scores
        # H: [node_num, edge_num]
        # el: [node_num, num_heads, 1]
        # em: [edge_num, num_heads, 1]
        # Expand dimensions for broadcasting
        H_expanded = H.unsqueeze(1)  # [node_num, 1, edge_num]
        # 
        el_expanded = el.unsqueeze(2)  # [node_num, num_heads, 1, 1]
        em_expanded = em.permute(1, 0, 2)   # [1, num_heads, edge_num, 1]
        
        # Compute attention scores
        e = el_expanded + em_expanded  # [node_num, num_heads, edge_num, 1]
        e = self.leaky_relu(e.squeeze(-1))  # [node_num, num_heads, edge_num]
        
        # Mask attention scores with hypergraph structure
        e = e * H_expanded.to(device)  # [node_num, num_heads, edge_num]
        
        # Compute softmax attention weights
        # First create a mask for zero entries in H
        mask = (H == 0).unsqueeze(1).to(device)  # [node_num, 1, edge_num]
        e_masked = e.masked_fill(mask, float('-inf'))
        a = F.softmax(e_masked, dim=-1)  # [node_num, num_heads, edge_num]
        a = self.attn_drop(a)
        
        # Stage 2: Hyperedge to Node Attention
        # Aggregate node features through hyperedges
        # feat_src: [node_num, num_heads, out_feats]
        # a: [node_num, num_heads, edge_num]
        
        # First compute weighted sum of node features for each hyperedge
        # H.T: [edge_num, node_num]
        # a: [node_num, num_heads, edge_num] -> permute to [edge_num, num_heads, node_num]
        # feat_src: [node_num, num_heads, out_feats] -> permute to [num_heads, node_num, out_feats]
        
        # Compute edge features: sum over nodes connected to each edge

        del  e_masked, em_expanded,el_expanded,  e, el, em
        weighted_features = feat_src.unsqueeze(2) * a.unsqueeze(3)
        
        # Now aggregate edge features back to nodes
        rst = torch.einsum('ne,nhef->nhf', H.float(), weighted_features)
        
        # Residual connection if defined
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval

        # Apply activation function if defined
        if self.activation:
            rst = self.activation(rst)

        torch.cuda.empty_cache()  # Explicitly clear CUDA cache if using GPU
        return rst.reshape(rst.size(0), -1)  # Flatten the output

