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
                 chunked_size=10000
                 ):
        super(SCAHGTLayer, self).__init__()
        # 参数初始化
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

        # 节点到超边的注意力参数
        self.n2h_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)  # 用于节点特征（查询）
        self.n2h_k = nn.Linear(in_dim, out_dim * n_heads, bias=False)  # 用于超边特征（键）
        self.n2h_v = nn.Linear(in_dim, out_dim * n_heads, bias=False)  # 用于节点特征（值）
        self.n2h_o = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)  # 输出变换

        self.edge_to_input = nn.Linear(edge_dim, in_dim, bias=False)
        
        # 超边到节点的注意力参数
        self.h2n_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)  # 用于节点特征（查询）
        self.h2n_k = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)  # 用于超边特征（键）
        self.h2n_v = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)  # 用于超边特征（值）
        self.h2n_o = nn.Linear(out_dim * n_heads, out_dim * n_heads, bias=False)  # 输出变换
        
        # 输出层
        self.o_linear = nn.Linear(self.out_channel, self.out_channel, bias=False)
        
        # 前馈网络
        self.FFN_h = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel * 4),
            nn.GELU(),
            nn.Linear(self.out_channel * 4, self.out_channel),
            nn.Dropout(feat_drop),
        )
        
        # 归一化层
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.out_channel)
            self.batch_norm2 = nn.BatchNorm1d(self.out_channel)
        
        # Dropout层
        self.feat_drop_layer = nn.Dropout(feat_drop)
        self.attn_drop_layer = nn.Dropout(attn_drop)
        
        # 残差连接
        if residual:
            if in_dim != self.out_channel:
                self.res_fc = nn.Linear(in_dim, self.out_channel, bias=False)
            else:
                self.res_fc = Identity()


    def build_hyperedge_indices(self, H):
        """优化后的超边索引构建，使用稀疏矩阵操作"""
        device = H.device
        num_hyperedges = H.size(1)
        
        # 转换为稀疏矩阵格式
        H_sparse = H.to_sparse_coo()
        hyperedge_ids, node_ids = H_sparse.indices()[1], H_sparse.indices()[0]
        
        # 排序并按超边分组
        sorted_idx = torch.argsort(hyperedge_ids)
        hyperedge_ids = hyperedge_ids[sorted_idx]
        node_ids = node_ids[sorted_idx]
        
        # 获取分割点
        _, counts = torch.unique_consecutive(hyperedge_ids, return_counts=True)
        split_indices = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)])
        
        # 生成超边索引和掩码
        hyperedge_indices = [node_ids[split_indices[i]:split_indices[i+1]] for i in range(num_hyperedges)]
        max_size = max(len(nodes) for nodes in hyperedge_indices)
        
        # 创建填充索引和掩码
        padded_indices = torch.full((num_hyperedges, max_size), -1, device=device, dtype=torch.long)
        mask = torch.zeros((num_hyperedges, max_size), device=device, dtype=torch.bool)
        for i, nodes in enumerate(hyperedge_indices):
            valid_len = len(nodes)
            padded_indices[i, :valid_len] = nodes
            mask[i, :valid_len] = True




        return padded_indices, mask

    def get_nonzero_indices_chunked(self, H ):
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
        row_chunk_size = self.chunk_size


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

    def node_to_hyperedge_attention(self, node_feats, hyperedge_feats, H, logger=None):
        """
        input : node_feats  [num_nodes, in_dim]
                hyperedge_feats [num_hyperedges, in_dim]
                H [num_nodes, num_hyperedges]


        节点到超边的注意力机制
        """
        device = node_feats.device
        num_nodes, num_hyperedges = H.shape

        edge_feats = self.edge_to_input(hyperedge_feats)
        
        # 线性变换
        Q = self.n2h_q(node_feats).view(num_nodes, self.n_heads, self.out_dim)
        K = self.n2h_k(edge_feats).view(num_hyperedges, self.n_heads, self.out_dim)  # hyperedge_feats -> edge_feats
        V = self.n2h_v(node_feats).view(num_nodes, self.n_heads, self.out_dim)
        
        # 获取超边包含的节点索引
        # H_sparse = H.to_sparse_coo()
        indices = self.get_nonzero_indices_chunked(H)
        hyperedge_ids = indices[:,1]
        node_ids = indices[:,0]
        
        # 检查 hyperedge_ids 是否超出范围
        if torch.any(hyperedge_ids >= num_hyperedges):
            raise ValueError(f"hyperedge_ids contains values out of range [0, {num_hyperedges}). Found max value: {hyperedge_ids.max()}")

        # 为每个(节点, 超边)对计算注意力分数
        Q_nodes = Q[node_ids]  # [num_edges, H, D]
        K_hyperedges = K[hyperedge_ids]  # [num_edges, H, D]
        
        # 计算注意力分数
        attn_scores = (Q_nodes * K_hyperedges).sum(dim=-1) / self.sqrt_dim  # [num_edges, H]
        
        # 在超边内进行softmax归一化
        attn_weights = scatter_softmax(attn_scores, hyperedge_ids, dim=0)  # [num_edges, H]
        
        # 加权聚合
        V_nodes = V[node_ids]  # [num_edges, H, D]
        weighted_V = V_nodes * attn_weights.unsqueeze(-1)  # [num_edges, H, D]
        
        # 聚合到超边
        hyperedge_updates = scatter_add(weighted_V, hyperedge_ids, dim=0, dim_size=num_hyperedges)  # [num_hyperedges, H, D]
        
        # 合并多头
        hyperedge_updates = hyperedge_updates.view(num_hyperedges, self.n_heads * self.out_dim)
        hyperedge_updates = self.n2h_o(hyperedge_updates)  # [num_hyperedges, out_dim]
        
        # mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        # logger.info("node to hyperedge GPU: allocated {:.2f} MB".format(mem_allocated))
                        
        return hyperedge_updates

    def hyperedge_to_node_attention(self, node_feats, hyperedge_feats, H, logger=None):
        """
        超边到节点的注意力机制
        """
        device = node_feats.device
        num_nodes, num_hyperedges = H.shape
        
        # 线性变换
        Q = self.h2n_q(node_feats).view(num_nodes, self.n_heads, self.out_dim)
        K = self.h2n_k(hyperedge_feats).view(num_hyperedges, self.n_heads, self.out_dim)
        V = self.h2n_v(hyperedge_feats).view(num_hyperedges, self.n_heads, self.out_dim)
        
        # 获取节点所属的超边索引
        # H_sparse = H.to_sparse_coo()
        indices = self.get_nonzero_indices_chunked(H )
        hyperedge_ids = indices[:,1]
        node_ids = indices[:,0]
        
        # 为每个(超边, 节点)对计算注意力分数
        Q_nodes = Q[node_ids]  # [num_edges, H, D] - 注意：这里的node_ids对应每个连接
        K_hyperedges = K[hyperedge_ids]  # [num_edges, H, D]
        
        # 计算注意力分数
        attn_scores = (Q_nodes * K_hyperedges).sum(dim=-1) / self.sqrt_dim  # [num_edges, H]
        
        # 在节点所属超边集合内进行softmax归一化
        attn_weights = scatter_softmax(attn_scores, node_ids, dim=0)  # [num_edges, H]
        
        # 加权聚合
        V_hyperedges = V[hyperedge_ids]  # [num_edges, H, D]
        weighted_V = V_hyperedges * attn_weights.unsqueeze(-1)  # [num_edges, H, D]
        
        # 聚合到节点
        node_updates = scatter_add(weighted_V, node_ids, dim=0, dim_size=num_nodes)  # [num_nodes, H, D]
        
        # 合并多头
        node_updates = node_updates.view(num_nodes, self.n_heads * self.out_dim)
        node_updates = self.h2n_o(node_updates)  # [num_nodes, out_dim]
        

        # mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        # logger.info("Hyperedge to node GPU: allocated {:.2f} MB".format(mem_allocated))

        return node_updates

    def forward(self, graph, q, k, v, edge_feat, H, logger ):
        """
        实现多层关系图转换器机制：
        1. 节点到超边注意力
        2. 超边到节点注意力
        """
        device = q.device
        H = H.to(device)
        num_nodes = H.size(0)
        
        # 特征变换
        q_in1 = self.feat_drop_layer(q)
        k_src = self.feat_drop_layer(k)
        v_src = self.feat_drop_layer(v)
                # === 节点到超边注意力 ===
        # 使用节点特征更新超边特征
        new_hyperedge_feats = self.node_to_hyperedge_attention(
            node_feats=k_src, 
            hyperedge_feats=edge_feat,
            H=H,
            logger=logger
        )
        
        # === 超边到节点注意力 ===
        # 使用更新后的超边特征更新节点特征
        node_messages = self.hyperedge_to_node_attention(
            node_feats=q_in1,
            hyperedge_feats=new_hyperedge_feats,
            H=H,
            logger=logger
        )

    

        # 节点更新
        if self.residual:
            resval = self.res_fc(q_in1)
            updated_nodes = node_messages + resval
        else:
            updated_nodes = node_messages
        
        # 归一化
        if self.batch_norm:
            updated_nodes = self.batch_norm1(updated_nodes)
        
        # FFN部分
        ffn_out = self.FFN_h(updated_nodes)
        
        # 残差连接
        if self.residual:
            ffn_out = updated_nodes + ffn_out
        
        # 最终归一化
        if self.batch_norm:
            ffn_out = self.batch_norm2(ffn_out)
        
        return ffn_out



# import torch
# import dgl

# # 示例输入数据
# num_nodes = 10  # 节点数量
# num_hyperedges = 5  # 超边数量
# in_dim = 64  # 输入特征维度
# out_dim = 128  # 输出特征维度
# n_heads = 4  # 多头注意力头数

# # 创建一个随机的DGL图（示例图结构）
# graph = dgl.graph((torch.randint(0, num_nodes, (num_nodes,)), torch.randint(0, num_nodes, (num_nodes,))))

# # 创建节点特征和超边特征（随机初始化）
# node_feats = torch.randn(num_nodes, in_dim)
# hyperedge_feats = torch.randn(num_hyperedges, in_dim)

# # 创建关联矩阵 H（稀疏矩阵）
# # 确保 hyperedge_ids 的值不会超出范围
# indices = torch.randint(0, num_nodes, (2, num_hyperedges * 2))  # 生成更多的边
# indices[1, :] = torch.randint(0, num_hyperedges, (num_hyperedges * 2,))  # 确保超边索引不会超出范围
# values = torch.ones(indices.size(1))  # 值为1
# H = torch.sparse_coo_tensor(
#     indices=indices,
#     values=values,
#     size=(num_nodes, num_hyperedges)
# ).coalesce()  # 确保 H 是 COO 格式

# # 验证 num_hyperedges 的值
# assert H.size(1) == num_hyperedges, f"num_hyperedges ({num_hyperedges}) does not match the number of columns in H ({H.size(1)})"

# # 初始化 GTLayer
# gt_layer = GTLayer(
#     in_dim=in_dim,
#     out_dim=out_dim,
#     edge_dim=in_dim,  # 超边特征维度与输入特征维度相同
#     n_heads=n_heads,
#     feat_drop=0.1,
#     attn_drop=0.1,
#     residual=True,
#     batch_norm=True,
#     edge_mode=None
# )

# # 前向传播
# with torch.no_grad():  # 示例中不进行梯度计算
#     output_node_feats = gt_layer(graph, node_feats, node_feats, node_feats, hyperedge_feats, H)

# print("输出节点特征的形状:", output_node_feats.shape)