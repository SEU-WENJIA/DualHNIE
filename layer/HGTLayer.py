import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_max
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 edge_dim,
                 n_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 residual=True,
                 batch_norm=True,
                 edge_mode=None):
        
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.residual = residual
        self.batch_norm = batch_norm
        self.edge_mode = edge_mode # {None, 'MUL', 'ADD'}

        self.q_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.k_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.v_linear = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        if self.edge_mode == 'DOT':
            self.e_linear = nn.Linear(edge_dim, out_dim *   n_heads, bias=False)
        elif self.edge_mode == 'GATE':
            self.e_linear = nn.Sequential(
                nn.Linear(edge_dim, n_heads),
                nn.BatchNorm1d(n_heads),
                nn.Sigmoid()
            )
        else:
            self.e_linear = nn.Linear(edge_dim, n_heads)

        self.out_channel = out_dim * n_heads
        self.sqrt_dim = out_dim ** 0.5
        self.o_linear = nn.Linear(self.out_channel, self.out_channel, bias=False)

        self.FFN_h = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel*2),
            nn.ReLU(),
            nn.Linear(self.out_channel*2, self.out_channel),
            nn.Dropout(feat_drop),
        )

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.out_channel)
            self.batch_norm2 = nn.BatchNorm1d(self.out_channel)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        if residual:
            if self.in_dim != self.out_channel:
                self.res_fc = nn.Linear(self.in_dim, self.out_channel, bias=False)
            else:
                self.res_fc = Identity()

    def build_hyperedge_indices(self, H):

        hyperedge_indices = []
        for j in range(H.size(1)):
            nodes = torch.where(H[:, j] == 1)[0]
            hyperedge_indices.append(nodes)

        return hyperedge_indices


    def edge_softmax(self, edges_feat, H, num_nodes):

        device  = edges_feat.device
        num_hyperedges, max_nodes = edges_feat.shape[:2]
        index_matrix = torch.full((num_hyperedges, max_nodes), -1, device = device)

        for j in range(num_hyperedges):
            nodes = torch.where(H.T[j])[0]
            vaild_len = min(len(nodes), max_nodes)
            index_matrix[j, :vaild_len] = nodes[:vaild_len]
        
        vaild_mask = (index_matrix !=-1)
        
        # Flatten the edges_feat
        edges_flat = edges_feat[vaild_mask].view(-1, edges_feat.size(-1))  #[N_vaild, 4]
        scatter_index  = index_matrix[vaild_mask].flatten()     # [N_vaild]

        max_val = torch.zeros((num_nodes,  edges_feat.size(-1)), device = device)
        max_val.scatter_reduce_(
            dim = 0,
            index = scatter_index.unsqueeze(-1).expand(-1, edges_feat.size(-1)),
            src = edges_flat,
            reduce = 'amax',
            include_self = False
        )


        exp_edges = torch.exp(edges_feat - max_val[index_matrix])  
        
        sum_exp = torch.zeros_like(max_val)
        sum_exp.scatter_add_(
            dim = 0,
            index = scatter_index.unsqueeze(-1).expand(-1, edges_feat.size(-1)),
            src =  exp_edges[vaild_mask].view(-1,edges_feat.size(-1))
            
        )

        return exp_edges / (sum_exp[index_matrix] + 1e-10)


    def forward(self, graph, q, k, v, edge_feat, H):
        with graph.local_scope():
            #　q: dst feature;  k,v: src feature

            H = H.to(edge_feat.device) 
            q_in1 = q_dst = self.feat_drop(q)     
            k_src = self.feat_drop(k)
            v_src = self.feat_drop(v)

            # scale dot attention
            q_dst = self.q_linear(q_dst).reshape(-1, self.n_heads, self.out_dim)
            k_src = self.k_linear(k_src).reshape(-1, self.n_heads, self.out_dim)
            v_src = self.v_linear(v_src).reshape(-1, self.n_heads, self.out_dim)

            
            num_hyperedges = H.shape[1]
            num_nodes = H.shape[0]

            hyperedge_indices = self.build_hyperedge_indices(H) 
            # hyperedge_nodes = hyperedge_indices # [num_hyperedges, max_hyperedge_size]

            k_src_hyperedges = []
            q_dst_hyperedges = []
            for nodes in hyperedge_indices:
                k_src_hyperedges.append(k_src[nodes])  # [num_hyperedges, max_hyperedge_size, n_heads, out_dim]
                q_dst_hyperedges.append(q_dst[nodes])  # [num_hyperedges, max_hyperedge_size, n_heads, out_dim]

            k_src_hyperedges = nn.utils.rnn.pad_sequence(k_src_hyperedges, batch_first=True)
            q_dst_hyperedges = nn.utils.rnn.pad_sequence(q_dst_hyperedges, batch_first=True)

            attn_score = (q_dst_hyperedges * k_src_hyperedges).sum(dim=-1) / self.sqrt_dim  # [num_hyperedges, max_hyperedge_size, n_heads]

            hyperedge_features = torch.mm(H.T, edge_feat)

            if self.edge_mode == 'DOT':
                rel_feat = self.e_linear(hyperedge_features).reshape(-1, self.n_heads, self.out_dim)  # [num_hyperedges, n_heads, out_dim]
                rel_attn = (q_dst_hyperedges * rel_feat.unsqueeze(1)).sum(dim=-1) / self.sqrt_dim  # [num_hyperedges, max_hyperedge_size, n_heads]
                attn_score = attn_score + rel_attn
            elif self.edge_mode == 'MUL':
                edge_attn = self.e_linear(hyperedge_features)  # [num_hyperedges, n_heads]
                attn_score = attn_score * edge_attn.unsqueeze(1)
            elif self.edge_mode == 'ADD':
                edge_attn = self.e_linear(hyperedge_features)  # [num_hyperedges, n_heads]
                attn_score = attn_score + edge_attn.unsqueeze(1)
            elif self.edge_mode == 'GATE':
                edge_attn = self.e_linear(hyperedge_features)
                attn_score = attn_score * edge_attn.unsqueeze(1)
                graph.edata.update({'t': attn_score})


            # attn_score = self.attn_drop(F.softmax(attn_score, dim=1))
            attn_score = self.attn_drop(self.edge_softmax(attn_score, H, num_nodes=num_nodes))

            # 消息传递            
            v_src_hyperedges = []
            for nodes in hyperedge_indices:
                v_src_hyperedges.append(v_src[nodes])  

            v_src_hyperedges = nn.utils.rnn.pad_sequence(v_src_hyperedges,batch_first=True)  # [num_hyperedges, max_hyperedge_size, n_heads, out_dim]
            
            messages = torch.einsum('ijkl,ijk->ijkl',v_src_hyperedges,attn_score) # [num_hyperedges, max_hyperedge_size, n_heads, out_dim]

            aggregated_messages = torch.zeros_like(q_dst)  # [num_nodes, n_heads, out_dim]
            messages = messages.reshape(num_hyperedges, -1, self.n_heads * self.out_dim)
            aggregated_messages = torch.einsum('ij,jkl->ikl',H, messages)  #torch.mm(H, messages.reshape(num_hyperedges, -1)).reshape(num_nodes, self.n_heads, self.out_dim)

            
            rst = aggregated_messages.sum(axis=1).squeeze()            
            rst = self.o_linear(rst) # .flatten(1))

        
            # add-norm
            if self.residual:
                resval = self.res_fc(q_in1)
                rst = rst + resval

            if self.batch_norm:
                rst = self.batch_norm1(rst)

            # FFN
            q_in2 = rst
            rst = self.FFN_h(rst)

            # add-norm
            if self.residual:
                rst = rst + q_in2

            if self.batch_norm:
                rst = self.batch_norm2(rst)

            return rst


