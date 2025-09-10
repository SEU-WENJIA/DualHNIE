import torch
import torch.nn as nn
import torch.nn.functional as F

class HGConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, bias=True, dropout=0.0):
        super(HGConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, X, H):
        """
        X: [N, F]  节点特征
        H: [N, E]  节点-超边关联矩阵 (稀疏或稠密都可以)
        """
        N, E = H.shape

        # 节点度和超边度
        Dv = torch.sum(H, dim=1)  # [N]
        De = torch.sum(H, dim=0)  # [E]

        # 归一化因子
        Dv_inv_sqrt = torch.pow(Dv, -0.5)
        Dv_inv_sqrt[torch.isinf(Dv_inv_sqrt)] = 0.0
        De_inv = torch.pow(De, -1.0)
        De_inv[torch.isinf(De_inv)] = 0.0

        # 卷积传播:  X' = Dv^-1/2 H De^-1 H^T Dv^-1/2 XW
        XW = self.fc(self.dropout(X))  # [N, out_feats]

        # 节点 -> 超边
        He = H * De_inv.unsqueeze(0)  # [N, E]
        X_e = torch.matmul(Dv_inv_sqrt.unsqueeze(1) * XW, He)  # [N, out_feats] -> aggregated to hyperedges

        # 超边 -> 节点
        Ht = H.t()
        X_v = torch.matmul(X_e, Ht)  # back to nodes
        X_v = Dv_inv_sqrt.unsqueeze(1) * X_v

        if self.activation is not None:
            X_v = self.activation(X_v)

        return X_v


class HGNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.5):
        super(HGNet, self).__init__()
        self.layer1 = HGConvLayer(in_feats, hidden_feats, activation=F.relu, dropout=dropout)
        self.layer2 = HGConvLayer(hidden_feats, out_feats, activation=None, dropout=dropout)

    def forward(self, X, H):
        """
        X: [N, F]  节点特征
        H: [N, E]  节点-超边关联矩阵
        """
        X = self.layer1(X, H)
        X = self.layer2(X, H)
        return X
