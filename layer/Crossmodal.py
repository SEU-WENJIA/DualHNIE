
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn  as nn 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
import sys
import pickle as pk
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



'''
This module implements various cross-modal fusion strategies for combining structural and semantic features.

'''

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, struct_dim, semantic_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            kdim=struct_dim,
            vdim=struct_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # ensures input is [batch, seq, dim]
        )
        self.norm1 = nn.LayerNorm(semantic_dim)
        self.norm2 = nn.LayerNorm(semantic_dim)
        self.ffn = nn.Sequential(
            nn.Linear(semantic_dim, 4 * semantic_dim),
            nn.ReLU(),
            nn.Linear(4 * semantic_dim, semantic_dim),
            nn.Dropout(dropout)
        )

    def forward(self, semantic_feats, struct_feats):
        """
        semantic_feats: [batch_size, seq_len, semantic_dim]
        struct_feats:   [batch_size, seq_len, struct_dim]
        """
        # Structural features act as key/value, semantic features as query
        attn_output, _ = self.cross_attn(
            query=semantic_feats,
            key=struct_feats,
            value=struct_feats
        )
        # Residual + LayerNorm
        semantic_feats = self.norm1(semantic_feats + attn_output)
        # Feed-forward network with residual connection
        ffn_output = self.ffn(semantic_feats)
        return self.norm2(semantic_feats + ffn_output)



class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.3, mode='bidirectional'):
        """
        Cross-modal attention fusion module.
        
        Args:
            embed_dim: Dimension of input features (Z_s and Z_t must have the same dim).
            num_heads: Number of attention heads.
            dropout: Dropout rate for attention.
            mode: Fusion direction - 'struct2text', 'text2struct', or 'bidirectional'.
        """
        super(CrossModalAttentionFusion, self).__init__()
        self.mode = mode
        self.attn_st2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t2st = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, Z_s, Z_t):
        """
        Inputs:
            Z_s: [N, D] Structural features
            Z_t: [N, D] Textual features
        Returns:
            Z_fused: [N, D] Cross-modally fused features
        """
        if self.mode == 'struct2text':
            # Structure queries textual
            fused, _ = self.attn_st2t(query=Z_s, key=Z_t, value=Z_t)
            return fused

        elif self.mode == 'text2struct':
            # Text queries structure
            fused, _ = self.attn_t2st(query=Z_t, key=Z_s, value=Z_s)
            return fused

        elif self.mode == 'bidirectional':
            # Structure → Text
            out_st2t, _ = self.attn_st2t(query=Z_s, key=Z_t, value=Z_t)
            # Text → Structure
            out_t2st, _ = self.attn_t2st(query=Z_t, key=Z_s, value=Z_s)
            # Concatenate both outputs and project
            fused = self.proj(torch.cat([out_st2t, out_t2st], dim=-1))
            return fused

        else:
            raise ValueError("Unsupported Fusion Modes !")



class GateFusion(nn.Module):
    def __init__(self, struct_dim, semantic_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or struct_dim  # projection space
        self.linear_struct = nn.Linear(struct_dim, hidden_dim)
        self.linear_semantic = nn.Linear(semantic_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, struct_feats, semantic_feats):
        # project to same space
        struct_proj = self.linear_struct(struct_feats)
        semantic_proj = self.linear_semantic(semantic_feats)
        gate_values = self.sigmoid(struct_proj + semantic_proj)
        return gate_values * struct_proj + (1 - gate_values) * semantic_proj


class AttentionFusion(nn.Module):
    def __init__(self, struct_dim, semantic_dim):
        super().__init__()
        self.linear_struct = nn.Linear(struct_dim, 1)
        self.linear_semantic = nn.Linear(semantic_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, struct_feats, semantic_feats):
        scores = torch.cat([
            self.linear_struct(struct_feats),
            self.linear_semantic(semantic_feats)
        ], dim=-1)  # [batch, 2]
        attn_weights = self.softmax(scores)
        return attn_weights[:, 0:1] * struct_feats + attn_weights[:, 1:2] * semantic_feats


class ConcatFusion(nn.Module):
    def __init__(self, struct_dim, semantic_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(struct_dim + semantic_dim, output_dim)

    def forward(self, struct_feats, semantic_feats):
        concat_feats = torch.cat([struct_feats, semantic_feats], dim=-1)
        return F.relu(self.linear(concat_feats))


class FixedWeightFusion(nn.Module):
    def __init__(self, eta=0.5):
        super().__init__()
        self.alpha = eta

    def forward(self, struct_feats, semantic_feats):
        return self.alpha * struct_feats + (1 - self.alpha) * semantic_feats


class AdaptiveWeightedFusion(nn.Module):
    def __init__(self, eta=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(eta, dtype=torch.float))

    def forward(self, struct_feats, semantic_feats):
        return self.alpha * struct_feats + (1 - self.alpha) * semantic_feats


class StructureSemanticsFusion(nn.Module):
    def __init__(self, struct_dim, semantic_dim, output_dim, fusion_mode='concat', eta=0.2):
        super().__init__()
        if fusion_mode == 'concat':
            self.fusion_layer = ConcatFusion(struct_dim, semantic_dim, output_dim)
        elif fusion_mode == 'fixed':
            self.fusion_layer = FixedWeightFusion(eta)
        elif fusion_mode == 'adaptive':
            self.fusion_layer = AdaptiveWeightedFusion(eta)
        elif fusion_mode == 'attention':
            self.fusion_layer = AttentionFusion(struct_dim, semantic_dim)
        elif fusion_mode == 'gate':
            self.fusion_layer = GateFusion(struct_dim, semantic_dim)
        elif fusion_mode == 'crossmodal_attention':
            self.fusion_layer = CrossModalAttentionFusion(struct_dim, semantic_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_mode}")

    def forward(self, struct_feats, semantic_feats):
        return self.fusion_layer(struct_feats, semantic_feats)









 

