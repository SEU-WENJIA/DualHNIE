import torch
import torch.nn as nn
from layer.SA_Layer import SALayer

class DualGAT(nn.Module):

    def __init__(self,  g, args, num_layers, rel_num, pred_dim, in_dim_struct, in_dim_semantic,
                 num_hidden, heads, activation, feat_drop, attn_drop,
                 negative_slope, residual, centrality, scale):
        
        super(DualGAT, self).__init__()
        self.g = g
        self.centrality = centrality
        self.scale = scale # True or False

        self.num_layers = num_layers


        self.sa_layers_struct = nn.ModuleList()
        self.sa_layers_semantic = nn.ModuleList()


        self.activation = activation
        self.dataset = args.dataset
        self.alpha = args.fusion
        
        # Scoring Network
        self.scoring_nn_struct = nn.ModuleList()
        self.scoring_nn_semantic = nn.ModuleList()

        self.heads = heads
        for _ in range(heads[0]):  
            self.scoring_nn_struct.append(nn.Sequential(
                nn.Dropout(feat_drop),
                nn.Linear(in_dim_struct, int(0.5*in_dim_struct)),
                nn.ReLU(),
                nn.Linear(int(0.5*in_dim_struct), 1)))
            

            self.scoring_nn_semantic.append(nn.Sequential(
                nn.Dropout(feat_drop),
                nn.Linear(in_dim_semantic, int(0.5*in_dim_semantic)),
                nn.ReLU(),
                nn.Linear(int(0.5*in_dim_semantic), 1)))
            

        # Relation Embedding
        self.rel_emb = nn.Embedding(rel_num, pred_dim)

        # hidden layers
        for l in range(0, num_layers):
            # due to multi-head, the in_dim_struct = num_hidden * num_heads
            self.sa_layers_struct.append(SALayer(
                1, 1, pred_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            
            self.sa_layers_semantic.append(SALayer(
                1, 1, pred_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))


        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1, heads[-2])))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)





    def forward(self, feats_struct, feats_semantic,  edge_types):


        
        if 'two' in self.dataset:

            h_struct = [score_nn(feats_struct) for score_nn in self.scoring_nn_struct]
            h_struct  = torch.cat(h_struct , dim=-1)

            edge_feats = self.rel_emb(edge_types)

            for l in range(self.num_layers):
                h_struct = self.sa_layers_struct[l](self.g, h_struct, edge_feats)
                if l != (self.num_layers-1):
                    h_struct = h_struct.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                    h_struct = h_struct.repeat(1, self.heads[l])

            # output scale
            logits_struct = h_struct.flatten(1) # [n_nodes, n_heads]



            h_semantic = [score_nn(feats_semantic) for score_nn in self.scoring_nn_semantic]
            h_semantic  = torch.cat(h_semantic , dim=-1)

            edge_feats = self.rel_emb(edge_types)

            for l in range(self.num_layers):
                h_semantic = self.sa_layers_semantic[l](self.g, h_semantic, edge_feats)
                if l != (self.num_layers-1):
                    h_semantic = h_semantic.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                    h_semantic = h_semantic.repeat(1, self.heads[l])

            # output scale
            logits_semantic = h_semantic.flatten(1) # [n_nodes, n_heads]


            logits = self.alpha *logits_struct + (1 - self.alpha) * logits_semantic


        elif 'semantic' in self.dataset:
            h_semantic = [score_nn(feats_semantic) for score_nn in self.scoring_nn_semantic]
            h_semantic  = torch.cat(h_semantic , dim=-1)

            edge_feats = self.rel_emb(edge_types)

            for l in range(self.num_layers):
                h_semantic = self.sa_layers_semantic[l](self.g, h_semantic, edge_feats)
                if l != (self.num_layers-1):
                    h_semantic = h_semantic.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                    h_semantic = h_semantic.repeat(1, self.heads[l])

            # output scale
            logits_semantic = h_semantic.flatten(1) # [n_nodes, n_heads]
            logits = logits_semantic



        elif 'struct' in self.dataset:
            h_struct = [score_nn(feats_struct) for score_nn in self.scoring_nn_struct]
            h_struct  = torch.cat(h_struct , dim=-1)

            edge_feats = self.rel_emb(edge_types)

            for l in range(self.num_layers):
                h_struct = self.sa_layers_struct[l](self.g, h_struct, edge_feats)
                if l != (self.num_layers-1):
                    h_struct = h_struct.flatten(1).mean(-1, keepdim=True) # [n_node, 1]
                    h_struct = h_struct.repeat(1, self.heads[l])

            # output scale
            logits_struct = h_struct.flatten(1) # [n_nodes, n_heads]

            logits = logits_struct

        if self.scale:     
            logits = nn.functional.leaky_relu(((self.centrality.unsqueeze(-1) * self.gamma + self.beta) * logits)\
                                        .mean(-1, keepdim=True))
        else:
            logits = logits.mean(-1, keepdim=True)



        return logits