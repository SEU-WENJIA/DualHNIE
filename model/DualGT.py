import torch
import torch.nn as nn
from graph_transformer import GTLayer
import dgl
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class DualGT(nn.Module):
    def __init__(self,
                 g,
                 args,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim_struct,
                 in_dim_semantic,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,   # True
                 centrality,  #  get_centrality(g,H) 
                 scale,   # True
                 batch_norm,  # True
                 edge_mode,   #  'MUL'
                 ret_feat=False,
                 rel_emb=None):
        

        
        super(DualGT, self).__init__()
        self.g = g
        self.centrality = centrality
        self.scale = scale # True or False
        self.return_feat = ret_feat
        self.dataset = args.dataset
        self.alpha = args.fusion

        self.num_layers = num_layers
        self.gat_layers_struct = nn.ModuleList()
        self.gat_layers_semantic = nn.ModuleList()



        # input projection (no residual)
        self.gat_layers_struct.append(GTLayer(
            in_dim_struct, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        
        self.gat_layers_semantic.append(GTLayer(
            in_dim_semantic, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))


        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim_struct = num_hidden * num_heads
            self.gat_layers_struct.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))
            

            self.gat_layers_semantic.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))           
            


        self.output_linear_struct = nn.Linear(num_hidden * heads[-2], 1)
        self.output_linear_semantic = nn.Linear(num_hidden * heads[-2], 1)



        if self.scale:
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb



    def forward(self, feats_struct, feats_semantic, edge_types):

        h_struct = feats_struct
        h_semantic = feats_semantic


        edge_feats = self.rel_emb(edge_types)

        if 'two' in self.dataset:

            for l in range(self.num_layers):

                h_struct = self.gat_layers_struct[l](self.g, q=h_struct, k=h_struct, v=h_struct, edge_feat=edge_feats)   
                h_semantic = self.gat_layers_semantic[l](self.g, q= h_semantic, k= h_semantic, v= h_semantic, edge_feat=edge_feats)   

            logits_struct = self.output_linear_struct(h_struct)   

            logits_semantic = self.output_linear_struct(h_semantic)   

            logits = self.alpha *logits_struct + (1 - self.alpha) * logits_semantic

        elif 'semantic' in self.dataset:
            for l in range(self.num_layers):

                h_semantic = self.gat_layers_semantic[l](self.g, q= h_semantic, k= h_semantic, v= h_semantic, edge_feat=edge_feats)   



            logits = self.output_linear_struct(h_semantic)        


        elif 'struct' in self.dataset:
            for l in range(self.num_layers):

                h_struct = self.gat_layers_struct[l](self.g, q=h_struct, k=h_struct, v=h_struct, edge_feat=edge_feats)   
                
            logits = self.output_linear_struct(h_struct)   
        
        # flexible adjustents
        if self.scale:
            logits = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logits)


        
        return logits



class DualGT_feat(nn.Module):
    def __init__(self,
                 args,
                 g,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim_struct,
                 in_dim_semantic,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 batch_norm,
                 edge_mode,
                 rel_emb=None):
        

        super(DualGT_feat, self).__init__()
        self.g = g

        self.num_layers = num_layers
        self.dataset = args.dataset
        self.fusion = args.fusion




        self.gat_layers_struct = nn.ModuleList()
        self.gat_layers_semantic = nn.ModuleList()



        # input projection (no residual)
        self.gat_layers_struct.append(GTLayer(
            in_dim_struct, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        

        self.gat_layers_semantic.append(GTLayer(
            in_dim_semantic, num_hidden, pred_dim, heads[0],
            feat_drop, attn_drop, residual, batch_norm, edge_mode))
        


        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim_struct = num_hidden * num_heads
            self.gat_layers_struct.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))
            
            self.gat_layers_semantic.append(GTLayer(
                num_hidden * heads[l-1], num_hidden, pred_dim, heads[l],
                feat_drop, attn_drop, residual, batch_norm, edge_mode))



        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb





    def forward(self, feats_struct, feats_semantic  , edge_types):



        h_struct = feats_struct
        h_semantic = feats_semantic
        edge_feats = self.rel_emb(edge_types)

        if 'two' in self.dataset:

            for l in range(self.num_layers):

                h_struct = self.gat_layers_struct[l](self.g, q=h_struct, k=h_struct, v=h_struct, edge_feat=edge_feats)   
                h_semantic = self.gat_layers_semantic[l](self.g, q= h_semantic, k= h_semantic, v= h_semantic, edge_feat=edge_feats)   

            logits_struct = self.output_linear_struct(h_struct)   

            logits_semantic = self.output_linear_struct(h_semantic)   

            logits = self.alpha *logits_struct + (1 - self.alpha) * logits_semantic

        elif 'semantic' in self.dataset:
            for l in range(self.num_layers):

                h_semantic = self.gat_layers_semantic[l](self.g, q= h_semantic, k= h_semantic, v= h_semantic, edge_feat=edge_feats)   

            logits = h_semantic 
        

        elif 'struct' in self.dataset:
            for l in range(self.num_layers):

                h_struct = self.gat_layers_struct[l](self.g, q=h_struct, k=h_struct, v=h_struct, edge_feat=edge_feats)   
                
            logits = h_struct        

        return logits


