import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('../')

from layer.Crossmodal import structure_semantics_fusion
from layer.HGConvLayer import HypergraphConvLayer
from layer.HG_SAlayer import HGSALayer
from layer.SCAHGTLayer import  SCAHGTLayer

class DualHGNN_Two(nn.Module):
    def __init__(self,
                 g,
                 configs,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim_struct,
                 in_dim_semantic,
                 num_hidden,
                 heads,
                 H,
                 feat_drop,
                 attn_drop,
                 residual,
                 centrality,
                 scale,
                 batch_norm,
                 edge_mode,
                 negative_slope,
                 logger ,
                 ret_feat=False,
                 rel_emb=None ):
        
        super(DualHGNN_Two, self).__init__()
        self.g = g
        self.centrality = centrality.to(g.device)
        # self.centrality = H.sum(-1).to(g.device)
        self.scale = scale # True or False
        self.return_feat = ret_feat

        self.num_layers = num_layers
        self.activation = nn.ReLU()

        # self.hgat_layers = nn.ModuleList()
        # self.hgt_layers = nn.ModuleList()
        self.model = configs.model


        self.loss_fn = torch.nn.MSELoss()
        self.semantic_mode = configs.semantic_mode
        self.structure_mode = configs.structure_mode

        self.feat_drop = feat_drop
        self.logger  = logger 

        self.h_dim = configs.num_hidden #*heads[-2]


        if 'dualhgat' in configs.model:
            # Structural encoding uses the hypergraph attention mechanism to extract the structural features of the knowledge graph.

            self.scoring_nn_struct = nn.ModuleList()
            self.heads = heads 
            for _ in range(heads[0]):
                self.scoring_nn_struct.append(nn.Sequential(
                        nn.Dropout(feat_drop),
                        nn.Linear(in_dim_struct, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, in_dim_struct)
                ))


            self.hgat_layers_struct = nn.ModuleList()
            for l in range(0,num_layers):
                if l == 0:
                    self.hgat_layers_struct.append(HGSALayer(
                                in_dim_struct*heads[l], 
                                1, # num_hidden,
                                pred_dim,  
                                heads[l]*num_hidden,
                                feat_drop, 
                                attn_drop,
                                negative_slope,
                                residual,
                                self.activation                      
                    ))
                else:
                    self.hgat_layers_struct.append(HGSALayer(
                                heads[l-1] * num_hidden,
                                1, # num_hidden,
                                pred_dim,  
                                heads[l]*num_hidden,
                                feat_drop, 
                                attn_drop,
                                negative_slope,
                                residual,
                                self.activation                      
                    ))  
            

            # Semantic Information Encoding Learning
            self.scoring_nn_semantic = nn.ModuleList()
            self.heads = heads 
            for _ in range(heads[0]):
                self.scoring_nn_semantic.append(nn.Sequential(
                        nn.Dropout(feat_drop),
                        nn.Linear(in_dim_semantic, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, in_dim_semantic)
                ))

            self.hgat_layers_semantic = nn.ModuleList()
            for l in range(0,num_layers):
                if l == 0:
                    self.hgat_layers_semantic.append(HGSALayer(
                                in_dim_semantic*heads[l], 
                                1, # num_hidden,
                                pred_dim,  
                                heads[l]*num_hidden,
                                feat_drop, 
                                attn_drop,
                                negative_slope,
                                residual,
                                self.activation                      
                    ))
                else:
                    self.hgat_layers_semantic.append(HGSALayer(
                                heads[l-1] * num_hidden,
                                1, # num_hidden,
                                pred_dim,  
                                heads[l]*num_hidden,
                                feat_drop, 
                                attn_drop,
                                negative_slope,
                                residual,
                                self.activation                      
                    ))  
            
        elif 'dualhgt' in configs.model:

            self.hgt_layers_struct = nn.ModuleList()
            self.hgt_layers_struct.append(SCAHGTLayer(
                            in_dim_struct, 
                            num_hidden, 
                            pred_dim, 
                            heads[0],
                            feat_drop, 
                            attn_drop, 
                            residual, 
                            batch_norm, 
                            edge_mode,
                            chunked_size=configs.chunked_size
                ))
            
            # hidden layers    
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.hgt_layers_struct.append(SCAHGTLayer(
                            num_hidden * heads[l-1],
                            num_hidden, 
                            pred_dim, 
                            heads[l],
                            feat_drop, 
                            attn_drop, 
                            residual, 
                            batch_norm, 
                            edge_mode,
                            chunked_size=configs.chunked_size
                ))



            self.hgt_layers_semantic = nn.ModuleList()
            self.hgt_layers_semantic.append(SCAHGTLayer(
                            in_dim_semantic, 
                            num_hidden, 
                            pred_dim, 
                            heads[0],
                            feat_drop, 
                            attn_drop, 
                            residual, 
                            batch_norm, 
                            edge_mode,
                            chunked_size=configs.chunked_size
                ))
            
            # hidden layers    
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.hgt_layers_semantic.append(SCAHGTLayer(
                            num_hidden * heads[l-1],
                            num_hidden, 
                            pred_dim, 
                            heads[l],
                            feat_drop, 
                            attn_drop, 
                            residual, 
                            batch_norm, 
                            edge_mode,
                            chunked_size=configs.chunked_size
                ))

        elif 'dualhgcn' in configs.model:
            # Hypergraph Convolutional Network Initialization — Structural View
            self.hgc_layers_struct = nn.ModuleList()
            self.hgc_layers_struct.append(HypergraphConvLayer(
                in_dim=in_dim_struct,
                out_dim=num_hidden*heads[-2],
                edge_dim=pred_dim,       
                use_edge_feat=True,      
                feat_drop=feat_drop,
                residual=residual,
                batch_norm=batch_norm
            ))


            for l in range(1, num_layers):
                self.hgc_layers_struct.append(HypergraphConvLayer(
                    in_dim=num_hidden*heads[-2],      
                    out_dim=num_hidden*heads[-2],
                    edge_dim=pred_dim,
                    use_edge_feat=True,
                    feat_drop=feat_drop,
                    residual=residual,
                    batch_norm=batch_norm
                ))

            # Hypergraph Convolutional Network Initialization - Semantic View
            self.hgc_layers_semantic = nn.ModuleList()
            self.hgc_layers_semantic.append(HypergraphConvLayer(
                in_dim=in_dim_semantic,
                out_dim=num_hidden*heads[-2],
                edge_dim=pred_dim,
                use_edge_feat=True,
                feat_drop=feat_drop,
                residual=residual,
                batch_norm=batch_norm
            ))


            for l in range(1, num_layers):
                self.hgc_layers_semantic.append(HypergraphConvLayer(
                    in_dim=num_hidden*heads[-2],
                    out_dim=num_hidden*heads[-2],
                    edge_dim=pred_dim,
                    use_edge_feat=True,
                    feat_drop=feat_drop,
                    residual=residual,
                    batch_norm=batch_norm
                ))



                    

        self.struct_output_linear = nn.Linear(num_hidden * heads[-2], 1)  # self.h_dim  or   1 


        self.semantic_output_linear = nn.Linear(num_hidden * heads[-2], 1) # self.h_dim or  1



        #  z1 :  n , heads[-2],  z2 : n, num_hidden * heads[-2]
        self.contrastive_proj = nn.Sequential(
            nn.Linear(heads[-2]*self.h_dim, heads[-2]*self.h_dim),   
            nn.ReLU(),
            nn.Linear(heads[-2]*self.h_dim, heads[-2]*self.h_dim)
        )
        self.temperature = 0.5 

        self.contrastive_size = configs.contrastive_size
        self.fusion_mode = configs.fusion_mode
        self.fusion_layer = structure_semantics_fusion(
                                                     struct_dim=1,  
                                                     semantic_dim=1, 
                                                     output_dim=1, 
                                                     fusion_mode=self.fusion_mode, 
                                                     eta = configs.eta
                                                     )



        self.loss_alpha = configs.alpha


        self.loss_beta = configs.beta
        self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
        self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))


        nn.init.ones_(self.gamma)
        nn.init.ones_(self.beta)

        self.eta = configs.eta


        # relation embedding
        if rel_emb is None:
            self.rel_emb = nn.Embedding(rel_num, pred_dim)
        else:
            self.rel_emb = rel_emb

    def attention_encode(self, kg_feats_struct, kg_feats_semantic , edge_types, H):

        if self.model == 'dualhgat' :

            hyper_attention_struct = [score_nn(kg_feats_struct) for score_nn in self.scoring_nn_struct]
            hyper_attention_struct = torch.cat(hyper_attention_struct, dim=-1)

            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                residual = hyper_attention_struct
                hyper_attention_struct = self.hgat_layers_struct[l](self.g, hyper_attention_struct, edge_feats, H, self.logger)
                if l != (self.num_layers-1):
                    hyper_attention_struct = hyper_attention_struct.flatten(1).mean(-1, keepdim=True)           # [n_node, 1]
                    hyper_attention_struct = hyper_attention_struct.repeat(1, self.heads[l]*self.h_dim)

            # structure output projection
            latent_space_struct = hyper_attention_struct
            logits_struct = self.struct_output_linear(hyper_attention_struct)    #h_struct.flatten(1) # [n_nodes, n_heads]     

            hyper_attention_semantic = [score_nn(kg_feats_semantic) for score_nn in self.scoring_nn_semantic]
            hyper_attention_semantic = torch.cat(hyper_attention_semantic, dim=-1)

            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                residual = hyper_attention_semantic
                hyper_attention_struct = self.hgat_layers_semantic[l](self.g, hyper_attention_semantic, edge_feats, H, self.logger)
                if l != (self.num_layers-1):
                    hyper_attention_semantic = hyper_attention_semantic.flatten(1).mean(-1, keepdim=True)           # [n_node, 1]
                    hyper_attention_semantic = hyper_attention_semantic.repeat(1, self.heads[l]*self.h_dim)

            # structure output projection
            latent_space_semantic = hyper_attention_semantic
            logits_semantic = self.semantic_output_linear(hyper_attention_semantic)    #h_struct.flatten(1) # [n_nodes, n_heads]     

        elif self.model == 'dualhgt':

            kg_feats_struct = kg_feats_struct
            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                kg_feats_struct = self.hgt_layers_struct[l](
                    self.g, q=kg_feats_struct, 
                    k=kg_feats_struct, 
                    v=kg_feats_struct, 
                    edge_feat=edge_feats, 
                    H=H,
                    logger =self.logger )   
                
            # semantic output projection
            latent_space_struct = kg_feats_struct
            logits_struct = self.struct_output_linear(kg_feats_struct)                



            kg_feats_semantic = kg_feats_semantic
            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                kg_feats_semantic = self.hgt_layers_semantic[l](self.g, q=kg_feats_semantic, k=kg_feats_semantic, v=kg_feats_semantic, edge_feat=edge_feats, H=H,logger =self.logger )   
                
            # semantic output projection
            latent_space_semantic = kg_feats_semantic
            logits_semantic = self.semantic_output_linear(kg_feats_semantic)                


        elif self.model == 'dualhgcn':


            kg_feats_struct = kg_feats_struct
            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                kg_feats_struct = self.hgc_layers_struct[l](
                    node_feats=kg_feats_struct,
                    H=H,
                    edge_feats=edge_feats
                )

            latent_space_struct = kg_feats_struct
            logits_struct = self.struct_output_linear(kg_feats_struct)



            kg_feats_semantic = kg_feats_semantic
            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                kg_feats_semantic = self.hgc_layers_semantic[l](
                    node_feats=kg_feats_semantic,
                    H=H,
                    edge_feats=edge_feats
                )

            latent_space_semantic = kg_feats_semantic
            logits_semantic = self.semantic_output_linear(kg_feats_semantic)



        return  latent_space_struct, latent_space_semantic, logits_struct, logits_semantic
    



    def contrastive_loss(self, z1, z2, indices ):

        if indices.shape[0] > 20000:
            self.contrastive_size = 20000
        else:
            self.contrastive_size = indices.shape[0]
        
        z1_batch = z1[indices]  # [self.contrastive_size, H]
        z2_batch = z2[indices]  # [self.contrastive_size, H]
        
        z1_proj = self.contrastive_proj(z1_batch)  # [self.contrastive_size, H]
        z2_proj = self.contrastive_proj(z2_batch)  # [self.contrastive_size, H]
        
        z1_norm = F.normalize(z1_proj, p=2, dim=1)
        z2_norm = F.normalize(z2_proj, p=2, dim=1)
        
        sample_size = self.contrastive_size   
        sample_indices = torch.randperm(sample_size)[:sample_size]
        
        z1_sample = z1_norm[sample_indices]
        z2_sample = z2_norm[sample_indices]
    
        sim_matrix = torch.mm(z1_sample, z2_sample.t()) / self.temperature

        labels = torch.arange(sample_size).to(z1.device)  # self.contrastive_size -> sample_size
        
        loss = F.cross_entropy(sim_matrix, labels)
        loss += F.cross_entropy(sim_matrix.T, labels)
        
        return loss/2
 
 
    def feature_interaction(self, struct_feats, semantic_feats):

        combined = torch.cat([struct_feats, semantic_feats], dim=-1)

        input_dim = combined.shape[-1]
        attention_layer = nn.Linear(input_dim, 1).to(combined.device)
        attention_weights = torch.sigmoid(attention_layer(combined))
        h_struct = struct_feats * attention_weights
        h_semantic = semantic_feats * (1 - attention_weights)

        return h_struct, h_semantic
    

    def forward(self, g, struct_feats, semantic_feats,  edge_types , H, dataset,   labels=None,idx=None, training=False,feat_drop=0.2):


        if 'two' in dataset:

            latent_space_struct, latent_space_semantic, logits_struct, logits_semantic = \
                self.attention_encode(struct_feats,semantic_feats, edge_types, H)

            logits = self.fusion_layer(logits_struct, logits_semantic)
        

        if training:
            if 'two' in dataset:

                loss_struct = self.loss_fn(logits_struct[idx], labels[idx].unsqueeze(-1))
                loss_content = self.loss_fn(logits_semantic[idx], labels[idx].unsqueeze(-1))

                contrastive_loss = self.contrastive_loss(latent_space_struct, latent_space_semantic, idx)

                loss_all = self.loss_fn(logits[idx].float()  , labels[idx].unsqueeze(-1).float()  )   
                
                mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                

                loss =  loss_all + self.loss_alpha*contrastive_loss +  self.loss_beta*  (loss_struct +  loss_content) / 2 
            

            return logits, loss, mem_allocated
        

        else:

            return logits


