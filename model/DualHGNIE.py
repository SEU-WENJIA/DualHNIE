import torch
import torch.nn as nn

from layer.HG_SAlayer import HGSALayer 
from layer.SCAHGTLayer import  SCAHGTLayer
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from layer.Crossmodal import structure_semantics_fusion

class DualHGNIE(nn.Module):
    def __init__(self,
                 g,
                 configs,
                 num_layers,
                 rel_num,
                 pred_dim,
                 in_dim_sturct,
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
        
        super(DualHGNIE, self).__init__()
        self.g = g
        self.centrality = centrality.to(g.device)

        self.scale = scale # True or False
        self.return_feat = ret_feat

        self.num_layers = num_layers
        self.activation = nn.ReLU()

        self.hgat_layers = nn.ModuleList()
        self.hgt_layers = nn.ModuleList()
        
        self.loss_fn = torch.nn.MSELoss()
        self.semantic_mode = configs.semantic_mode
        self.structure_mode = configs.structure_mode
        self.feat_drop = feat_drop
        self.logger  = logger 

        self.h_dim = configs.num_hidden #*heads[-2]



        if 'concat' in configs.dataset:
             
            self.scoring_nn = nn.ModuleList()
            self.heads = heads 
            for _ in range(heads[0]):
                self.scoring_nn.append(nn.Sequential(
                        nn.Dropout(feat_drop),
                        nn.Linear( in_dim_sturct +in_dim_semantic , num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden,  in_dim_sturct +in_dim_semantic )
                ))


            for l in range(0,num_layers):
                if l == 0:
                    self.hgat_layers.append(HGSALayer(
                                (in_dim_sturct +in_dim_semantic)*heads[l], 
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
                    self.hgat_layers.append(HGSALayer(
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
            

            # 文本编码模块：采用HyperGraph Transformer，提取上下文本之间关联
            # input projection (no residual)   
            self.hgt_layers.append(SCAHGTLayer(
                            in_dim_sturct +in_dim_semantic, 
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
                self.hgt_layers.append(SCAHGTLayer(
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

            # self.struct_output_linear = nn.Linear(heads[-2], 1)  # self.h_dim  or   1 
            self.struct_output_linear = nn.Linear(num_hidden*heads[-2], 1)

        else:

            # 结构编码采用，超图注意力机制，提取知识图谱的结构特征
            self.scoring_nn = nn.ModuleList()
            self.heads = heads 
            for _ in range(heads[0]):
                self.scoring_nn.append(nn.Sequential(
                        nn.Dropout(feat_drop),
                        nn.Linear(in_dim_sturct, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, in_dim_sturct)
                ))


            for l in range(0,num_layers):
                if l == 0:
                    self.hgat_layers.append(HGSALayer(
                                in_dim_sturct*heads[l], 
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
                    self.hgat_layers.append(HGSALayer(
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
            

            # 文本编码模块：采用HyperGraph Transformer，提取上下文本之间关联
            # input projection (no residual)   
            self.hgt_layers.append(SCAHGTLayer(
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
                self.hgt_layers.append(SCAHGTLayer(
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



        # 结构输出层
        # self.struct_output_linear = nn.Sequential(
        #     nn.Linear(heads[-2], num_hidden),
        #     nn.Sigmoid(),
        #     nn.Linear(num_hidden, 1)
        # )
            self.struct_output_linear = nn.Linear(num_hidden*heads[-2], 1)  # self.h_dim  or   1 


        # 语义输出层
        # self.semantic_output_linear  = nn.Sequential(
        #     nn.Linear(num_hidden * heads[-2], num_hidden),
        #     nn.Sigmoid(),
        #     nn.Linear(num_hidden, 1)
        # )        

        self.semantic_output_linear = nn.Linear(num_hidden * heads[-2], 1) # self.h_dim or  1

        # self.semantic_projection_layer = nn.Linear(num_hidden * heads[-2], heads[-2])


        #  z1 :  n , heads[-2],  z2 : n, num_hidden * heads[-2]
        self.contrastive_proj = nn.Sequential(
            nn.Linear(heads[-2]*self.h_dim, heads[-2]*self.h_dim),  # 投影头
            nn.ReLU(),
            nn.Linear(heads[-2]*self.h_dim, heads[-2]*self.h_dim)
        )
        self.temperature = 0.5  # 对比学习温度参数

        self.contrastive_size = configs.contrastive_size

        # self.gate_linear = nn.Linear(2 ,1)
        self.fusion_mode = configs.fusion_mode
        self.fusion_layer = structure_semantics_fusion(struct_dim=1,  semantic_dim=1, output_dim=1, fusion_mode=self.fusion_mode, eta = configs.eta)



        self.loss_alpha = configs.alpha

        # nn.Parameter(torch.FloatTensor(size=(1,)))
        self.loss_beta = configs.beta
        # nn.Parameter(torch.FloatTensor(size=(1,)))
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

    def attention_encode(self, kg_feats, edge_types, H, mode):
        '''
        后续消融实验准备：
        原结构采用hyper_attention编码方式
        原语义采用hyper_transformer编码方式编码
        '''
        if mode == 'hyper_attention' :

            hyper_attention = [score_nn(kg_feats) for score_nn in self.scoring_nn]
            hyper_attention = torch.cat(hyper_attention, dim=-1)

            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            #  聚合邻居信息——> 聚合超边信息
            for l in range(self.num_layers):
                residual = hyper_attention
                hyper_attention = self.hgat_layers[l](self.g, hyper_attention, edge_feats, H, self.logger)
                if l != (self.num_layers-1):
                    hyper_attention = hyper_attention.flatten(1).mean(-1, keepdim=True)           # [n_node, 1]
                    hyper_attention = hyper_attention.repeat(1, self.heads[l]*self.h_dim)

            # structure output projection
            latent_space = hyper_attention
            logits = self.struct_output_linear(hyper_attention)    #h_struct.flatten(1) # [n_nodes, n_heads]            

        elif mode == 'hyper_transformer':

            kg_feats = kg_feats
            edge_feats = self.rel_emb(edge_types)  # [rel_num, num_feature]

            for l in range(self.num_layers):
                kg_feats = self.hgt_layers[l](self.g, q=kg_feats, k=kg_feats, v=kg_feats, edge_feat=edge_feats, H=H,logger =self.logger )   
                
            # semantic output projection
            # latent_space = self.semantic_projection_layer(kg_feats)
            latent_space = kg_feats
            logits = self.semantic_output_linear(kg_feats)                       

        return  latent_space, logits
    

    def contrastive_loss(self, z1, z2, indices ):
        """
        计算结构特征和语义特征的对比损失
        z1: 结构特征 [N, H]
        z2: 语义特征 [N, H]
        indices: 训练节点索引 [self.contrastive_size]
        """
        if indices.shape[0] > 20000:
            self.contrastive_size = 20000
        else:
            self.contrastive_size = indices.shape[0]
        
        # 1. 提取训练节点特征
        z1_batch = z1[indices]  # [self.contrastive_size, H]
        z2_batch = z2[indices]  # [self.contrastive_size, H]
        
        # 2. 通过投影头
        z1_proj = self.contrastive_proj(z1_batch)  # [self.contrastive_size, H]
        z2_proj = self.contrastive_proj(z2_batch)  # [self.contrastive_size, H]
        
        # 3. 归一化特征向量
        z1_norm = F.normalize(z1_proj, p=2, dim=1)
        z2_norm = F.normalize(z2_proj, p=2, dim=1)
        
        # 4. 计算相似度矩阵

            # 4. 采样计算相似度矩阵
        sample_size = self.contrastive_size   # 根据 GPU 内存调整
        sample_indices = torch.randperm(sample_size)[:sample_size]
        
        z1_sample = z1_norm[sample_indices]
        z2_sample = z2_norm[sample_indices]
    
        sim_matrix = torch.mm(z1_sample, z2_sample.t()) / self.temperature

        
        # 5. 构建标签（对角线为正样本）
        labels = torch.arange(sample_size).to(z1.device)  # self.contrastive_size -> sample_size
        
        # 6. 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        loss += F.cross_entropy(sim_matrix.T, labels)
        
        return loss/2
 
 
    def feature_interaction(self, struct_feats, semantic_feats):
        # 使用注意力机制增强特征交互
        combined = torch.cat([struct_feats, semantic_feats], dim=-1)
        # 定义注意力层（线性层）
        input_dim = combined.shape[-1]
        attention_layer = nn.Linear(input_dim, 1).to(combined.device)
        attention_weights = torch.sigmoid(attention_layer(combined))
        h_struct = struct_feats * attention_weights
        h_semantic = semantic_feats * (1 - attention_weights)

        return h_struct, h_semantic
    

    def forward(self, g, struct_feats, semantic_feats,  edge_types , H, dataset,   labels=None,idx=None, training=False,feat_drop=0.2):

        '''
        1. 是否考虑融入新的信息来促进信息的流动和相互的影响
        2. 是否考虑对比学习来提高，
            2.1 结构嵌入向量  与  文本嵌入向量的之间距离  
        '''

        # if feat_drop > 0:
        #     struct_feats = F.dropout(struct_feats,  feat_drop, self.training)
        #     semantic_feats = F.dropout(semantic_feats,  feat_drop, self.training)

        '''
        这里为了体现双通道方法的优势，应当对比
        
        
        
        
        '''

        if 'two' in dataset:
            struct_space, struct_h = self.attention_encode(struct_feats, edge_types, H, self.structure_mode)
            semantic_space, semantic_h = self.attention_encode(semantic_feats, edge_types, H, self.semantic_mode)



            logit_struct =  struct_h  # self.output_layer1(struct_h)

            # if self.scale:
            #     logit_struct = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)

            logit_semantic = semantic_h   #self.output_layer2(semantic_h)

            logits = self.fusion_layer(logit_struct, logit_semantic)
        
        elif  'concat' in dataset:
            struct_semantic_feats = torch.concat([struct_feats, semantic_feats], dim=1)
            _,  struct_semantic_feats_h1 = self.attention_encode(struct_semantic_feats, edge_types, H, self.structure_mode)
            # struct_semantic_feats_h2 = self.attention_encode(struct_semantic_feats, edge_types, H, self.semantic_mode)
            logits = struct_semantic_feats_h1 

        elif 'semantic' in dataset:
            torch.cuda.empty_cache()

            _ ,  semantic_h = self.attention_encode(semantic_feats, edge_types, H, self.semantic_mode)
            logits = semantic_h

        else:
            _, struct_h_1 = self.attention_encode(struct_feats, edge_types, H, self.structure_mode)

            # _, struct_h_2 = self.attention_encode(struct_feats, edge_types, H, self.structure_mode)
            
            logit_struct =  struct_h_1  # self.output_layer1(struct_h)

            # if self.scale:
            #     logit_struct = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)
            
            logits = logit_struct

        
        if training:
            if 'two' in dataset:

                loss_struct = self.loss_fn(logit_struct[idx], labels[idx].unsqueeze(-1))
                loss_content = self.loss_fn(logit_semantic[idx], labels[idx].unsqueeze(-1))

                contrastive_loss = self.contrastive_loss(struct_space, semantic_space, idx)

                loss_all = self.loss_fn(logits[idx].float()  , labels[idx].unsqueeze(-1).float()  )   
                
                mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                

                # loss = self.loss_weighting([loss_all, loss_struct, loss_content, contrastive_loss])
        
                # loss = loss_all + 0.2*  (loss_struct +  loss_content) / 2 
                # loss = abs(self.beta) /abs(self.gamma+self.beta) * loss_struct + abs(self.gamma) /abs(self.gamma+self.beta) * loss_content + 0.1 * contrastive_loss + 0.2*  (loss_struct +  loss_content) / 2 
                
                # loss = loss_all + 0.1 * contrastive_loss
                loss =  loss_all + self.loss_alpha*contrastive_loss +  self.loss_beta*  (loss_struct +  loss_content) / 2 
            
            else:
                # loss_struct = self.loss_fn(logit_struct[idx], labels[idx].unsqueeze(-1))
                # loss_content = self.loss_fn(logit_content[idx], labels[idx].unsqueeze(-1))

                loss_all = self.loss_fn(logits[idx].float()  , labels[idx].unsqueeze(-1).float()  )                   
                
                loss = loss_all  

                mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2

            return logits, loss, mem_allocated
        

        else:

            return logits


