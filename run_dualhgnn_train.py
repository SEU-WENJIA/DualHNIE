import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn  as nn 
import os
import sys
import pickle as pk
import logging
from datetime import datetime
class RealTimeFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from model.DualHGNN import DualHGNN_Two
from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed,  get_rank_metrics, rank_evaluate, get_hypergraph_centrality
from utils.metric import overlap
from utils.load_data_hypergraph import load_data


def main(args,logger):

     
    logger.info(f"Starting training with arguments: {args}")
    # set_random_seed(0)
    ndcg_scores = []
    spearmans = []
    overlaps = []
    rmses = []
    medAEs = []

    ndcg_scores_20 = []
    ndcg_scores_50 = []
    ndcg_scores_200 = []
    ndcg_scores_test = []

    overlaps_20 = []
    overlaps_50 = []
    overlaps_200 = []
    overlaps_test = []


    spearmans_20 = []
    spearmans_50 = []
    spearmans_200 = []
    spearmans_test = []



    rmses_20 = []
    medAEs_20 = []
    rmses_50 = []
    medAEs_50 = []
    rmses_200 = []
    medAEs_200 = []
    rmses_test = []
    medAEs_test = [] 

    # set the save path
    save_root = 'results/' + args.dataset + '_GTRAN-REL/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):
        g,  edge_types, H, hyperedges, edges,  rel_num, struct_feats, semantic_feats, labels, train_idx, val_idx, test_idx = load_data(args.data_path,  cross_id, args.dataset)


        struct_feats = torch.tensor(struct_feats).cuda().to(torch.bfloat16)
        semantic_feats = torch.tensor(semantic_feats).cuda().to(torch.bfloat16)
        labels = torch.tensor(labels).cuda().to(torch.bfloat16)
        H = H.to(torch.bfloat16)

        # g = data[0]
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True

        g = g.int() 

        num_struct_feats = struct_feats.shape[1]
        num_semantic_feats = semantic_feats.shape[1]
        n_edges = g.number_of_edges()

        print("""----Data statistics------'
          #Edges %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
              (n_edges,
               len(train_idx),
               len(val_idx),
               len(test_idx)))


        rel_num += 1
        n_edges = g.number_of_edges()

        # create model
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = DualHGNN_Two(
                    g,
                    args,
                    args.num_layers,
                    rel_num,
                    args.pred_dim,
                    num_struct_feats,
                    num_semantic_feats,
                    args.num_hidden,
                    heads,
                    H,
                    args.in_drop,
                    args.attn_drop,
                    args.residual,
                    get_hypergraph_centrality(g,H),
                    args.scale,
                    args.norm,
                    args.edge_mode,
                    args.negative_slope,
                    logger 
                )


        model = model.cuda()
        model = model.to(torch.bfloat16)
        edge_types = edge_types.cuda()

        print(model)
        model_path = save_root + str(cross_id) + '_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.patience, save_path=model_path, min_epoch=args.min_epoch)

        if cuda:
            model.cuda()
            edge_types = edge_types.cuda()

        loss_fcn = torch.nn.MSELoss()

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()


            logits, training_loss, mem_allocated = model(g, struct_feats,semantic_feats, edge_types, H , args.dataset , labels =labels, idx = train_idx,  training=True)

            total_loss = training_loss  
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            if epoch >= 3:
                dur.append(time.time() - t0)

            train_ndcg = get_rank_metrics(logits[train_idx], labels[train_idx], 100)


            model.eval()


            with torch.no_grad():
                val_logits, loss, _ = model(g, struct_feats,semantic_feats, edge_types, H , args.dataset , labels =labels, idx = val_idx,  training=True)
                val_loss, val_ndcg, val_spm, val_medianAE  = rank_evaluate(val_logits[val_idx], labels[val_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)
                test_loss, test_ndcg, test_spm, test_medianAE  = rank_evaluate(val_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)

            if args.early_stop:
                if args.spm:
                    stop = stopper.step(val_spm, epoch, model)
                else:
                    stop = stopper.step(val_ndcg, epoch, model)

                if stop:
                    print('best epoch :', stopper.best_epoch)
                    break

              # MB

            # 日志记录
            logger.info(
                "Epoch {:05d} | Time(s) {:.3f} | Loss {:.3f} | TrainNDCG {:.3f} | "
                "ValSPM {:.3f} | ValNDCG {:.3f} | TestSPM {:.3f} | TestNDCG {:.3f} | "
                "Test_medianAE {:.3f} | ETputs(KTEPS) {:.2f} | GPU: allocated {:.2f} MB".format(
                    epoch, np.mean(dur), training_loss.item(), train_ndcg,
                    val_spm, val_ndcg, test_spm, test_ndcg, test_medianAE,
                    n_edges / np.mean(dur) / 1000, mem_allocated
                )
            )

            torch.cuda.empty_cache()



 
        if args.early_stop:
            model.load_state_dict(torch.load(model_path))


        model.eval()

        with torch.no_grad():
            test_logits  = model(g, struct_feats,semantic_feats, edge_types, H , args.dataset , labels , training=False)

            test_loss, test_ndcg, test_spearman, test_medianAE = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)  #len(test_idx)
            test_overlap = overlap(labels[test_idx], test_logits[test_idx], 100)

            logger.info(f"Final Test Results for fold {cross_id+1}:")
            logger.info("Test NDCG {:.3f} | Test Loss {:.3f} | Test Spearman {:.3f} | Test Overlap {:.3f} | Test_medianAE {:.3f}".
                  format(test_ndcg, test_loss, test_spearman, test_overlap, test_medianAE))
                        
            # print('------------------------------------------------------------------------------------'*2)

            test_loss_20, test_ndcg_20,test_spearman_20,  test_medianAE_20 = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 20, loss_fcn, spearman=True)
            test_overlap_20 = overlap(labels[test_idx], test_logits[test_idx], 20)  


            test_loss_50, test_ndcg_50, test_spearman_50,  test_medianAE_50 = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 50, loss_fcn, spearman=True)
            test_overlap_50 = overlap(labels[test_idx], test_logits[test_idx], 50)  


            test_loss_200, test_ndcg_200, test_spearman_200, test_medianAE_200 = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 200, loss_fcn, spearman=True)
            test_overlap_200 = overlap(labels[test_idx], test_logits[test_idx], 200)              

            test_loss_test, test_ndcg_test, test_spearman_test,  test_medianAE_test = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), len(test_idx), loss_fcn, spearman=True)
            test_overlap_test = overlap(labels[test_idx], test_logits[test_idx], len(test_idx))              

 

        ndcg_scores.append(test_ndcg)
        spearmans.append(test_spearman)
        rmses.append(torch.sqrt(test_loss).item())
        overlaps.append(test_overlap)
        medAEs.append(test_medianAE)


        ndcg_scores_20.append(test_ndcg_20)
        overlaps_20.append(test_overlap_20)
        spearmans_20.append(test_spearman_20)
        rmses_20.append(torch.sqrt(test_loss_20).item())
        medAEs_20.append(test_medianAE_20)


        ndcg_scores_50.append(test_ndcg_50)
        overlaps_50.append(test_overlap_50)
        spearmans_50.append(test_spearman_50)
        rmses_50.append(torch.sqrt(test_loss_50).item())
        medAEs_50.append(test_medianAE_50)
        

        ndcg_scores_200.append(test_ndcg_200)
        overlaps_200.append(test_overlap_200)
        spearmans_200.append(test_spearman_200)
        rmses_200.append(torch.sqrt(test_loss_200).item())
        medAEs_200.append(test_medianAE_200)
        

        ndcg_scores_test.append(test_ndcg_test)
        overlaps_test.append(test_overlap_test)
        spearmans_test.append(test_spearman_test)
        rmses_test.append(torch.sqrt(test_loss_test).item())
        medAEs_test.append(test_medianAE_test)
        


    print()
    ndcg_scores = np.array(ndcg_scores)
    spearmans = np.array(spearmans)
    rmses = np.array(rmses)
    overlaps = np.array(overlaps)
    medAEs = np.array(medAEs)

    ndcg_scores_20 = np.array(ndcg_scores_20)
    spearmans_20 = np.array(spearmans_20)
    overlaps_20 = np.array(overlaps_20)
    rmses_20 = np.array(rmses_20)
    medAEs_20 = np.array(medAEs_20)

    ndcg_scores_50 = np.array(ndcg_scores_50)
    spearmans_50 = np.array(spearmans_50)
    overlaps_50 = np.array(overlaps_50)
    rmses_50 = np.array(rmses_50)
    medAEs_50 = np.array(medAEs_50)

    ndcg_scores_200 = np.array(ndcg_scores_200)
    spearmans_200 = np.array(spearmans_200)
    overlaps_200 = np.array(overlaps_200)
    rmses_200 = np.array(rmses_200)
    medAEs_200 = np.array(medAEs_200)    

    ndcg_scores_test = np.array(ndcg_scores_test)
    spearmans_test = np.array(spearmans_test)
    overlaps_test = np.array(overlaps_test)
    rmses_test = np.array(rmses_test)
    medAEs_test = np.array(medAEs_test)    


    results = {'ndcg': ndcg_scores,
               'spearman': spearmans,
               'rmse': rmses,
               'overlap': overlaps,
               'args': vars(args)}
    
    result_path = save_root + args.dataset + args.save_path.replace('checkpoint.pt', '') + 'result.pk'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    pk.dump(results, open(result_path, 'wb'))

    logger.info("\n\n" + "="*100)
    logger.info("FINAL RESULTS ACROSS ALL FOLDS")
    logger.info("="*100)
    
    # 定义打印结果的函数
    def log_results(k, ndcg, spm, overlap, rmse, medAE):
        logger.info(f"Top-{k} Results:")
        logger.info(f"  NDCG:    {np.mean(ndcg):.4f} ± {np.std(ndcg):.4f}")
        logger.info(f"  Spearman: {np.mean(spm):.4f} ± {np.std(spm):.4f}")
        logger.info(f"  Overlap: {np.mean(overlap):.4f} ± {np.std(overlap):.4f}")
        logger.info(f"  RMSE:    {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
        logger.info(f"  MedianAE: {np.mean(medAE):.4f} ± {np.std(medAE):.4f}")
        logger.info("-"*80)
    
    # 打印不同K值的结果
    log_results(20, ndcg_scores_20, spearmans_20, overlaps_20, rmses_20, medAEs_20)
    log_results(50, ndcg_scores_50, spearmans_50, overlaps_50, rmses_50, medAEs_50)
    log_results(100, ndcg_scores, spearmans, overlaps, rmses, medAEs)
    log_results(200, ndcg_scores_200, spearmans_200, overlaps_200, rmses_200, medAEs_200)
    log_results("Test", ndcg_scores_test, spearmans_test, overlaps_test, rmses_test, medAEs_test)
    
    # 保存结果
    results = {
        'ndcg_20': ndcg_scores_20,
        'spearman_20': spearmans_20,
        'overlap_20': overlaps_20,
        'rmse_20': rmses_20,
        'medAE_20': medAEs_20,
        
        'ndcg_50': ndcg_scores_50,
        'spearman_50': spearmans_50,
        'overlap_50': overlaps_50,
        'rmse_50': rmses_50,
        'medAE_50': medAEs_50,
        
        'ndcg_100': ndcg_scores,
        'spearman': spearmans,
        'overlap': overlaps,
        'rmse': rmses,
        'medAE': medAEs,
        
        'ndcg_200': ndcg_scores_200,
        'spearman_200': spearmans_200,
        'overlap_200': overlaps_200,
        'rmse_200': rmses_200,
        'medAE_200': medAEs_200,
        
        'ndcg_test': ndcg_scores_test,
        'spearman_test': spearmans_test,
        'overlap_test': overlaps_test,
        'rmse_test': rmses_test,
        'medAE_test': medAEs_test,
        
        'args': vars(args)
    }

    logger.info(f"Results saved to {result_path}")






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DualHGNN_Two')

    # datasets set 
    parser.add_argument("--dataset", type=str, default='FB15k_rel_two',
                        help="The input dataset. Can be FB15k_rel")
    
    # FB15k_rel_two, IMDB_S_rel_two, TMDB_rel_two, MUSIC_rel_Two
    parser.add_argument("--data_path", type=str, default='./datasets/fb15k_rel.pk',
                        help="path of dataset")
    
    # training parameters
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=1,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=10000, 
                        help="number of training epochs")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument('--spm', action="store_true", default=True)    
    
    # model details
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=20,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--loss_beta', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=1000,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--scale', action="store_true", default=False,
                        help="utilize centrality to scale scores")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='granrel_checkpoint.pt',
                        help='the path to save the best model')

    parser.add_argument('--norm', action="store_true", default=True)
    parser.add_argument('--edge-mode', type=str, default='MUL')
    
    parser.add_argument('--semantic_mode', type = str, default='hyper_transformer',help='semantic encoding mode')  
    parser.add_argument('--structure_mode', type=str, default = 'hyper_attention', help='stracture encoding mode')
    parser.add_argument('--list-num', type=int, default=100)
    parser.add_argument('--contrastive_size', type=int, default=2000)

    parser.add_argument('--fusion_mode', type=str, default='adaptive', help='the fusion module design between structure and semantic prediction')
    parser.add_argument('--eta', type=float, default=0.2,
                        help="the flexible paramters for fusion between structure and semantic embeddings")
    parser.add_argument('--chunked_size', type=int, default=10000,
                        help="the size of sparse chunked for transformer")
    
    parser.add_argument('--model', type = str, default= 'dualhgcn',help='the model of dualHGNN')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="the flexible paramters for loss fuction of contrastive learning object")
    parser.add_argument('--beta', type=float, default=0.2,
                        help="the flexible paramters for loss fuction of unimodal prediction results")

    args = parser.parse_args()

    log_dir = f"./logs/{args.dataset}/{args.fusion_mode}/{args.num_layers}/{args.num_heads}/{args.num_hidden}/{args.eta}/{args.chunked_size}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/training_{timestamp}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = RealTimeFileHandler(log_filename)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    logger.info(f"Arguments: {args}")

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    print(args)

    main(args, logger)



