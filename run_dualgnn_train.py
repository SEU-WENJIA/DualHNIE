import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import pickle as pk
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from model.DualGAT import DualGAT
from model.DualGT import DualGT
from model.DualGAT_GT import DualGAT_GT
from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed,  get_rank_metrics, rank_evaluate, convert_to_gpu, get_graph_centrality
from utils.metric import overlap
from utils.load_data_graph import load_data


def main(args):

    set_random_seed(0)

    ndcg_scores = []
    spearmans = []
    rmses = []
    overlaps = []

    # set the save path
    save_root = 'results/' + args.dataset + '_DualGNN/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):

        g, edge_types, _, rel_num, features_struct, features_semantic, labels, train_idx, val_idx, test_idx = \
            load_data(args.data_path, args.dataset, cross_id)     

        torch.cuda.set_device(args.gpu)
        features_struct = features_struct.cuda()
        features_semantic = features_semantic.cuda()

        labels = labels.cuda()

        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            g = g.int().to(args.gpu)

        num_feats_struct = features_struct.shape[1]
        num_feats_semantic = features_semantic.shape[1]


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

        # add self loop, to be sure edge_type
        g = dgl.add_self_loop(g)
        new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
        edge_types = torch.cat([edge_types, new_edge_types], 0)
        rel_num += 1
        n_edges = g.number_of_edges()


        # create model
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

        if args.model == 'dualgat':
            model = DualGAT(g,    
                        args,
                        args.num_layers,
                        rel_num,
                        args.pred_dim,
                        num_feats_struct ,
                        num_feats_semantic, 
                        args.num_hidden,
                        heads,
                        F.elu,
                        args.in_drop,
                        args.attn_drop,
                        args.negative_slope,
                        args.residual,
                        get_graph_centrality(g),
                        args.scale)      #initiate the model 
        elif args.model =='dualtrans':
            model = DualGT(
                    g,
                    args,
                    args.num_layers,
                    rel_num,
                    args.pred_dim,
                    num_feats_struct,
                    num_feats_semantic,
                    args.num_hidden,
                    heads,
                    args.in_drop,
                    args.attn_drop,
                    args.residual,
                    get_graph_centrality(g),
                    args.scale,
                    args.norm,
                    args.edge_mode)
        elif args.model =='dualgat_gt':
            model = DualGAT_GT(
                    g,
                    args,
                    args.num_layers,
                    rel_num,
                    args.pred_dim,
                    num_feats_struct,
                    num_feats_semantic,
                    args.num_hidden,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual,
                    get_graph_centrality(g),
                    args.scale,
                    args.norm,
                    args.edge_mode)

        print(model)
        model_path = save_root + str(cross_id) + '_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.patience, save_path=model_path,
                                           min_epoch=args.min_epoch)
        if cuda:
            model.cuda()
            edge_types = edge_types.cuda()
            
        # loss_fcn = torch.nn.CrossEntropyLoss()
        loss_fcn = torch.nn.MSELoss()

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(features_struct, features_semantic,  edge_types)   # return shaoe [N，1]
            loss = loss_fcn(logits[train_idx], labels[train_idx].unsqueeze(-1))  #[0.7*nodes_num]  MSE loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_ndcg = get_rank_metrics(logits[train_idx], labels[train_idx], 100)  

            model.eval()
            with torch.no_grad():
                # loss, ndcg_score, spear_score, medianAE_score
                val_logits = model(features_struct, features_semantic,  edge_types)    # [0.1*nodes_num]  Val datasets to valadate the accuracy of model trainding 
                val_loss, val_ndcg, val_spm, _  = rank_evaluate(val_logits[val_idx], labels[val_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)
                test_loss, test_ndcg, test_spm, _  = rank_evaluate(val_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)

            if args.early_stop:
                if args.spm:
                    stop = stopper.step(val_spm, epoch, model)
                else:
                    stop = stopper.step(val_ndcg, epoch, model)

                if stop:
                    print('best epoch :', stopper.best_epoch)
                    break

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainNDCG {:.4f} |"
                  " ValSPM {:.4f} | ValNDCG {:.4f} | TestSPM {:.4f} | TestNDCG {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), loss.item(), train_ndcg,
                         val_spm, val_ndcg, test_spm, test_ndcg, n_edges / np.mean(dur) / 1000))

        print()
        if args.early_stop:
            model.load_state_dict(torch.load(model_path))
        model.eval()


        with torch.no_grad():
            test_logits = model(features_struct, features_semantic,  edge_types) 
            test_loss, test_ndcg, test_spearman , _ = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 100, loss_fcn, spearman=True)
            test_overlap = overlap(labels[test_idx], test_logits[test_idx], 100)
            print("Test NDCG {:.4f} | Test Loss {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
                  format(test_ndcg, test_loss, test_spearman, test_overlap))

        ndcg_scores.append(test_ndcg)
        spearmans.append(test_spearman)
        rmses.append(torch.sqrt(test_loss).item())
        overlaps.append(test_overlap)

    print()
    ndcg_scores = np.array(ndcg_scores)
    print('ndcg: ', ndcg_scores, ndcg_scores.mean(), np.std(ndcg_scores))

    spearmans = np.array(spearmans)
    print('spearmans: ', spearmans, spearmans.mean(), np.std(spearmans))

    rmses = np.array(rmses)
    print('RMSE: ', rmses, rmses.mean(), np.std(rmses))

    overlaps = np.array(overlaps)
    print(overlaps, overlaps.mean(), np.std(overlaps))

    results = {'ndcg': ndcg_scores,
               'spearman': spearmans,
               'rmse': rmses,
               'overlap': overlaps,
               'args': vars(args)}

    result_path = save_root + args.save_path.replace('checkpoint.pt', '') + 'result.pk'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    pk.dump(results, open(result_path, 'wb'))






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DualGAT')
    parser.add_argument("--dataset", type=str, default='TMDB_two',
        help="The input dataset. Can be FB15k_rel")
        # FB15k_rel_semantic  IMDB_S_rel

    parser.add_argument("--data_path", type=str, default='./datasets/tmdb_rel.pk',
                        help="path of dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=3,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=10000,   
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=1,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=1000,  #1000
                        help="indicates whether to use early stop or not")
    parser.add_argument('--scale', action="store_true", default=False,
                        help="utilize centrality to scale scores")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument('--save-path', type=str, default='geni_checkpoint.pt',
                        help='the path to save the best model')
    
    parser.add_argument('--fusion', type=float, default=0.2,
                        help="param controls fusion gate")
    
    parser.add_argument('--model', type = str, default= 'dualgat',help='the model of dualHGNN')

    parser.add_argument('--spm', action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    main(args)