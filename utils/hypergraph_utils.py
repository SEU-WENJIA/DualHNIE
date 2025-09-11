import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import f1_score
from .metric import ndcg, spearman_sci
import pdb
from sklearn.metrics import f1_score, median_absolute_error

def convert_to_gpu(*data, device):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    return tuple(res)


def set_random_seed(seed=0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model, model_path):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    """
    count the size of trainable parameters in model (KB)
    :param model: model
    :return:
    """
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num


def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    """
    calculate NDCG@k metric
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :return:
    """
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)

def median_AE(y_true, y_pred):

    y_true = y_true.reshape(-1).detach().cpu().to(torch.float32).numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().to(torch.float32).numpy()
    return median_absolute_error(y_true, y_pred)


def rank_evaluate(predicts, labels, NDCG_k, loss_func, spearman=False):
    """
    evaluation used for validation or test
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :param loss_func: loss function
    :return:
    """
    with torch.no_grad():
        loss = loss_func(predicts, labels)
    if spearman:
        ndcg_score, spear_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        medianAE_score = median_AE(predicts, labels)
        return loss, ndcg_score, spear_score, medianAE_score
    else:
        ndcg_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        medianAE_score = median_AE(predicts, labels)

        return loss, ndcg_score, medianAE_score



def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    return centrality