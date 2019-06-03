from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
# import tensorflow as tf
from sklearn import metrics
import random
import os
import sys

from dgl import DGLGraph
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

import utils
# from models import GCN, MLP

from gcn import GCN
# from gcn_update import GCN

#from gcn_mp import GCN
#from gcn_spmv import GCN

seed = random.randint(1, 200)
np.random.seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def transform_labels(labels):
    data = [np.argmax(one_hot)for one_hot in labels]
    return np.asarray(data)


def main(args):
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size, labels = utils.load_corpus(
        args.dataset)

    print("adj shape ", str(adj.shape))
#   features = np.zeros((adj.shape[0], 300))
    print("feature shape ", str(features.shape))

# #     # print(adj[0], adj[1])
# #     features = sp.identity(features.shape[0])  # featureless

# #     # Some preprocessing
# #     features = utils.preprocess_features(features)

#     print("feature shape ", str(features.shape))

#     print(type(features))
#     print(features)

#     values = features.data
# #     indices = np.vstack((features.row, features.col))
# #     i = torch.LongTensor(indices)
# #     v = torch.FloatTensor(values)
# #     shape = features.shape
# #     t_features= torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    print("train_size", train_size)
    print("train_mask shape ", str(train_mask.shape))
    print("test_mask_shape ", str(test_mask.shape))
    print("val_mask_shape", str(val_mask.shape))

    t_features = torch.FloatTensor(features.todense())
    #labels = np.concatenate([y_train, y_val, y_test], axis=0)
    print("before label shape ", str(labels.shape))
    n_classes = labels.shape[1]
    labels = transform_labels(labels)
    print("After transform label ", str(labels.shape))
    t_labels = torch.LongTensor(labels)
    t_train_mask = torch.ByteTensor(train_mask)
    t_val_mask = torch.ByteTensor(val_mask)
    t_test_mask = torch.ByteTensor(test_mask)
    in_feats = features.shape[1]


    adj_nx = nx.from_scipy_sparse_matrix(adj)
    n_edges = adj_nx.number_of_edges()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              t_train_mask.sum().item(),
              t_val_mask.sum().item(),
              t_test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        t_features = t_features.cuda()
        t_labels = t_labels.cuda()
        t_train_mask = t_train_mask.cuda()
        t_val_mask = t_val_mask.cuda()
        t_test_mask = t_test_mask.cuda()

    print("Begin to convert to graph.")
    g = DGLGraph(adj_nx)
    print("Convert to DGLGraph done.")

    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    # model = GCN(g,
    #             in_feats,
    #             args.n_hidden,
    #             n_classes)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        print("epoch ", str(epoch))
        model.train()
        # if epoch >= 3:
        t0 = time.time()
        # forward
        logits = model(t_features)
        loss = loss_fcn(logits[t_train_mask], t_labels[t_train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch >= 3:
        dur.append(time.time() - t0)

        acc = evaluate(model, t_features, t_labels, t_val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, t_features, t_labels, t_test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN')
    # parameter from text_gcn
    parser.add_argument('--dataset', type=str, default = '20ng', help = 'Dataset string.')
    parser.add_argument("--lr", type=float, default=0.02, help="Initial learning rate.")
    parser.add_argument("--n_epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n_hidden", type=int, default=200,
            help="number of hidden gcn units")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--weight_decay", type=float, default=0,
            help="Weight for L2 loss")
    parser.add_argument("--early_stopping", type=float, default=10,
            help="Tolerance for early stopping (# of epochs).")
    parser.add_argument("--max_degree", type=float, default=3,
            help="Maximum Chebyshev polynomial degree.")          

    # parameter from dgl
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--n_layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)