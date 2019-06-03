import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def build_graph():
    graph = DGLGraph()
    graph.add_nodes(5)
    graph.add_edges([0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3])
    return graph


def build_dataset():
    graph = build_graph()
    features = np.random.rand(5, 10)
    labels = np.random.randint(2, size=5)
    train_mask = np.asarray([1, 1, 1, 0, 0])
    val_mask = np.asarray([0, 0, 0, 1, 0])
    test_mask = np.asarray([0, 0, 0, 0, 1])
    in_feats = features.shape[1]
    n_classes = 2
    n_edges = graph.number_of_edges()

    return graph, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges


def main(args):
    # # load and preprocess dataset
    # data = load_data(args)

    # print(data.features.shape)
    # print(data.labels.shape)
    # print(data.train_mask.shape)
    # print(data.val_mask.shape)
    # print(data.test_mask.shape)

    # print(type(data))
    # print(type(data.graph))

    # features = torch.FloatTensor(data.features)
    # labels = torch.LongTensor(data.labels)
    # train_mask = torch.ByteTensor(data.train_mask)
    # val_mask = torch.ByteTensor(data.val_mask)
    # test_mask = torch.ByteTensor(data.test_mask)
    # in_feats = features.shape[1]
    # n_classes = data.num_labels
    # n_edges = data.graph.number_of_edges()

    _graph, _features, _labels, _train_mask, _val_mask, _test_mask, _in_feats, _n_classes, _n_edges = build_dataset()
    
    print(_features.shape)
    print(_labels.shape)
    print("_train_mask_shap ", str(_train_mask.shape))
    print(_train_mask)
    print("val_mask.shape", str(_val_mask.shape))
    features = torch.FloatTensor(_features)
    labels = torch.LongTensor(_labels)
    train_mask = torch.ByteTensor(_train_mask)
    val_mask = torch.ByteTensor(_val_mask)
    test_mask = torch.ByteTensor(_test_mask)
    in_feats = _features.shape[1]
    n_classes = _n_classes
    n_edges = _graph.number_of_edges() 

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    # g = data.graph

    # # add self loop
    # if args.self_loop:
    #         g.remove_edges_from(g.selfloop_edges())
    #         g.add_edges_from(zip(g.nodes(), g.nodes()))
    # g = DGLGraph(g)

    g = _graph

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
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
