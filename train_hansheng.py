import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
import argparse
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
parser = argparse.ArgumentParser(description='MyVAE')
parser.add_argument("--resume",
                    default=False,
                    action='store_true',
                    help="Whether to gen graph again.")
parser.add_argument('--dataset', type=str, default='R8',
                    help='dataset name')
args = parser.parse_args()

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = args.dataset

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 100
word_vector_map = {}
if not args.resume:
    # shulffing
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()
    # print(doc_train_list)
    # print(doc_test_list)

    doc_content_list = []
    f = open('data/corpus/' + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    # print(doc_content_list)

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    print(train_ids)
    random.shuffle(train_ids)

    # partial labeled data
    #train_ids = train_ids[:int(0.2 * len(train_ids))]

    train_ids_str = '\n'.join(str(index) for index in train_ids)
    f = open('data/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    print(test_ids)
    random.shuffle(test_ids)

    test_ids_str = '\n'.join(str(index) for index in test_ids)
    f = open('data/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()

    ids = train_ids + test_ids
    print(ids)
    print(len(ids))

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    word_doc_list = {}

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    '''
    Word definitions begin
    '''
    '''
    definitions = []

    for word in vocab:
        word = word.strip()
        synsets = wn.synsets(clean_str(word))
        word_defs = []
        for synset in synsets:
            syn_def = synset.definition()
            word_defs.append(syn_def)
        word_des = ' '.join(word_defs)
        if word_des == '':
            word_des = '<PAD>'
        definitions.append(word_des)

    string = '\n'.join(definitions)


    f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
    f.write(string)
    f.close()

    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vec.fit_transform(definitions)
    tfidf_matrix_array = tfidf_matrix.toarray()
    print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

    word_vectors = []

    for i in range(len(vocab)):
        word = vocab[i]
        vector = tfidf_matrix_array[i]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        word_vector = word + ' ' + temp
        word_vectors.append(word_vector)

    string = '\n'.join(word_vectors)

    f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
    f.write(string)
    f.close()

    word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)
    word_embeddings_dim = len(embd[0])
    '''

    '''
    Word definitions end
    '''

    # label list
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()

    # x: feature vectors of training docs, no initial features
    # slect 90% training set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        # y.append(one_hot)
        y.append(label_index)
    y = np.array(y)
    # print(y)

    # tx: feature vectors of test docs, no initial features
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                    shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        # ty.append(one_hot)
        ty.append(label_index)
    ty = np.array(ty)
    print(ty)

    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words

    word_vectors = np.random.uniform(-0.01, 0.01,
                                    (vocab_size, word_embeddings_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))


    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        # ally.append(one_hot)
        ally.append(label_index)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        # ally.append(one_hot)
        ally.append(0)

    ally = np.array(ally)
    print('max label index of ally',np.max(ally),'min :',np.min(ally))
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    '''
    Doc word heterogeneous graph
    '''

    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    # word vector cosine similarity as weights

    '''
    for i in range(vocab_size):
        for j in range(vocab_size):
            if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
                vector_i = np.array(word_vector_map[vocab[i]])
                vector_j = np.array(word_vector_map[vocab[j]])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(vocab[i], vocab[j], similarity)
                    row.append(train_size + i)
                    col.append(train_size + j)
                    weight.append(similarity)
    '''
    # doc word frequency
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                    word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))






    # ##############################################################################
    # We can also visualize the graph by converting it to a `networkx
    # <https://networkx.github.io/documentation/stable/>`_ graph:

    # import networkx as nx
    # # Since the actual graph is undirected, we convert it for visualization
    # # purpose.
    # nx_G = G.to_networkx().to_undirected()
    # # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

    # ##############################################################################
    # Step 2: assign features to nodes or edges
    # --------------------------------------------
    # Graph neural networks associate features with nodes and edges for training.
    # For our classification example, we assign each node's an input feature as a one-hot vector:
    # node :math:`v_i`'s feature vector is :math:`[0,\ldots,1,\dots,0]`,
    # where the :math:`i^{th}` position is one.

    # In DGL, we can add features for all nodes at once, using a feature tensor that
    # batches node features along the first dimension. This code below adds the one-hot
    # feature for all nodes:
    def dump(weight, row, col, node_size, train_size, test_size, ally, y, allx,tx):
        u = {}
        u['node_size'] = node_size
        u['train_size'] = train_size
        u['test_size'] = test_size
        # dump objects
        f = open("data/dgl.{}.weight".format(dataset), 'wb')
        pkl.dump(weight, f)
        f.close()

        f = open("data/dgl.{}.row".format(dataset), 'wb')
        pkl.dump(row, f)
        f.close()

        f = open("data/dgl.{}.col".format(dataset), 'wb')
        pkl.dump(col, f)
        f.close()

        f = open("data/dgl.{}.graph".format(dataset), 'wb')
        pkl.dump(u, f)
        f.close()

        f = open("data/dgl.{}.ally".format(dataset), 'wb')
        pkl.dump(ally, f)
        f.close()

        f = open("data/dgl.{}.y".format(dataset), 'wb')
        pkl.dump(y, f)
        f.close()

        f = open("data/dgl.{}.ty".format(dataset), 'wb')
        pkl.dump(ty, f)
        f.close()

        f = open("data/dgl.{}.allx".format(dataset), 'wb')
        pkl.dump(allx, f)
        f.close()

        f = open("data/dgl.{}.tx".format(dataset), 'wb')
        pkl.dump(tx, f)
        f.close()
        # f = open("data/ind.{}.allx".format(dataset), 'wb')
        # pkl.dump(allx, f)
        # f.close()

        # f = open("data/ind.{}.ally".format(dataset), 'wb')
        # pkl.dump(ally, f)
        # f.close()

        # f = open("data/ind.{}.adj".format(dataset), 'wb')
        # pkl.dump(adj, f)
        # f.close()
    dump(weight,row,col,node_size,train_size,test_size,ally,y,allx,tx)


def load():
    # x = sp.csr_matrix()
    import os
    pwd = os.getcwd()
    print('pwd :',pwd)
    f = open(pwd + "/data/dgl.{}.weight".format(dataset), 'rb')
    print('path :',pwd + "/data/dgl.{}.weight".format(dataset))
    weight = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.row".format(dataset), 'rb')
    row = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.col".format(dataset), 'rb')
    col = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.graph".format(dataset), 'rb')
    u = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.ally".format(dataset), 'rb')
    ally = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.y".format(dataset), 'rb')
    y = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.ty".format(dataset), 'rb')
    ty = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.allx".format(dataset), 'rb')
    allx = pkl.load(f)
    f.close()

    f = open(pwd + "/data/dgl.{}.tx".format(dataset), 'rb')
    tx = pkl.load(f)
    f.close()
    # f = open(pwd + "/data/ind.{}.adj".format(dataset), 'rb')
    # adj = pkl.load(f)
    # f.close()
    return weight,row,col,u,ally,y,ty,allx,tx
weight,row,col,u,ally,y,ty,allx,tx = load()
node_size = u['node_size']
train_size = u['train_size']
test_size = u['test_size']


def build_my_graph():
    global weight,row,col
    def initializer(shape, dtype, ctx, id_range):
        return torch.ones(shape, dtype=dtype, device=ctx)
    g = dgl.DGLGraph(multigraph=True)
    g.set_n_initializer(initializer)
    g.add_nodes(node_size)
    cnt = 0
    # weight = weight[:100]
    # row = row[:100]
    # col = col[:100]
    # print('len(row) :',len(row))
    g.add_edges(row,col)
    # for item,irow,icol in zip(weight,row,col):
    #     # print('irow :',irow,'icol :',icol)
        # g.add_edges(irow , icol)
    #     # print('item :',item,'cnt :',cnt)
    #     # print('g edges:',g.edges())
    #     # print('len :',len(g.edges))
    #     # print('We have %d nodes.' % g.number_of_nodes())
    #     # print('We have %d edges.' % g.number_of_edges())
    #     g.edges[cnt].data['w'] = torch.tensor([item])
    #     cnt += 1
    g.edges[row,col].data['w'] = torch.tensor(weight)
    return g
G = build_my_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

# G.ndata['feat'] = torch.eye(G.number_of_nodes())


# Define the message & reduce function
# NOTE: we ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    # return {'msg' : edges.src['h'] * edges.data['w']}
    # return {'msg' : edges.src['h'] * edges.data['w'][0]}
    return {'msg' : edges.src['h']}


def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

def print_edgemsg(edges):
    print('w :',edges.data['w'],'h :',edges.src['h'] ,"w * h :",edges.data['w'][0] * edges.src['h'])
    return {'msg' : edges.src['h']}
    
# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # for u,v in zip(*g.edges()):
        #     g.send((u,v),print_edgemsg)
        # trigger message passing on all edges 
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        # print('h shape:',h.size())
        return self.linear(h)

###############################################################################
# In general, the nodes send information computed via the *message functions*,
# and aggregates incoming information with the *reduce functions*.
#
# We then define a deeper GCN model that contains two GCN layers:

# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, node_size, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        # self.embedlayer = nn.Embedding(node_size, in_feats)
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        # lookup_tensor = torch.tensor(inputs , dtype=torch.long)
        # print('lookup_tensor :',lookup_tensor)
        # embeds = self.embedlayer(lookup_tensor)
        # h = self.gcn1(g, embeds)
        h = self.gcn1(g, inputs)
        # print('h storage :',sys.getsizeof(h.storage()))
        h = torch.relu(h)
        # print('h storage :',sys.getsizeof(h.storage()))
        h = self.gcn2(g, h)
        return h

num = np.max(ally)
print('num :',num)


###############################################################################
# Step 4: data preparation and initialization
# -------------------------------------------
#
# We use one-hot vectors to initialize the node features. Since this is a
# semi-supervised setting, only the instructor (node 0) and the club president
# (node 33) are assigned labels. The implementation is available as follow.


# inputs = sp.torch.eye(G.number_of_nodes())
node_size = G.number_of_nodes()
net = GCN(node_size, word_embeddings_dim, 100, num + 1)
# inputs = range(node_size)
# inputs = torch.eye(node_size)

# print('input storage :',sys.getsizeof(inputs.storage()))

# node_arr = np.random.random(size=(node_size,200))
# inputs = torch.from_numpy(node_arr).float()

inputsa = allx.toarray()
inputsb = tx.toarray()
inputs = np.concatenate((inputsa, inputsb))
inputs = torch.from_numpy(inputs).float()

# i = torch.LongTensor([range(node_size),range(node_size)])
# v = torch.FloatTensor(node_size * [1])
# inputs = torch.sparse.FloatTensor(i, v, torch.Size([node_size,node_size]))
ylength = len(y)

# print('y length :',len(y))
# print('y :',y)
labeled_nodes = torch.tensor(range(ylength))  # only the instructor and the president nodes are labeled
labels = torch.from_numpy(y)  # their labels are different
test_labels = torch.from_numpy(ty)
###############################################################################
# Step 5: train then visualize
# ----------------------------
# The training loop is exactly the same as other PyTorch models.
# We (1) create an optimizer, (2) feed the inputs to the model,
# (3) calculate the loss and (4) use autograd to optimize the model.
idx_test = range(len(ally),len(ally) + test_size)
def test(epoch):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        logits = net(G,inputs)
        logp = F.log_softmax(logits, 1)
        # print('shape 1:',logp[idx_test].size(), 'shape 2 :',test_labels.size())
        # for item,lb in zip(logp[idx_test],test_labels):
        #     print("item :",item,"lb :",lb)
        test_loss += F.nll_loss(logp[idx_test], test_labels)
        res = logp[idx_test].detach().numpy()
        res = res.argmax(axis=1)
        correct = np.sum(np.equal(res,ty))
        print('epoch %d correct %d acc %.4f' % (epoch, correct, correct / test_size))
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    time_start = time.time()
    logits = net(G, inputs)
    # we save the logits for visualization later
    # all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    time_end = time.time()
    print('time :',time_end - time_start)
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    test(epoch)

idx_test = range(len(ally),len(ally) + test_size)
def test(epoch):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        logits = net(G,inputs)
        test_loss += F.nll_loss(logp[idx_test], test_labels)
        res = logits.detach().numpy()
        res = res.argmax(axis=1)
        correct = np.sum(res,ty)
        print('epoch %d correct %d acc %.4f' % (epoch, correct, correct / test_size))

        
