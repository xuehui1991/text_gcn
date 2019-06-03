import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    # return {'msg' : edges.src['h'] * edges.data['w']}
    #return {'msg' : edges.src['h'] * edges.data['w'][0]}
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


class GCN(nn.Module):
    def __init__(self, g, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        # self.embedlayer = nn.Embedding(node_size, in_feats)
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)
        self.g = g

    def forward(self, inputs):
        # lookup_tensor = torch.tensor(inputs , dtype=torch.long)
        # print('lookup_tensor :',lookup_tensor)
        # embeds = self.embedlayer(lookup_tensor)
        # h = self.gcn1(g, embeds)
        h = self.gcn1(self.g, inputs)
        # print('h storage :',sys.getsizeof(h.storage()))
        h = torch.relu(h)
        # print('h storage :',sys.getsizeof(h.storage()))
        h = self.gcn2(self.g, h)
        return h
