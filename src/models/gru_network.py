#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    """GRU network"""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(GRUModel, self).__init__()
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, 'GRU')(ninp, nhid, nlayers, dropout=dropout)
        
        self.decoder = nn.Linear(nhid, ntoken)


    def forward(self, input_):
        emb = self.drop(self.encoder(input_))
        output, _ = self.rnn(emb)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        #return F.softmax(decoded, dim=1)
        return decoded

if __name__ == '__main__':

    net = GRUModel(20, 2, 50, 2)
    seq = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(1)
    net(seq)
    #print(seq)
    #print(seq.shape)
    #print(net(seq))
