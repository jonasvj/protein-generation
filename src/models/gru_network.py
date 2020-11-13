#!/usr/bin/env python3
import torch
import torch.nn as nn

class GRUNet(nn.Module):
    """GRU network"""

    def __init__(self, n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=0.5, bidirectional=False):
        super(GRUNet, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.encoder = nn.Embedding(self.n_tokens, self.embedding_size)
        self.drop = nn.Dropout(self.dropout)

        self.rnn = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          dropout=self.dropout,
                          bidirectional=self.bidirectional)
        
        self.decoder = nn.Linear(self.hidden_size, self.n_tokens)

    def forward(self, input_):
        emb = self.drop(self.encoder(input_))
        output, hidden = self.rnn(emb)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.n_tokens)

        return decoded

if __name__ == '__main__':

    n_tokens = 20
    embedding_size = 5
    hidden_size = 50
    n_layers = 2
    dropout = 0.5
    bidirectional = False

    net = GRUNet(n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=dropout, bidirectional=bidirectional)

    input_ = torch.tensor([i for i in range(10)]).unsqueeze(1)
    output = net(input_)
    print(output)
    print(output.shape)