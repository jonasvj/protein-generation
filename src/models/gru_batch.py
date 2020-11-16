#!/usr/bin/env python3
import torch
import torch.nn as nn

class GRUNet(nn.Module):
    """GRU network that accepts a mini batch"""

    def __init__(self, n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=0.5, bidirectional=False):
        super(GRUNet, self).__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_idx = 0

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.encoder = nn.Embedding(self.n_tokens, self.embedding_size,
                                    padding_idx=self.pad_idx)
        self.drop = nn.Dropout(self.dropout)

        self.rnn = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          dropout=self.dropout,
                          bidirectional=self.bidirectional)
        
        self.decoder = nn.Linear(self.hidden_size*self.num_directions,
                                 self.n_tokens)

    def forward(self, inputs, input_lengths):
        """
        Input should be a sorted list of sequences
        """
        """
        # Sort inputs by length
        inputs = sorted(inputs, key=lambda x: len(x), reverse=True)
        input_lengths = [len(input_) for input_ in inputs]
    
        input_tensor = torch.zeros((max(input_lengths), len(inputs))).long()
        # Fill input tensor with input sequences
        for i, length in enumerate(input_lengths):
            input_tensor[0:length, i] = inputs[i][:]
        """
        input_tensor = inputs
        
        emb = self.drop(self.encoder(input_tensor))

        emb = nn.utils.rnn.pack_padded_sequence(emb, input_lengths)
        output, hidden = self.rnn(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        output = self.drop(output)
        decoded = self.decoder(output).permute(0,2,1)

        return decoded


if __name__ == '__main__':

    n_tokens = 5
    embedding_size = 3
    hidden_size = 12
    n_layers = 2
    dropout = 0.5
    bidirectional = False

    net = GRUNet(n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=dropout, bidirectional=bidirectional)

    input_ = [torch.tensor([1, 2, 3]), torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4])]
    output = net(input_)