#!/usr/bin/env python3
import torch
import torch.nn as nn

class GruNet(nn.Module):
    """GRU network that accepts a mini batch"""

    def __init__(self, n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=0.5, bidirectional=False, pad_idx=0):
        super(GruNet, self).__init__()
        self.model = "gru"
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

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

    def forward(self, input_tensor, input_lengths):
        """Expects input tensor of shape (batch, max_seq_len)"""

        # Embed input
        emb = self.encoder(input_tensor)
        emb = self.drop(emb)
        # Reshape from (batch, max_seq_len, emb) to (max_seq_len, batch, emb)
        emb = emb.permute(1, 0, 2)
        
        # RNN
        emb = nn.utils.rnn.pack_padded_sequence(emb, input_lengths)
        output, hidden_state = self.rnn(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.drop(output)
        
        # Get embedding of sequence
        emb_seq = hidden_state[-1]

        # Decode 
        decoded = self.decoder(output)

        # Reshape from (max_seq_len, batch, n_tokens) to
        # (batch, n_tokens, max_seq_length)
        decoded = decoded.permute(1,2,0)

        return {'output': decoded, 'emb_1': emb_seq, 'emb_2': emb_seq}

if __name__ == '__main__':

    n_tokens = 7
    embedding_size = 3
    hidden_size = 12
    n_layers = 2
    dropout = 0.5
    bidirectional = False

    net = GruNet(n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=dropout, bidirectional=bidirectional)

    input_ = torch.LongTensor([[1,4,5,1,2],[1,2,3,0,0]])
    input_lengths = [5, 3]

    output = net(input_, input_lengths)
    print(output['output'].shape)
    print(output['emb_1'].shape)
    print(output['emb_2'].shape)
