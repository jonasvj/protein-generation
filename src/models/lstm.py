#!/usr/bin/env python3
import torch
import torch.nn as nn

class LstmNet(nn.Module):
    """LSTM network that accepts a mini batch"""

    def __init__(self, n_tokens, embedding_size, hidden_size, n_layers,
                 dropout=0.5, bidirectional=False, pad_idx=0):
        super(LstmNet, self).__init__()
        self.model = "lstm"
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

        self.rnn = nn.LSTM(input_size=self.embedding_size,
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
        output, state = self.rnn(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.drop(output)

        # Get embedding of sequence
        seq_emb_1 = state[0][-1]
        seq_emb_2 = state[1][-1]

        # Decode 
        decoded = self.decoder(output)

        # Reshape from (max_seq_len, batch, n_tokens) to
        # (batch, n_tokens, max_seq_length)
        decoded = decoded.permute(1,2,0)

        return {'output': decoded, 'emb_1': seq_emb_1, 'emb_2': seq_emb_2}

if __name__ == '__main__':
    models_args = {'n_tokens': 10,
                   'embedding_size': 5,
                   'hidden_size': 32,
                   'n_layers': 2,
                   'dropout': 0.5,
                   'bidirectional': False,
                   'pad_idx': 0}

    net = LstmNet(**models_args)
    input_ = torch.LongTensor([[1,4,5,1,2],[1,2,3,0,0]])
    input_lengths = [5, 3]
    output = net(input_, input_lengths)