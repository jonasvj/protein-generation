#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    """Postional encoder used in transformer model. Adapted from 
        github.com/pytorch/examples/blob/master/word_language_model/model.py"""
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (
            -math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model. Adapted from 
        github.com/pytorch/examples/blob/master/word_language_model/model.py"""
    def __init__(self, n_tokens, embedding_size, n_heads, hidden_size,
    n_layers, dropout=0.5, pad_idx=0, max_len=5000):
        super(TransformerModel, self).__init__()
        self.model = "transformer"
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.max_len = max_len

        # Embedding and positional encoding
        self.encoder = nn.Embedding(self.n_tokens, self.embedding_size)
        self.pos_encoder = PositionalEncoding(self.embedding_size,
                                              self.dropout,
                                              self.max_len)

        # Encoder blocks
        self.encoder_layers = TransformerEncoderLayer(self.embedding_size,
                                                      self.n_heads,
                                                      self.hidden_size,
                                                      self.dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers,
                                                      self.n_layers)

        # Decoder
        self.decoder = nn.Linear(embedding_size, n_tokens)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(
            torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input_tensor):
        """Expects input tensor of shape (batch, max_seq_len)"""
        # Create masks
        input_mask = self._generate_square_subsequent_mask(
            input_tensor.size(1)).to(input_tensor.device)
        
        input_key_padding_mask = torch.eq(input_tensor,
            torch.ones(input_tensor.shape,
            device=input_tensor.device) * self.pad_idx)
        
        # Reshape from (batch, max_seq_len) to (max_seq_len, batch)
        input_tensor = input_tensor.permute(1,0)

        # Embedding and postional encoding
        emb = self.encoder(input_tensor) * math.sqrt(self.embedding_size)
        emb = self.pos_encoder(emb)

        # Transformer encoder
        output = self.transformer_encoder(emb, input_mask,
            input_key_padding_mask)
        
        # Get sequence embedding
        emb_mean = output.mean(dim=0)
        emb_max = output.max(dim=0)[0]

        # Decoder
        output = self.decoder(output)

        # Reshape from (max_seq_len, batch, n_tokens) to
        # (batch, n_tokens, max_seq_len)
        output = output.permute(1,2,0)

        return {'output': output, 'emb_1': emb_mean, 'emb_2': emb_max}

if __name__ == '__main__':
    models_args = {'n_tokens': 10,
                   'embedding_size': 12,
                   'n_heads': 4,
                   'hidden_size': 32,
                   'n_layers': 2,
                   'dropout': 0.5,
                   'pad_idx': 0,
                   'max_len': 500}

    net = TransformerModel(**models_args)
    input_ = torch.LongTensor([[1,4,5,1,2],[1,2,3,0,0]])
    output = net(input_)
