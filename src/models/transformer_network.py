import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Tranformer Network"""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        """
        ntoken: Size of dictionary of embeddings (20 for amino acids)
        ninp: Number of expected features in input, size of embedding layer
        nhead: Number of heads in the multi-headed attention models
        nhid: Dimension of the feedforward neural network
        nlayers: Number of layers
        """
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout) #Positional encoding necessary, as model contains no recurrence and no convolution
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) #The layer is made up of self-attn and feedforward neural network
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) #A stack of nlayers encoder lwayers
        self.encoder = nn.Embedding(ntoken, ninp) #Stores the embeddings of a fixed dictionary and size
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_):
        mask = self.generate_square_subsequent_mask(len(input_)).to(input_.device)
        src = self.encoder(input_) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output

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