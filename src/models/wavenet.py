#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(torch.nn.Conv1d):
    """Causal 1d convolution"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class ResidualBlock(nn.Module):
    """Residual block of WaveNet architecture"""
    def __init__(self, residual_channels, dilation_channels, skip_channels,
    kernel_size=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.filter_conv = CausalConv1d(in_channels=self.residual_channels,
                                        out_channels=self.dilation_channels,
                                        kernel_size=self.kernel_size,
                                        dilation=self.dilation)
        
        self.gated_conv = CausalConv1d(in_channels=self.residual_channels,
                                       out_channels=self.dilation_channels,
                                       kernel_size=self.kernel_size,
                                       dilation=self.dilation)
        
        self._1x1_conv_res = nn.Conv1d(in_channels=self.dilation_channels,
                                       out_channels=self.residual_channels,
                                       kernel_size=1,
                                       dilation=1)
        
        self._1x1_conv_skip = nn.Conv1d(in_channels=self.dilation_channels,
                                        out_channels=self.skip_channels,
                                        kernel_size=1,
                                        dilation=1)
    
    def forward(self, x):
        # Filter convolution and gated convolution
        x_f = self.filter_conv(x)
        x_g = self.gated_conv(x)
      
        z_f = torch.tanh(x_f)
        z_g = torch.sigmoid(x_g)
        
        # Multiply filter convolution and gated convolution elementwise
        z = torch.mul(z_f, z_g)
        skip = self._1x1_conv_skip(z)
        residual = x + self._1x1_conv_res(z)

        return skip, residual

class WaveNet(nn.Module):
    """Neural network with WaveNet architecture including global inputs"""

    def __init__(self, n_tokens, embedding_size, n_dilations=4, n_repeats=1,
    kernel_size=2, residual_channels=16, dilation_channels=16,
    skip_channels=16, final_channels=16, pad_idx=0):
        super(WaveNet, self).__init__()
        self.model = 'wavenet'
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size
        self.n_dilations = n_dilations
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.final_channels = final_channels
        self.pad_idx = pad_idx

        self.encoder = nn.Embedding(self.n_tokens, self.embedding_size,
                                    padding_idx=self.pad_idx)
        
        self.causal_conv = CausalConv1d(in_channels=self.embedding_size,
                                        out_channels=self.residual_channels,
                                        kernel_size=self.kernel_size,
                                        dilation=1)

        self.dilations = [2**i for i in range(self.n_dilations)]*self.n_repeats
        self.residual_blocks = nn.ModuleList()

        for dilation in self.dilations:
            self.residual_blocks.append(
                ResidualBlock(residual_channels=self.residual_channels,
                              dilation_channels=self.dilation_channels,
                              skip_channels=self.skip_channels,
                              kernel_size=self.kernel_size,
                              dilation=dilation))
        
        self.final_1x1_conv = nn.Conv1d(in_channels=self.skip_channels,
                                        out_channels=self.final_channels,
                                        kernel_size=1,
                                        dilation=1)
        
        self.out_1x1_conv = nn.Conv1d(in_channels=self.final_channels,
                                      out_channels=self.n_tokens,
                                      kernel_size=1,
                                      dilation=1)

    def forward(self, input_tensor):
        """Expects input of shape (batch, max_seq_len) and 
           (batch, n_globals)"""
        # Embedding
        emb = self.encoder(input_tensor)

        # Reshape from (batch, max_seq_len, emb) to (batch, emb, max_seq_len)
        emb = emb.permute(0,2,1)

        # First causal convolution
        residual = self.causal_conv(emb)
        
        # Residual blocks
        skips_total = torch.zeros((residual.size(0),
                                   self.skip_channels,
                                   residual.size(2)),
                                  device=input_tensor.device)
        for residual_block in self.residual_blocks:
            skip, residual = residual_block(residual)
            skips_total += skip
        
        output = torch.relu(skips_total)
        output = self.final_1x1_conv(output)
        output = torch.relu(output)

        # Get sequence embedding
        emb_mean = output.mean(dim=2)
        emb_max = output.max(dim=2)[0]

        # Decode ouput
        output = self.out_1x1_conv(output)

        return {'output': output, 'emb_1': emb_mean, 'emb_2': emb_max}

if __name__ == '__main__':
    models_args = {'n_tokens': 10,
                   'embedding_size': 12,
                   'n_dilations': 4,
                   'n_repeats': 2,
                   'kernel_size': 2,
                   'residual_channels': 16,
                   'dilation_channels': 32,
                   'skip_channels': 8,
                   'final_channels': 4,
                   'pad_idx': 0}

    net = WaveNet(**models_args)
    input_ = torch.LongTensor([[1,4,5,1,2], [1,2,3,0,0]])
    output = net(input_)