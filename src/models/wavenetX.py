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
    def __init__(self, in_channels, out_channels, n_globals, kernel_size=1,
    stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_globals = n_globals
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.filter_conv = CausalConv1d(in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        dilation=self.dilation)
        
        self.gated_conv = CausalConv1d(in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        dilation=self.dilation)
        
        self.filter_linear = nn.Linear(in_features=self.n_globals,
                                       out_features=self.in_channels)

        self.gated_linear = nn.Linear(in_features=self.n_globals,
                                      out_features=self.in_channels)
        
        self._1x1_conv = nn.Conv1d(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  dilation=1)
    
    def forward(self, x, h):
        # Linear projections of global inputs
        h_f = self.filter_linear(h).unsqueeze(2)
        h_g = self.gated_linear(h).unsqueeze(2)

        # Filter convolution and gated convolution
        x_f = self.filter_conv(x)
        x_g = self.gated_conv(x)
      
        z_f = torch.tanh(x_f + h_f)
        z_g = torch.sigmoid(x_g + h_g)
        
        # Multiply filter convolution and gated convolution elementwise
        z = torch.mul(z_f, z_g)
        skip = self._1x1_conv(z)

        residual = x + skip

        return residual, skip

class WaveNetX(nn.Module):
    """Neural network with the WaveNet architecture"""

    def __init__(self, n_tokens, n_globals, n_outputs, embedding_size,
    n_dilations, kernel_size=2, stride=1, res_channels=16, f_channels=8):
        super(WaveNetX, self).__init__()
        self.model = 'wavenetX'
        self.n_tokens = n_tokens
        self.n_globals = n_globals
        self.n_outputs = n_outputs
        self.embedding_size = embedding_size
        self.n_dilations = n_dilations
        self.kernel_size = kernel_size
        self.stride = stride
        self.residual_channels = res_channels
        self.final_channels = f_channels
        self.pad_idx = 0

        self.encoder = nn.Embedding(self.n_tokens, self.embedding_size,
                                    padding_idx=self.pad_idx)
        
        self.causal_conv = CausalConv1d(in_channels=self.embedding_size,
                                        out_channels=self.residual_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        dilation=1)

        self.residual_blocks = nn.ModuleList()

        for i in range(self.n_dilations):
            dilation = 2**(i+1)
            self.residual_blocks.append(
                ResidualBlock(in_channels=self.residual_channels,
                              out_channels=self.residual_channels,
                              n_globals=self.n_globals*self.embedding_size,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              dilation=dilation))
        
        self.final_1x1_conv_1 = nn.Conv1d(in_channels=self.residual_channels,
                                          out_channels=self.final_channels,
                                          kernel_size=1,
                                          stride=1,
                                          dilation=1)
        
        self.final_1x1_conv_2 = nn.Conv1d(in_channels=self.final_channels,
                                          out_channels=self.n_outputs,
                                          kernel_size=1,
                                          stride=1,
                                          dilation=1)

    def forward(self, input_tensor, global_input):
        """Expects input of shape (batch, max_seq_len) and 
           (batch, n_globals)"""
        # Embedding
        emb = self.encoder(input_tensor)
        emb_global = self.encoder(global_input)

        # Reshape from (batch, max_seq_len, emb) to (batch, emb, max_seq_len)
        emb = emb.permute(0,2,1)
        # Reshape from (batch, n_global, emb) to (batch, n_global*emb)
        emb_global = emb_global.view(emb_global.size(0), -1)
    
        residual = self.causal_conv(emb)

        skips_total = torch.zeros(residual.shape, device=input_tensor.device)
        for residual_block in self.residual_blocks:
            residual, skip = residual_block(residual, emb_global)
            skips_total += skip
        
        output = torch.relu(skips_total)
        output = self.final_1x1_conv_1(output)
        output = torch.relu(output)
        output = self.final_1x1_conv_2(output)

        return {'output': output}

if __name__ == '__main__':

    n_tokens = 10
    n_globals = 4
    n_outputs = 5
    embedding_size = 3
    n_dilations = 3
    kernel_size = 2
    res_channels=16
    f_channels=8

    net = WaveNetX(n_tokens, n_globals, n_outputs, embedding_size,
        n_dilations, kernel_size=kernel_size, stride=1,
        res_channels=res_channels, f_channels=f_channels)

    input_ = torch.tensor([1, 2, 3, 3, 4, 0, 0]).unsqueeze(0)
    input_global = torch.tensor([5,6,7,8]).unsqueeze(0)
    output = net(input_, input_global)
    output = output['output']
    print(output)
    print(output.shape)