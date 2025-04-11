from dataclasses import dataclass 

import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:str="valid"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=1, bias=True,)
        
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels,)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    

class Encoder(nn.Module):
    # 5 layer convolutional network
    def __init__(self, num_conv:int, params, w2v_large=False,):
        super().__init__()
        self.num_conv = num_conv
        self.enc_params = params
        self.layer0 = ConvLayer(1, 512, *self.enc_params[0],)
        self.encoder = nn.ModuleList([ConvLayer(512, 512, *self.enc_params[i],) \
                                      for i in range(1, num_conv)])
        self.linear = None
        if w2v_large:
            # Two additional linear transformation
            self.linear = nn.Sequential(
                nn.Linear(512, 512),
                nn.Linear(512, 512),
            )
            
        
    def forward(self, x):
        x = self.layer0(x)
        for layer in self.encoder:
            x = layer(x)
            
        if self.linear:
            x = torch.transpose(x, 2, 1)
            x = self.linear(x)
            x = torch.transpose(x, 2, 1)
        
        return x
    
class Context(nn.Module):
    
    def __init__(self, num_layers:int, params=None, w2v_large:bool=False,):
        super().__init__()
        self.num_layers = num_layers
        assert len(params) == self.num_layers
        self.c_params = params
        self.c_layers = nn.ModuleList([ConvLayer(512, 512, *self.c_params[i], padding="same") \
                                      for i in range(self.num_layers)])
        self.w2v_large = w2v_large
    
    def forward(self, z):
        for layer in self.c_layers:
            if self.w2v_large:
                # add skip connections to larger models
                residual = z
                z = layer(z) + residual
            else:
                z = layer(z)
        
        return z
    
    
class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, encoder, context,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        
    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)
        
        return z, c
