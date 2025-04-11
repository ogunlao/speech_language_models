from dataclasses import dataclass 

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

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
    

class Wav2VecLoss(nn.Module):
    def __init__(self, k_steps:int, num_neg:int):
        super().__init__()
        self.k_steps = k_steps
        self.num_neg = num_neg
        self.linear = nn.Linear(512, 512, bias=True)
        
    def forward(self, feat_enc:Encoder, feat_context:Context) -> torch.tensor:
        feat_context = torch.transpose(feat_context, 2, 1)
        feat_context = self.linear(feat_context)
        feat_context = torch.transpose(feat_context, 2, 1)
        
        loss = self.compute_contrastive_loss(feat_enc, feat_context,)
        return loss
        
    def compute_contrastive_loss(self, z:torch.tensor, c:torch.tensor,): 
        # num_neg is same as lambda_
        # z, c -> batch, channel, time
        sample_len = z.size()[2]
        total_loss = 0.0 
        cont_matrix = z.transpose(1, 2) @ c
        for k in range(self.k_steps):
            pos = torch.diagonal(cont_matrix, offset=-1*k, dim1=1, dim2=2)
            pos = F.logsigmoid(pos)
            pos_loss = torch.sum(pos)
            
            weight = torch.full_like(cont_matrix[0], fill_value=1/sample_len)
            neg_idx = torch.multinomial(weight, num_samples=self.num_neg, replacement=False)
            neg_idx = neg_idx.transpose(0, 1).unsqueeze(0)
            neg_idx = neg_idx.expand(2, -1, -1)
            neg = cont_matrix.gather(-1, neg_idx)
            neg = F.logsigmoid(-1*neg)
            neg_loss = torch.sum(neg)
            
            total_loss += pos_loss + self.num_neg*neg_loss
        
        return total_loss
        

if __name__ == "__main__":
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    enc = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], w2v_large=False)
    context = Context(9, [(3, 1) for _ in range(9)],)
    # context = Context(12, [(i, 1) for i in range(2, 14)], w2v_large=True)

    w2v = Wav2VecFeatureExtractor(enc, context)
    feat_enc, feat_context = w2v(x)

    loss_fn = Wav2VecLoss(4, 10)
    loss = loss_fn(feat_enc, feat_context)
    print(loss)
