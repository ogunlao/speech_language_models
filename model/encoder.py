from dataclasses import dataclass 

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: str = "valid", dropout_prob: float = 0.0,):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, groups=1, bias=True,)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels,)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    

class Encoder(nn.Module):
    # 5 layer convolutional network
    def __init__(self, num_conv:int, enc_params: list, 
                 dropout_prob: float, w2v_large=False,):
        super().__init__()
        self.num_conv = num_conv
        self.enc_params = enc_params
        self.layer0 = ConvLayer(1, 512, *self.enc_params[0], dropout_prob=dropout_prob)
        self.encoder = nn.ModuleList([ConvLayer(512, 512, *self.enc_params[i],) \
                                      for i in range(1, num_conv)])
        
        # Two additional linear transformation for wav2vec_large
        self.linear = None
        if w2v_large:
            self.linear = nn.Sequential(
                nn.Linear(512, 512),
                nn.Dropout(p=dropout_prob),
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
    
    
class ContextNetwork(nn.Module):
    
    def __init__(self, num_layers:int, c_params=None, 
                 dropout_prob: float = 0.0, w2v_large:bool=False,):
        super().__init__()
        self.num_layers = num_layers
        assert len(c_params) == self.num_layers
        self.c_params = c_params
        self.c_layers = nn.ModuleList([ConvLayer(512, 512, *self.c_params[i], 
                                                 padding="same", dropout_prob=dropout_prob) \
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
    

class Wav2VecLoss(nn.Module):
    def __init__(self, k_steps:int, num_neg:int, feat_dim: int,):
        super().__init__()
        self.k_steps = k_steps
        self.num_neg = num_neg
        self.feat_dim = feat_dim
        
        # step specific affine transformations
        self.proj_steps = nn.ModuleList(nn.Linear(self.feat_dim, self.feat_dim, bias=True) \
            for i in range(self.k_steps))
        
    def forward(self, feat_enc:Encoder, feat_context:ContextNetwork) -> tuple:

        loss = self.compute_contrastive_loss(feat_enc, feat_context,)
        return loss
        
    def compute_contrastive_loss(self, z:torch.tensor, c:torch.tensor,):
        """Futute time step prediction loss with negative contrastive loss

        Args:
            z (torch.tensor): _description_
            c (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        # num_neg is same as lambda_
        # z, c -> batch, channel, time
            
        bs, channel, sample_len = z.size()
        total_pos_loss, total_neg_loss = 0.0, 0.0

        for k in range(self.k_steps): # 4 projection layers
            # predict k steps in the future
            c_step = self.proj_steps[k](c.transpose(2, 1))
            c_step = c_step.transpose(2, 1)
            
            z_k, c_k = z[:,:,k:], c_step[:,:,:sample_len-k]
            cont_matrix = z_k.transpose(1, 2) @ c_k
            pos = torch.diagonal(cont_matrix, dim1=1, dim2=2)
            pos = F.logsigmoid(pos)
            pos_loss = torch.sum(pos)
            
            total_pos_loss += pos_loss
            
            time = c_k.size(2)
            for t in range(time):
                c_t = c_k[:, :, t]
                
                # sample detractors from z
                p = torch.zeros((bs, time,)).fill_(1/time)
                neg_indices = p.multinomial(num_samples=self.num_neg, replacement=False)
                
                # Gather negative samples
                neg_samples = torch.gather(z, dim=2, index=neg_indices.unsqueeze(1).repeat(1, channel, 1))  # (bs, channel, num_neg)

                # Compute negative loss for this time step
                neg = torch.bmm(neg_samples.transpose(1, 2), c_t.unsqueeze(-1))  # (bs, num_neg, 1)

                # neg = neg_samples.transpose(1, 2) @ c_t.T
                neg = F.logsigmoid(-1*neg)
                neg_loss = torch.sum(neg)
                total_neg_loss += neg_loss
        
        total_loss = total_pos_loss + self.num_neg*total_neg_loss
        return -1 * total_pos_loss, -1 * total_neg_loss, -1 * total_loss
        
