import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperparam import VQ_Wav2vecHyperParam


class VQ(nn.Module):
    """A vector quantization codebook class for VQ-VAE"""
    def __init__(self, codebook_size: int, codebook_dim: int,
                 num_groups: int,
                 share_codebook_variables: bool,
                 params: VQ_Wav2vecHyperParam):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_groups = num_groups
        self.share_codebook_variables = share_codebook_variables
        if self.num_groups == 1:
            self.codebook = nn.Embedding(num_embeddings=self.codebook_size, 
                                        embedding_dim=self.codebook_dim)
        elif self.num_groups > 1:
            assert self.codebook_dim % self.num_groups == 0.
            if self.share_codebook_variables:
                self.codebook = nn.Embedding(num_embeddings=self.codebook_size, 
                                        embedding_dim=self.codebook_dim // self.num_groups)
            else:
                self.codebooks = nn.ModuleList(
                    nn.Embedding(num_embeddings=self.codebook_size, 
                            embedding_dim=self.codebook_dim // self.num_groups) for _ in range(self.num_groups)
                )
        
        self.use_gumbel = params.use_gumbel
        if self.use_gumbel:
            self.gumbel_proj = nn.Sequential(
                nn.Linear(params.feat_dim, params.feat_dim, bias=False,),
                nn.Dropout(params.dropout_prob),
                nn.ReLU(),
                nn.Linear(params.feat_dim, self.codebook_size * self.num_groups, bias=False,)
            )
    
    def gumbel_estimator(self, x, annealing_weight):
        bs, channel, time = x.size()
        logits = self.gumbel_proj(x.transpose(2, 1)) # bs, time, cb_size*num_grps
        
        # apply the gumbel softmax
        u = torch.rand(logits.size())
        v = -1*torch.log(-1*torch.log(u))
        
        logits = (logits + u)/annealing_weight
        
        if self.num_groups > 1:
            logits = logits.reshape(-1, time, self.codebook_size, self.num_groups)

            quantized, indexes = [], []
            for group in range(self.num_groups):
                group_logits = logits[:, :, :, group]
                group_logits = F.softmax(group_logits, dim=2)
                group_indexes = torch.argmax(group_logits, dim=2)
                
                if self.share_codebook_variables:
                    group_quantized = self.codebook(group_indexes) # batch, time, channels
                else:
                    group_quantized = self.codebooks[group](group_indexes) # batch, time, channels
                quantized.append(group_quantized)
                indexes.append(group_indexes)
            quantized = torch.concat(quantized, dim=2)
            quantized = quantized.transpose(2, 1)
        else:
            logits = F.softmax(logits, dim=2)
            indexes = torch.argmax(logits, dim=2)
            
            quantized = self.codebook(indexes)
            quantized = quantized.transpose(2, 1)
        
        return quantized, indexes
    
    def find_closest_emb(self, x, emb):
        emb = emb.unsqueeze(0).unsqueeze(1)
        x = x.permute(0, 2, 1).unsqueeze(2)
        dist = torch.sum((emb - x)**2, dim=-1)
        
        indexes = torch.argmin(dist, dim=2)
        
        return indexes
        
    def kmeans_estimator(self, x):
        # x: batch, channels, time
        
        if self.num_groups > 1:
            batch, channels, time = x.size()
            x = x.view(batch, self.codebook_dim//self.num_groups, self.num_groups, time)
            quantized, indexes = [], []
            for group in range(self.num_groups):
                x_group = x[:, :, group, :].squeeze(2)
                if self.share_codebook_variables:
                    codebook = self.codebook
                else:
                    codebook = self.codebooks[group]
                
                # find nearest cluster for each
                group_indexes = self.find_closest_emb(x_group, codebook.weight)
                group_quantized = codebook(group_indexes)
                
                quantized.append(group_quantized)
                indexes.append(group_indexes)
                
            # combine retrieved quantized vectors from different groups
            quantized = torch.concat(quantized, dim=2)
            quantized = quantized.transpose(2, 1)
        else:
            # find nearest cluster for each 
            indexes = self.find_closest_emb(x, self.codebook.weight)
            
            # retrieve quantized embeddings
            quantized = self.codebook(indexes)
            quantized = quantized.transpose(2, 1)
        
        return quantized, indexes
    
    def forward(self, x: torch.tensor, annealing_weight: float=None,):
        # use straight-through estimator
        if self.use_gumbel:
            quantized, indexes = self.gumbel_estimator(x, annealing_weight)
        else:
            quantized, indexes = self.kmeans_estimator(x,)

        return quantized, indexes
            
        
    
        