import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperparam import VQ_Wav2vecHyperParam


class VQ(nn.Module):
    """A vector quantization codebook class for VQ-VAE"""
    def __init__(self, codebook_size: int, codebook_dim: int,
                 num_groups: int, 
                 params: VQ_Wav2vecHyperParam):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_groups = num_groups
        if params.codebook_type == "simple":
            self.codebook = nn.Embedding(num_embeddings=self.codebook_size, 
                                        embedding_dim=self.codebook_dim)
        elif params.codebook_type == "shared":
            assert self.codebook_dim % self.num_groups == 0.
            nn.codebooks = nn.ModuleList(
                nn.Embedding(num_embeddings=self.codebook_size, 
                        embedding_dim=self.codebook_dim//self.num_groups) for _ in range(self.num_groups)
            )
        
        self.use_gumbel = params.use_gumbel
        if self.use_gumbel:
            self.gumbel_proj = nn.Sequential(
                nn.Linear(params.feat_dim, params.feat_dim, bias=False,),
                nn.Dropout(params.dropout_prob),
                nn.ReLU(),
                nn.Linear(params.feat_dim, self.codebook_size, bias=False,)
            )
    
    def gumbel_estimator(self, x, annealing_weight):
        logits = self.gumbel_proj(x.transpose(2, 1))
        
        # apply the gumbel softmax
        u = torch.rand(logits.size())
        v = -1*torch.log(-1*torch.log(u))
        
        logits = (logits + u)/annealing_weight
        logits = F.softmax(logits, dim=2)

        indexes = torch.argmax((logits + u)/annealing_weight, dim=2)
        
        quantized = self.codebook(indexes)
        quantized = quantized.transpose(2, 1)
        
        return quantized, indexes
    
    def kmeans_estimator(self, x):
        # x: batch, channels, time
        
        codes = self.codebook.weight # num_code, channels
        
        # find nearest cluster for each 
        codes = codes.unsqueeze(0).unsqueeze(1)
        x = x.permute(0, 2, 1).unsqueeze(2)
        dist = torch.sum((codes - x)**2, dim=-1)
        
        indexes = torch.argmin(dist, dim=2)
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
            
        
    
        