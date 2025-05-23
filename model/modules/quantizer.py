import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.config import Wav2vecHyperParam


class VQ(nn.Module):
    """A vector quantization codebook class for VQ-VAE"""
    def __init__(self, codebook_size: int, codebook_dim: int,
                 num_groups: int,
                 share_codebook_variables: bool,
                 use_gumbel: bool,
                 params: Wav2vecHyperParam,
                 diversity_weight: float | None = None,):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_groups = num_groups
        self.diversity_weight = diversity_weight
        self.embedding_dim = self.codebook_dim // self.num_groups
        self.share_codebook_variables = share_codebook_variables
        self.use_gumbel = use_gumbel
        if self.num_groups == 1:
            self.codebook = nn.Embedding(num_embeddings=self.codebook_size, 
                                        embedding_dim=self.codebook_dim)
            self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)
            
        # create multiple groups of codebook
        elif self.num_groups > 1:
            assert self.codebook_dim % self.num_groups == 0.
            if self.share_codebook_variables:
                self.codebook = nn.Embedding(num_embeddings=self.codebook_size, 
                                        embedding_dim=self.embedding_dim,)
                self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)
            else:
                self.codebooks = nn.ModuleList(
                    nn.Embedding(num_embeddings=self.codebook_size, 
                            embedding_dim=self.embedding_dim) for _ in range(self.num_groups)
                )
                for codebook in self.codebooks:
                    codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)
        
        if self.use_gumbel:
            self.gumbel_proj = nn.Sequential(
                nn.Linear(params.feat_dim, params.feat_dim, bias=False,),
                nn.Dropout(params.dropout_prob),
                nn.ReLU(),
                nn.Linear(params.feat_dim, self.codebook_size * self.num_groups, bias=False,)
            )
    
    def gumbel_estimator(self, x, annealing_weight):
        batch, channels, time = x.size()
        logits = self.gumbel_proj(x.transpose(2, 1)) # B, T, codebook_size*num_grps
        
        # apply the gumbel softmax
        u = torch.rand(logits.size())
        v = -1*torch.log(-1*torch.log(u))
        
        logits = (logits + v) / annealing_weight
        
        if self.num_groups > 1:
            logits = logits.reshape(-1, time, self.codebook_size, self.num_groups)

            quantized, indices = [], []
            for group in range(self.num_groups):
                group_logits = logits[:, :, :, group]
                group_logits = F.softmax(group_logits, dim=2)
                group_indices = torch.argmax(group_logits, dim=2)
                
                if self.share_codebook_variables:
                    group_quantized = self.codebook(group_indices) # batch, time, channels
                else:
                    group_quantized = self.codebooks[group](group_indices) # batch, time, channels
                quantized.append(group_quantized)
                indices.append(group_indices)
        else:
            logits = F.softmax(logits, dim=2)
            indices = torch.argmax(logits, dim=2)
            
            quantized = self.codebook(indices)
            quantized = quantized.transpose(2, 1)
        
        return quantized, indices
    
    def find_closest_emb(self, x, emb):
        B, C, T = x.shape
        x = x.permute(0, 2, 1).contiguous() # B T C
        x_flat = x.view(-1, self.embedding_dim) # (BT) C

        # Calculate distances
        distances = torch.sum(x_flat**2, dim=1, keepdim=True) + \
                    torch.sum(emb**2, dim=1) - \
                    2 * torch.matmul(x_flat, emb.t())

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (BT) 1
        encoding_indices = encoding_indices.reshape(B, T)
        
        return encoding_indices
        
    def kmeans_estimator(self, x):
        # x: B, C, T
        
        if self.num_groups > 1:
            batch, channels, time = x.size()
            x = x.view(batch, self.embedding_dim, self.num_groups, time)
            quantized, indices = [], []
            for group in range(self.num_groups):
                x_group = x[:, :, group, :].squeeze(2)
                if self.share_codebook_variables:
                    codebook = self.codebook
                else:
                    codebook = self.codebooks[group]
                
                # find nearest cluster for each
                group_indices = self.find_closest_emb(x_group, codebook.weight)
                group_quantized = codebook(group_indices)
                
                quantized.append(group_quantized)
                indices.append(group_indices)
                
            # combine retrieved quantized vectors from different groups
            quantized = torch.concat(quantized, dim=2)
            quantized = quantized.transpose(2, 1)
        else:
            # find nearest cluster for each 
            indices = self.find_closest_emb(x, self.codebook.weight)
            
            # retrieve quantized embeddings
            quantized = self.codebook(indices)
            quantized = quantized.transpose(2, 1)
        
        return quantized, indices
    
    def forward(self, x: torch.tensor, annealing_weight: float=None, 
                return_diversity_loss: bool=False,):
        # use straight-through estimator
        if self.use_gumbel:
            quantized, indices = self.gumbel_estimator(x, annealing_weight)
        else:
            quantized, indices = self.kmeans_estimator(x,)
            return quantized, indices
        # compute diversity loss for wav2vec2
        if self.diversity_weight and return_diversity_loss:
            diversity_loss = self.compute_diversity_loss(quantized, indices)
        
        if isinstance(quantized, list):
            # combine retrieved quantized vectors from different groups
            quantized = torch.concat(quantized, dim=2)
            quantized = quantized.transpose(2, 1)
        
        if return_diversity_loss:
            return quantized, indices, diversity_loss
        
        return quantized, indices 
    
    def compute_diversity_loss(self, quantized_groups, indices):
        # Diversity loss used in Wav2vec2.0
        assert isinstance(quantized_groups, list)
        
        B, C, T = quantized_groups[0].size()
        total_exp_g = 0.0
        for group in range(self.num_groups):
            quantized_g = quantized_groups[group]
            h_q = quantized_g*torch.log(torch.clamp(quantized_g, 1e-9))
            exp_min_sum_hq = torch.exp(-h_q.sum())
            total_exp_g += exp_min_sum_hq
        
        total_q = self.num_groups * B * T
        diversity_loss = (total_q - total_exp_g) / total_q
        
        return diversity_loss
            
        
        