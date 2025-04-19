import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.encoder import Encoder, ContextNetwork

class VQVaeLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        return self.loss(output.view(-1, 256),
                target.long().view(-1))
        
        
class Wav2VecLoss(nn.Module):
    def __init__(self, k_steps: int, num_neg: int, feat_dim: int,):
        super().__init__()
        self.k_steps = k_steps
        self.num_neg = num_neg
        self.feat_dim = feat_dim
        
        # step specific affine transformations
        self.proj_steps = nn.ModuleList(nn.Linear(self.feat_dim, self.feat_dim, bias=True) \
            for i in range(self.k_steps))
        
    def forward(self, feat_enc: torch.tensor, feat_context: torch.tensor) -> tuple:

        loss = self.compute_contrastive_loss(feat_enc, feat_context,)
        return loss
        
    def compute_contrastive_loss(self, z: torch.tensor, c: torch.tensor,):
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


class Wav2Vec2Loss(nn.Module):
    def __init__(self, k_steps: int, num_neg: int, feat_dim: int,):
        super().__init__()
        self.k_steps = k_steps
        self.num_neg = num_neg
        self.feat_dim = feat_dim
        
        # step specific affine transformations
        self.proj_steps = nn.ModuleList(nn.Linear(self.feat_dim, self.feat_dim, bias=True) \
            for i in range(self.k_steps))
        
    def forward(self, feat_enc, feat_context, masked_indices) -> tuple:

        loss = self.compute_contrastive_loss(feat_enc, feat_context, masked_indices)
        return loss
        
    def compute_contrastive_loss(self, quantized: torch.tensor, context: torch.tensor, masked_indices):
        """Futute time step prediction loss with negative contrastive loss

        Args:
            quantized (torch.tensor): _description_
            context (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        # num_neg is same as lambda_
        # z, c -> batch, channel, time
        # TODO: implement contrastive loss
        bs, channel, sample_len = quantized.size()
        total_loss = 0.0

        return  -1 * total_loss
    
        