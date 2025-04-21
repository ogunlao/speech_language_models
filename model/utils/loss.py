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
        
    def compute_contrastive_loss(self, encoded: torch.tensor, context: torch.tensor,):
        """Futute time step prediction loss with negative contrastive loss

        Args:
            z (torch.tensor): _description_
            c (torch.tensor): _description_

        Returns:
            tuple: tuple of positive contrastive loss, negative contrastive loss, total loss
        """
        # num_neg is same as lambda_
        # encoded, context -> batch, channel, time
            
        bs, channel, sample_len = encoded.size()
        total_pos_loss, total_neg_loss = 0.0, 0.0

        for k in range(self.k_steps): # 4 projection layers
            # predict k steps in the future
            c_step = self.proj_steps[k](context.transpose(2, 1))
            c_step = c_step.transpose(2, 1)
            
            encoded_k, c_k = encoded[:,:,k:], c_step[:,:,:sample_len-k]
            cont_matrix = encoded_k.transpose(1, 2) @ c_k
            pos = torch.diagonal(cont_matrix, dim1=1, dim2=2)
            pos = F.logsigmoid(pos)
            pos_loss = torch.sum(pos)
            
            total_pos_loss += pos_loss
            
            time = c_k.size(2)
            for t in range(time):
                c_t = c_k[:, :, t]
                
                # sample detractors from encoded
                encoded_bar_t = torch.cat([encoded[:, :, :t], encoded[:, :, t+1:]], dim=2)
                p = torch.zeros((bs, time-1)).fill_(1/(time-1))
                neg_indices = p.multinomial(num_samples=self.num_neg, replacement=False)
                
                # Gather negative samples
                neg_samples = torch.gather(encoded_bar_t, dim=2, index=neg_indices.unsqueeze(1).repeat(1, channel, 1))  # (bs, channel, num_neg)

                # Compute negative loss for this time step
                neg = torch.bmm(neg_samples.transpose(1, 2), c_t.unsqueeze(-1))  # (bs, num_neg, 1)

                # neg = neg_samples.transpose(1, 2) @ c_t.T
                neg = F.logsigmoid(-1*neg)
                neg_loss = torch.sum(neg)
                total_neg_loss += neg_loss
        
        total_loss = total_pos_loss + self.num_neg*total_neg_loss
        return -1 * total_pos_loss, -1 * total_neg_loss, -1 * total_loss
        

class Wav2VecDiscreteBertLoss(nn.Module):
    def __init__(self, feat_dim, codebook_size):
        super().__init__()
        self.vocab_proj = nn.Linear(feat_dim, codebook_size)
        
    def forward(self, quantized, indices, mask, context_output,):
        return self.compute_masked_language_loss(quantized, indices, mask, context_output,)
        
    def compute_masked_language_loss(self, quantized, indices, mask, context_output,):
        # compute dicretebert loss by predicting the masked index
        # mask (B, T)
        B, C, T = quantized.size()
        
        # select masked indices
        masked_indices = indices[mask]
        
        # select masked context vectors
        mask = mask.view(B, T)
        context_output = context_output.transpose(2, 1).view(-1, C)
        predicted = context_output[mask.view(-1), :]
        
        # project samples to size of codebode
        predicted_proj = self.vocab_proj(predicted)
        
        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(predicted_proj, masked_indices)
        
        return loss
        
    
class Wav2VecContBertLoss(nn.Module):
    def __init__(self, num_neg: int, clamped_logit_weight: float,):
        super().__init__()
        self.num_neg = num_neg
        self.clamped_logit_weight = clamped_logit_weight
        
    def forward(self, feat_proj, mask, context_output):
        return self.compute_masked_language_loss(feat_proj, mask, context_output)
        
    def compute_masked_language_loss(self, feat_proj, mask, context_output):
        
        B, C, T = feat_proj.size()
        total_loss, total_aux_loss = 0.0, 0.0

        # select contrast samples
        for batch in range(B):
            mask_b = mask[batch, :] # T
            masked_feat_b = feat_proj[batch].transpose(0, 1)[mask_b, :] # T, C
            masked_context_b = context_output[batch].transpose(0, 1)[mask_b, :] # T, C
            
            time = masked_context_b.size(0) 
            for t in range(time):
                masked_context_b_t = masked_context_b[t, :].unsqueeze(0) # C
                masked_feat_b_t = masked_feat_b[t, :].unsqueeze(0) # C
                
                # sample 10 masked distractors from masked context for negative except index of pos
                neg_feat = torch.cat([masked_feat_b[:t, :], masked_feat_b[t+1:, :]], axis=0)

                p = torch.zeros((time-1, )).fill_(1/(time-1))
                neg_indices = p.multinomial(num_samples=self.num_neg, replacement=False)
                
                # TODO: implement selecting negative samples from other batches
                
                # Gather negative samples
                neg_feat_samples = torch.gather(neg_feat, dim=0, index=neg_indices.unsqueeze(1).repeat(1, C))  # (num_neg, channel)
                pos_with_neg_feat = torch.cat([masked_feat_b_t, neg_feat_samples], dim=0) # (num_neg+1, channel)
                
                # Compute dot product for this time step
                logits_b_t = masked_context_b_t @ pos_with_neg_feat.transpose(0, 1)  # (1, num_neg+1)
                
                # compute InfoNCE loss
                # select the index for the positive dot product for timestep T
                info_nce_loss_b_t = nn.functional.softmax(logits_b_t, dim=1)[0, 0]
                
                total_loss += info_nce_loss_b_t
                
                # Compute auxilliary loss
                # aply soft clamp to logits
                clamped_logits_b_t = self.clamped_logit_weight* torch.tanh(
                    logits_b_t / self.clamped_logit_weight
                    )
                aux_squared_logits_sum = torch.sum(clamped_logits_b_t ** 2)
                total_aux_loss += aux_squared_logits_sum
        
        return total_loss, total_aux_loss
    

class Wav2Vec2Loss(nn.Module):
    def __init__(self, k_steps: int, num_neg: int, 
                 feat_dim: int, softmax_weight: float,):
        super().__init__()
        self.k_steps = k_steps
        self.num_neg = num_neg
        self.feat_dim = feat_dim
        self.softmax_weight = softmax_weight
        
        # step specific affine transformations
        self.proj_steps = nn.ModuleList(nn.Linear(self.feat_dim, self.feat_dim, bias=True) \
            for i in range(self.k_steps))
        
    def forward(self, feat_enc, feat_context, mask) -> tuple:

        loss = self.compute_contrastive_loss(feat_enc, feat_context, mask)
        return loss
        
    def compute_contrastive_loss(self, quantized, context_output, mask,):
        
        B, C, T = quantized.size()
        total_loss = 0.0

        # select contrast samples, keep batch from w2v and context output
        for batch in range(B):
            mask_b = mask[batch, :] # T
            masked_feat_b = quantized[batch].transpose(0, 1)[mask_b, :] # T, C
            masked_context_b = context_output[batch].transpose(0, 1)[mask_b, :] # T, C
            
            time = masked_context_b.size(0) 
            for t in range(time):
                masked_context_b_t = masked_context_b[t, :].unsqueeze(0) # C
                masked_feat_b_t = masked_feat_b[t, :].unsqueeze(0) # C
                
                # sample 10 masked distractors from masked context for negative except index of pos
                neg_feat = torch.cat([masked_feat_b[:t, :], masked_feat_b[t+1:, :]], axis=0)

                p = torch.zeros((time-1, )).fill_(1/(time-1))
                neg_indices = p.multinomial(num_samples=self.num_neg, replacement=False)
                
                # Gather negative samples
                neg_feat_samples = torch.gather(neg_feat, dim=0, index=neg_indices.unsqueeze(1).repeat(1, C))  # (num_neg, channel)
                pos_with_neg_feat = torch.cat([masked_feat_b_t, neg_feat_samples], dim=0) # (num_neg+1, channel)
                
                # Compute dot product for this time step
                logits_b_t = masked_context_b_t @ pos_with_neg_feat.transpose(0, 1)  # (1, num_neg+1)
                c_norm = torch.linalg.norm(masked_context_b_t, dim=1)
                feat_norm = torch.linalg.norm(pos_with_neg_feat, dim=1)

                similarity_score = logits_b_t / (c_norm * feat_norm)
                similarity_score = similarity_score / self.softmax_weight
                
                contrastive_loss_b_t = -1 * nn.functional.log_softmax(similarity_score, dim=1)[0, 0]
                
                total_loss += contrastive_loss_b_t
        
        return total_loss