import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .modules.encoder import Encoder
from conformer import Conformer
from .modules.quantizer import VQ
from .utils.config import Wav2vec2HyperParam
from .utils.loss import Wav2Vec2Loss

import lightning as L


class Wav2Vec2FeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: Conformer, 
                 quantizer: VQ, params: Wav2vec2HyperParam,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.quantizer = quantizer
        self.mask_embedding = torch.nn.Parameter(torch.randn(params.feat_dim))
        self.params = params
        self.w2v2_loss_fn = Wav2Vec2Loss(params.k_steps, params.num_neg, params.feat_dim)
        self.diversity_weight = params.diversity_weight

    def mask_input(self, quantized, mask_prob, mask_span):
        B, C, q_len = quantized.size()
        num_mask_start = int(q_len*mask_prob)
        mask_weight = torch.full((B, q_len), fill_value=1/q_len)
        masked_index = torch.multinomial(mask_weight, 
                                    num_samples=num_mask_start, replacement=False)
        
        # expand to mask_span
        mask = torch.zeros((B, q_len), dtype=torch.bool,)
        for batch in range(B):
            mask_starts = masked_index[batch]
            if len(mask_starts) > 0:
                starts = torch.tensor(mask_starts, dtype=torch.long,)
                offsets = torch.arange(mask_span + 1, dtype=torch.long)
                time_indices = (starts.unsqueeze(-1) + offsets.unsqueeze(0)).flatten()
                valid_indices = time_indices[(time_indices >= 0) & (time_indices < q_len)]
                if len(valid_indices) > 0:
                    mask[batch, valid_indices] = True
                    
        fill_array = self.mask_embedding
        masked_quantized = quantized.clone()
        mask_expanded = mask.unsqueeze(1) # (B, 1, T)
        fill_expanded = fill_array.unsqueeze(0).unsqueeze(-1) # (1, C, 1)
        masked_quantized = torch.where(mask_expanded, fill_expanded.expand_as(masked_quantized), masked_quantized)
        
        return masked_quantized, mask, masked_index
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        
        curr_annealing_weight = self.compute_gumbel_annealing_weight()
        q_outputs = self.quantizer(encoded, curr_annealing_weight, return_diversity_loss=True,)
            
        quantized, indices, diversity_loss = q_outputs
        
        # mask a portion of encoded
        masked_encoded, mask, masked_indices = self.mask_input(encoded, self.params.mask_prob, 
                                                        self.params.mask_span,)
        
        context_output = self.context(masked_encoded.transpose(2, 1))
        context_output = context_output.transpose(2, 1)
        
        return masked_encoded, q_outputs, context_output, masked_indices
    
    def compute_gumbel_annealing_weight(self):
        end = self.params.annealing_weight_end
        if not self.training: return end
        
        start = self.params.annealing_weight_start 
        anneal_time = self.params.anneal_time
        curr_epoch = self.current_epoch
        
        # annealed for some training epochs and then kept constant for the remainder
        annealing_weight = start + (start - end) * curr_epoch/(
                                        -anneal_time*self.trainer.max_epochs)
        return max(annealing_weight, end)
    
    def training_step(self, batch, batch_idx):
        
        x = batch["audio"]
        encoded, q_outputs, context_output, masked_indices = self.forward(x)
        quantized, indices, diversity_loss = q_outputs
        
        # compute wav2vec loss
        w2v2_loss = self.w2v2_loss_fn(quantized, context_output, masked_indices)
        
        total_loss = w2v2_loss + self.diversity_weight * diversity_loss
        
        self.log_dict({"train_loss": total_loss,
                       "diversity_loss": diversity_loss, "w2v2_loss": w2v2_loss, }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        
        x = batch["audio"]
        encoded, q_outputs, context_output, masked_indices = self.forward(x)

        quantized, indices, diversity_loss = q_outputs
        
        # compute wav2vec loss
        w2v2_loss = self.w2v2_loss_fn(quantized, context_output, masked_indices)
        
        val_total_loss = w2v2_loss + self.diversity_weight * diversity_loss
        
        self.log_dict({"val_loss": val_total_loss, "diversity_loss": diversity_loss,
                        "val_w2v2_loss": w2v2_loss,}, prog_bar=True)

    def configure_optimizers(self):
        # implement cosine annealing with warm up from https://stackoverflow.com/a/75089936
        
        optimizer = torch.optim.Adam(self.parameters(), 
                                        lr=self.params.lr,)
        train_scheduler = CosineAnnealingLR(optimizer, self.params.epochs)
        warmup_scheduler = LambdaLR(optimizer, 
                            lr_lambda=lambda current_step: 1 / (
                                10 ** (float(self.params.warmup_epochs - current_step))
                                )
                            )

        scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], 
                                 [self.params.warmup_epochs])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
    
    
if __name__ == "__main__":
    params = Wav2vec2HyperParam()
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    encoder = Encoder(5, 512, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=params.dropout_prob, w2v_large=True)
    context = Conformer(
        dim = 512,
        depth = 12,          # 12 blocks
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    )
    
    vq = VQ(codebook_size=params.codebook_size,
            codebook_dim=params.feat_dim,
            num_groups=params.num_groups,
            share_codebook_variables=params.share_codebook_variables,
            use_gumbel=params.use_gumbel,
            diversity_weight=params.diversity_weight,
            params=params,)

    vq_w2v_model = Wav2Vec2FeatureExtractor(encoder, context, vq, params=params)

    feat_enc, q_outputs, feat_context = vq_w2v_model(x)
    quantized, indices, commit_loss = q_outputs
    
