import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .modules.encoder import Encoder, ContextNetwork
from conformer import Conformer
from .vq_wav2vec import VQ_Wav2VecFeatureExtractor
from .modules.quantizer import VQ
from .utils.config import VQ_Wav2vecHyperParam, VQ_w2v_DIscreteBERTHyperParam

import lightning as L


class VQw2vDiscreteBert(L.LightningModule):
    def __init__(self, feat_extractor: VQ_Wav2VecFeatureExtractor,
                 context_network: Conformer,
                 params: VQ_w2v_DIscreteBERTHyperParam,):
        super().__init__()
        self.feat_ext = feat_extractor
        self.context_network = context_network
        self.mask_embedding = nn.Parameter(torch.randn(params.feat_dim))
        self.params = params
        self.discrete_bert_loss_fn = nn.CrossEntropyLoss()
        self.vocab_proj = nn.Linear(self.params.feat_dim, self.params.codebook_size)

    def mask_input(self, quantized, mask_prob, mask_span):
        B, C, q_len = quantized.size()
        
        # select indices to mask
        num_mask_start = int(q_len*mask_prob)
        mask_weight = torch.full((B, q_len), fill_value=1/q_len)
        mask_start_index = torch.multinomial(mask_weight, 
                                    num_samples=num_mask_start, replacement=False)
        
        # For each mask index, compute a span aaccording to normal dist
        mask_spans = torch.zeros_like(mask_start_index).fill_(mask_span).float()
        mask_spans = torch.normal(mean=mask_spans, std=mask_spans).to(torch.long)
        mask_spans = torch.nn.functional.relu(mask_spans)
        
        # expand to mask_span
        mask = torch.zeros((B, q_len), dtype=torch.bool,)
        max_span = torch.max(mask_spans)
        time_indices = []
        for batch in range(B):
            mask_starts = mask_start_index[batch]
            mask_span_b = mask_spans[batch]
            if len(mask_starts) > 0:
                mask_starts = mask_starts.to(torch.long)
                offsets = torch.zeros(mask_starts.size()[0], max_span + 1, dtype=torch.long)
                for i, mask_span in enumerate(mask_span_b):
                    offset_arr = torch.arange(mask_span + 1, dtype=torch.long)
                    offsets[i, :len(offset_arr)] = offset_arr
                
                time_indices_b = (mask_starts.unsqueeze(1) + offsets).flatten()
                time_indices_b = torch.unique(time_indices_b)
                time_indices_b = time_indices_b[(time_indices_b < q_len)]
                if len(time_indices_b) > 0:
                    mask[batch, time_indices_b] = True
                time_indices.append(time_indices_b)
                    
        
        # fill masked indices with trainable param
        fill_array = self.mask_embedding
        masked_quantized = quantized.clone()
        mask_expanded = mask.unsqueeze(1) # (B, 1, T)
        fill_expanded = fill_array.unsqueeze(0).unsqueeze(-1) # (1, C, 1)
        masked_quantized = torch.where(mask_expanded, fill_expanded.expand_as(masked_quantized), masked_quantized)
        
        return masked_quantized, mask
        
        
    def forward(self, x):
        with torch.no_grad():
            encoded = self.feat_ext.encoder(x)
        
            # using gumbel softmax to quantize encoded
            q_outputs = self.feat_ext.quantizer(encoded, self.params.annealing_weight_end)
            quantized, indices = q_outputs
        
        # mask a portion of quantized
        masked_q, mask = self.mask_input(quantized, self.params.mask_prob, 
                                                        self.params.mask_span,)
        
        context_output = self.context_network(masked_q.transpose(2, 1))
        context_output = context_output.transpose(2, 1)
        
        return encoded, q_outputs, context_output, mask
    
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
        loss = self.discrete_bert_loss_fn(predicted_proj, masked_indices)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        
        x = batch["audio"]
        encoded, q_outputs, context_output, mask = self.forward(x)
        quantized, indices = q_outputs
        
        # compute cross entropy loss
        train_loss = self.compute_masked_language_loss(quantized, indices, 
                                                       mask, context_output,)
        
        self.log_dict({"train_loss": train_loss, }, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        
        x = batch["audio"]
        encoded, q_outputs, context_output, mask = self.forward(x)
        quantized, indices = q_outputs
        
        # compute cross entropy loss
        val_loss = self.compute_masked_language_loss(quantized, indices, 
                                                       mask, context_output,)
        
        self.log_dict({"val_w2v2_loss": val_loss,}, prog_bar=True)

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
    
    # first initialise a vq-wav2vec model
    vq_w2v_params = VQ_Wav2vecHyperParam()
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    vq_w2v_encoder = Encoder(5, 512, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=vq_w2v_params.dropout_prob, w2v_large=True)
    vq_w2v_context = ContextNetwork(12, 512, [(i, 1) for i in range(2, 14)], 
                             dropout_prob=vq_w2v_params.dropout_prob, w2v_large=True)
    
    vq = VQ(codebook_size=vq_w2v_params.codebook_size,
            codebook_dim=vq_w2v_params.feat_dim,
            num_groups=vq_w2v_params.num_groups,
            share_codebook_variables=vq_w2v_params.share_codebook_variables,
            use_gumbel=vq_w2v_params.use_gumbel,
            params=vq_w2v_params,)
    
    feat_extractor = VQ_Wav2VecFeatureExtractor(vq_w2v_encoder, vq_w2v_context, 
                                                vq, vq_w2v_params)
    
    # Then,use feat_extractor to setup DiscreteBert
    
    params = VQ_w2v_DIscreteBERTHyperParam()
    
    context = Conformer(
        dim = 256,
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

    vq_w2v_model = VQw2vDiscreteBert(feat_extractor, context, params=params)

    feat_enc, q_outputs, feat_context = vq_w2v_model(x)
    quantized, indices, commit_loss = q_outputs
    
