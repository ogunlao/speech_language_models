import copy
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .modules.encoder import Encoder, ContextNetwork
from .modules.quantizer import VQ
from .utils.config import VQ_Wav2vecHyperParam
from .utils.loss import Wav2VecLoss

import lightning as L


class VQ_Wav2VecFeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: ContextNetwork, 
                 quantizer: VQ, params: VQ_Wav2vecHyperParam,):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = encoder
        self.context = context
        self.quantizer = quantizer
        self.params = params
        self.w2v_loss_fn = Wav2VecLoss(params.k_steps, params.num_neg, params.feat_dim)
        self.commitment_weight = params.commitment_weight
        if self.params.use_gumbel:
            self.automatic_optimization = True


    def forward(self, x):
        z = self.encoder(x)
        
        curr_annealing_weight = self.compute_gumbel_annealing_weight()
        q_outputs = self.quantizer(z, curr_annealing_weight)
        z_hat, indices = q_outputs
        commit_loss = 0.0
        c = self.context(z_hat)
        
        return z, q_outputs, c
    
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
        if not self.automatic_optimization:
            return self.train_kmeans(batch, batch_idx)
        
        x = batch["audio"]
        z, q_outputs, c = self.forward(x)
        commit_loss = 0.0
        z_hat, indices = q_outputs
        
        l2_loss = torch.sum((z.detach() - z_hat)**2)
        commit_loss = torch.sum((z - z_hat.detach())**2)
        
        # compute wav2vec loss
        pos_loss, neg_loss, w2v_loss = self.w2v_loss_fn(z, c)
        
        total_loss = w2v_loss + l2_loss +  commit_loss
        
        self.log_dict({"train_loss": total_loss, "l2_loss": l2_loss, 
                       "commit_loss": commit_loss, "w2v_loss": w2v_loss, }, prog_bar=True)

        return total_loss
    
    def train_kmeans(self, batch: torch.tensor, batch_idx):
        """Use k means to determine corresponding code to encoder output, and \
            copy gradient from decoder output to encoder input. 

        Args:
            batch (torch.tensor): A single training batch of bs x frames
            batch_idx (torch.tensor): batch index
        """
        x = batch["audio"]
        
        opt = self.optimizers()
        opt.zero_grad()
        
        z = self.encoder(x)
        z_hat, indices = self.quantizer(z)
        decoder_input = z_hat.clone().detach().requires_grad_(True)
        c = self.context(decoder_input)
        
        # compute wav2vec loss
        pos_loss, neg_loss, w2v_loss = self.w2v_loss_fn(z, c)
        w2v_loss.backward(retain_graph=True)  # Gradients flow up to z_hat
        
        # Get gradients of loss w.r.t. decoder_input
        decoder_input_grad = decoder_input.grad.clone()
        
        # Manually copy gradients to the encoder's output
        if z.grad is None:
            z.grad = decoder_input_grad
        else:
            z.grad = z.grad + decoder_input_grad
        
        l2_loss = torch.sum((z.detach() - z_hat)**2)
        commit_loss = torch.sum((z - z_hat.detach())**2)
        
        auxilliary_loss = l2_loss +  self.commitment_weight*commit_loss
        self.manual_backward(auxilliary_loss)
        
        total_loss = w2v_loss + auxilliary_loss
        
        opt.step()
        
        self.log_dict({"train_loss": total_loss, "l2_loss": l2_loss, 
                       "commit_loss": commit_loss, "w2v_loss": w2v_loss, }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        z, q_outputs, c = self.forward(x)

        z_hat, indices = q_outputs
        
        # compute l2 loss and commitment loss
        l2_loss = torch.sum((z.detach() - z_hat)**2)
        commit_loss = torch.sum((z - z_hat.detach())**2)
        
        # compute wav2vec loss
        pos_loss, neg_loss, w2v_loss = self.w2v_loss_fn(z, c)
        
        val_total_loss = w2v_loss + l2_loss + commit_loss
        
        self.log_dict({"val_loss": val_total_loss, "val_commit_loss": commit_loss,
                        "val_w2v_loss": w2v_loss, "val_l2_loss": l2_loss}, prog_bar=True)

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
    params = VQ_Wav2vecHyperParam()
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    enc = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=params.dropout_prob, w2v_large=True)
    context = ContextNetwork(12, [(i, 1) for i in range(2, 14)], 
                             dropout_prob=params.dropout_prob, w2v_large=True)
    
    vq = VQ(codebook_size=params.codebook_size,
            codebook_dim=params.feat_dim,
            num_groups=params.num_groups,
            share_codebook_variables=params.share_codebook_variables,
            use_gumbel=params.use_gumbel,
            params=params,)

    vq_w2v_model = VQ_Wav2VecFeatureExtractor(encoder, context, vq, params=params)

    feat_enc, q_outputs, feat_context = vq_w2v_model(x)
    quantized, indices, commit_loss = q_outputs
    

    
    
        