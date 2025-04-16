import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .modules.encoder import Encoder
from .modules.wavenet import WaveNet as Decoder
from .modules.quantizer import VQ
from .utils.config import VQVAE_HyperParam
from .utils.loss import VQVaeLoss

import lightning as L


class VQ_VAE(L.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 quantizer: VQ, params: VQVAE_HyperParam,):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.commitment_weight = params.commitment_weight
        self.params = params
        
        self.vae_loss_fn = VQVaeLoss()

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices = self.quantizer(z)
        
        return z, quantized, indices
    
    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        x_quantized = batch["audio_quantized"]
        speaker_emb = batch["speaker_emb"]
        
        z, quantized, indices = self.forward(x)
        
        l2_loss = F.mse_loss(z.detach(), quantized)
        commit_loss = F.mse_loss(z, quantized.detach())
        auxilliary_loss = l2_loss +  self.commitment_weight*commit_loss
        
        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        
        # decoder conditioned on speaker emb
        # x starts with a zero
        zero_frame = torch.zeros((x.shape[0], 1, 1))
        padded_x = torch.cat([zero_frame, x[:, :, :-1]], dim=2)
        
        x_hat = self.decoder(padded_x, quantized, speaker_emb)
        x_hat = self.decoder(x, quantized, speaker_emb)
        
        # compute wav2vec loss
        vae_loss = self.vae_loss_fn(x_hat, x_quantized)
        
        total_loss = vae_loss + auxilliary_loss
        
        self.log_dict({"train_loss": total_loss, "l2_loss": l2_loss, 
                       "commit_loss": commit_loss, "vae_loss": vae_loss, }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        x_quantized = batch["audio_quantized"]
        speaker_emb = batch["speaker_emb"]
        
        z, quantized, indices = self.forward(x)
        
        # compute l2 loss and commitment loss
        l2_loss = F.mse_loss(z.detach(), quantized)
        commit_loss = F.mse_loss(z, quantized.detach())
        auxilliary_loss = l2_loss +  self.commitment_weight*commit_loss
        
        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        
        # decoder conditioned on speaker emb
        # x starts with a zero
        zero_frame = torch.zeros((x.shape[0], 1, 1))
        padded_x = torch.cat([zero_frame, x[:, :, :-1]], dim=2)
        x_hat = self.decoder(padded_x, quantized, speaker_emb)
        
        # compute wav2vec loss
        vae_loss = self.vae_loss_fn(x_hat, x_quantized)
        
        val_total_loss = vae_loss + auxilliary_loss
        
        self.log_dict({"val_loss": val_total_loss, "val_commit_loss": commit_loss,
                        "val_vae_loss": vae_loss, "val_l2_loss": l2_loss}, prog_bar=True)
    
    def regenerate_audio(self, x, speaker_emb: torch.tensor | None):
        # x dim (B, 1, T)
        with torch.no_grad():
            z, quantized, indices = self.forward(x)
            
            # Straight Through Estimator
            quantized = z + (quantized - z).detach()
            
            # decoder conditioned on speaker emb
            # x starts with a zero
            zero_frame = torch.zeros((x.shape[0], 1, 1))
            padded_x = torch.cat([zero_frame, x[:, :, :-1]], dim=2)
            x_hat = self.decoder(padded_x, quantized, speaker_emb)
        
        return x_hat, quantized
        
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
    params = VQVAE_HyperParam()
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    
    # using wav2vec encoder network
    encoder = Encoder(6, params.feat_dim, [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=params.dropout_prob, w2v_large=True)
    
    decoder = Decoder()
    
    vq = VQ(codebook_size=params.codebook_size,
            codebook_dim=params.feat_dim,
            num_groups=params.num_groups,
            share_codebook_variables=params.share_codebook_variables,
            use_gumbel=params.use_gumbel,
            params=params,)

    vq_vae_model = VQ_VAE(encoder, decoder, vq, params=params)

    feat_enc, q_outputs, feat_recovered = vq_vae_model(x)
    quantized, indices, commit_loss = q_outputs

