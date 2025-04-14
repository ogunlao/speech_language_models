import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .encoder import Encoder, ContextNetwork, Wav2VecLoss
from .hyperparam import VQ_Wav2vecHyperParam
from vector_quantize_pytorch import VectorQuantize

import lightning as L


class VQ_Wav2VecFeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: ContextNetwork, 
                 quantizer: VectorQuantize, params: VQ_Wav2vecHyperParam,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.quantizer = quantizer
        self.params = params
        self.loss_fn = Wav2VecLoss(params.k_steps, params.num_neg)
        # self.commitment_weight = params.commitment_weight

    def forward(self, x):
        z = self.encoder(x)
        q_outputs = self.quantizer(z.transpose(2, 1))
        z_hat, indices, commit_loss = q_outputs
        c = self.context(z_hat.transpose(2, 1))
        
        return z, q_outputs, c
    
    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        z, q_outputs, c = self.forward(x)
        z_hat, indices, commit_loss = q_outputs
        
        l2_loss = torch.sum((z.detach() - z_hat.transpose(2, 1))**2)
        
        # compute wav2vec loss
        pos_loss, neg_loss, w2v_loss = self.loss_fn(z, c)
        
        total_loss = w2v_loss + l2_loss +  commit_loss
        
        self.log_dict({"train_loss": total_loss, "l2_loss": l2_loss, 
                       "commit_loss": commit_loss, "w2v_loss": w2v_loss, }, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["audio"]
        z, q_outputs, c = self.forward(x)
        z_hat, indices, commit_loss = q_outputs
        
        l2_loss = torch.sum(z.detach() - z_hat.transpose(2, 1))
        
        # compute wav2vec loss
        pos_loss, neg_loss, w2v_loss = self.loss_fn(z, c)
        
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
    enc = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], w2v_large=True)
    context = ContextNetwork(12, [(i, 1) for i in range(2, 14)], w2v_large=True)
    
    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,     # codebook size
        decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = params.co   # the weight on the commitment loss
    )
    vq_w2v_model = VQ_Wav2VecFeatureExtractor(encoder, context, vq, params=params)

    feat_enc, q_outputs, feat_context = vq_w2v_model(x)
    quantized, indices, commit_loss = q_outputs
    

    
    
        