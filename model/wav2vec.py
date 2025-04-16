import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .modules.encoder import Encoder, ContextNetwork
from .utils.config import Wav2vecHyperParam
from .utils.loss import Wav2VecLoss

import lightning as L


class Wav2VecFeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: ContextNetwork, 
                 params: Wav2vecHyperParam,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.params = params
        self.loss_fn = Wav2VecLoss(params.k_steps, params.num_neg, params.feat_dim)

    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)
        return z, c
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch["audio"]
        z, c = self.forward(x)
        
        pos_loss, neg_loss, total_loss = self.loss_fn(z, c)
        self.log_dict({"pos_loss": pos_loss, "neg_loss": neg_loss,
                        "train_loss": total_loss,}, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch["audio"]
        z, c = self.forward(x)
        
        pos_loss, neg_loss, total_loss = self.loss_fn(z, c)
        self.log_dict({"val_pos_loss": pos_loss, "val_neg_loss": neg_loss,
                        "val_loss": total_loss,})

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
    params = Wav2vecHyperParam()
    
    x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
    enc = Encoder(5, 512, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=params.dropout_prob, w2v_large=False)
    context = ContextNetwork(9, 512, [(3, 1) for _ in range(9)], dropout_prob=params.dropout_prob)
    # context = ContextNetwork(12, 512, [(i, 1) for i in range(2, 14)], 
    #                           dropout_prob=params.dropout_prob, w2v_large=True)
    

    w2v = Wav2VecFeatureExtractor(enc, context, params)
    feat_enc, feat_context = w2v(x)