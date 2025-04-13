import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.encoder import Encoder, ContextNetwork, Wav2VecLoss
from model.hyperparam import Wav2vecHyperParam

import lightning as L
from datasets import load_dataset


class Wav2VecFeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: ContextNetwork, 
                 params: Wav2vecHyperParam,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.params = params
        self.loss_fn = Wav2VecLoss(params.k_steps, params.num_neg)

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
    
    
# prepare dataset
train_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="train")
dev_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="test")


def collate_fn(data):
    audios, sentences = [], []
    min_len = params.max_batch_len # max len of audio
    for item in data:
        audio = torch.tensor(item["audio"]["array"])
        min_len = min(min_len, audio.shape[0])
        audios.append(audio)
    
    new_audios = []
    for audio in audios:
        # randomly decide to crop in front or end of audio
        if torch.rand(1) <= 0.5:
            new_audios.append(audio[:min_len].unsqueeze(0))
        else:
            new_audios.append(audio[-1*min_len:].unsqueeze(0))
    
    audios = torch.cat(new_audios, dim=0)
    return {"audio": audios.unsqueeze(1)}


# collect default parameters
params = Wav2vecHyperParam()

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers,)
dev_loader = DataLoader(dev_dataset, batch_size=params.val_batch_size, 
                        shuffle=False, collate_fn=collate_fn,)

# model
encoder = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                    w2v_large=True if params.model_name=="w2v_large" else False)
context = ContextNetwork(9, [(3, 1) for _ in range(9)], 
                    w2v_large=True if params.model_name=="w2v_large" else False)

w2v_model = Wav2VecFeatureExtractor(encoder, context, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(w2v_model, train_loader, dev_loader)
