import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.encoder import Encoder, ContextNetwork, Wav2VecLoss

import lightning as L
from datasets import load_dataset


class Wav2VecFeatureExtractor(L.LightningModule):
    def __init__(self, encoder: Encoder, context: ContextNetwork, 
                 k_steps: int, num_neg: int,):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.loss_fn = Wav2VecLoss(k_steps, num_neg)

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
        
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=1e-6,)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "train_loss",
            },
        }
    
    
# prepare dataset
train_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="train")
dev_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="test")


batch_size = 4
def collate_fn(data):
    audios, sentences = [], []
    min_len = 150_000 # max len of audio
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


train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, 
                        shuffle=False, collate_fn=collate_fn)

# model
encoder = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], w2v_large=False)
context = ContextNetwork(9, [(3, 1) for _ in range(9)], w2v_large=False)

w2v_model = Wav2VecFeatureExtractor(encoder, context, k_steps=12, num_neg=10)

# train model
trainer = L.Trainer(max_epochs=10, 
                    gradient_clip_val=4,
                    accumulate_grad_batches=16,
                    enable_checkpointing=True,
                    )
trainer.fit(w2v_model, train_loader, dev_loader)
