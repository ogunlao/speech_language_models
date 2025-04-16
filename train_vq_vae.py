import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.modules.encoder import Encoder
from model.modules.wavenet import WaveNet as Decoder, quantize, encode_mu_law
from model.utils.config import VQVAE_HyperParam
from model.vq_vae import VQ_VAE
from model.modules.quantizer import VQ

import lightning as L
from datasets import load_dataset
    
from model.utils.dataset import WavenetDataset

# prepare dataset
train_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="train")
dev_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="test")


def collate_fn(data):
    audios, audio_qs, speaker_embs = [], [], []
    
    for item in data:
        audio = item["audio"]["array"]
        speaker_emb = item.get("speaker", None)
        position = np.random.choice(len(audio) - 20000)
        audio =  audio[position:position + 20000]
        audio = torch.tensor(audio).unsqueeze(0)
        audio_q = quantize(encode_mu_law(audio))
        
        audios.append(audio)
        audio_qs.append(audio_q)
        
        if speaker_emb is not None:
            speaker_embs.append(speaker_emb.unsqueeze(0))
    
    audios = torch.cat(audios, dim=0)
    audio_qs = torch.cat(audio_qs, dim=0)
    if speaker_embs:
        speaker_embs = torch.cat(speaker_embs, dim=0)
    else:
        speaker_embs = None
    
    return {"audio": audios.unsqueeze(1), 
            "audio_quantized": audio_qs.unsqueeze(1),
            "speaker_emb": speaker_embs,}


# collect default parameters
params = VQVAE_HyperParam()

decoder = Decoder(
    num_mels=params.feat_dim, kernel_size=2, residual_channels=120, skip_channels=240,
                 dilation_depth=8, dilation_repeat=2, quantized_values=256
    )

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers,)
dev_loader = DataLoader(dev_dataset, batch_size=params.val_batch_size, 
                        shuffle=False, collate_fn=collate_fn,)

# model
encoder = Encoder(params.num_conv, params.feat_dim, [(params.filter_size, params.stride) for i in range(params.num_conv)], 
                  dropout_prob=params.dropout_prob, w2v_large=True)

vq = VQ(codebook_size=params.codebook_size,
        codebook_dim=params.feat_dim,
        num_groups=params.num_groups,
        share_codebook_variables=params.share_codebook_variables,
        use_gumbel=params.use_gumbel,
        params=params,)

vq_vae_model = VQ_VAE(encoder, decoder, vq, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(vq_vae_model, train_loader, dev_loader)
