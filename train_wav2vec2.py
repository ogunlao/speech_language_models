import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.modules.encoder import Encoder
from model.wav2vec2 import Wav2Vec2FeatureExtractor
from model.utils.config import Wav2vec2HyperParam
from model.utils.loss import Wav2VecLoss

from model.modules.quantizer import VQ

from conformer import Conformer

import lightning as L
from datasets import load_dataset
    
    
# prepare dataset
train_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="train")
dev_dataset = load_dataset("Siyong/speech_timit", cache_dir="../data", split="test")

# create function to randomly crop audio 
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

    # normalise audio to have zero mean and unit variance
    means = torch.mean(audios, dim=1).reshape(-1, 1)
    stds = torch.std(audios, dim=1).reshape(-1, 1)

    audios = (audios - means) / stds
    
    return {"audio": audios.unsqueeze(1)}


# collect default parameters
params = Wav2vec2HyperParam()

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers,)
dev_loader = DataLoader(dev_dataset, batch_size=params.val_batch_size, 
                        shuffle=False, collate_fn=collate_fn,)

# model
encoder = Encoder(5, 512, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                  dropout_prob=params.dropout_prob,
                    w2v_large=True if params.model_name=="w2v_large" else False)
context = Conformer(
    dim = params.feat_dim,
    depth = params.num_layers,
    dim_head = params.dim_head,
    heads = params.heads,
    ff_mult = params.ff_expansion_size,
    conv_expansion_factor = params.conv_expansion_factor,
    conv_kernel_size = params.conv_kernel_size,
    attn_dropout = params.attn_dropout,
    ff_dropout = params.ff_dropout,
    conv_dropout = params.conv_dropout,
)

vq = VQ(codebook_size=params.codebook_size,
        codebook_dim=params.feat_dim,
        num_groups=params.num_groups,
        share_codebook_variables=params.share_codebook_variables,
        use_gumbel=params.use_gumbel,
        diversity_weight=params.diversity_weight,
        params=params,)

w2v2_model = Wav2Vec2FeatureExtractor(encoder, context, vq, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(w2v2_model, train_loader, dev_loader)
