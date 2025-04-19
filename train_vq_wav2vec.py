import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.modules.encoder import Encoder, ContextNetwork
from model.utils.config import VQ_Wav2vecHyperParam
from model.vq_wav2vec import VQ_Wav2VecFeatureExtractor
from model.modules.quantizer import VQ

import lightning as L
from datasets import load_dataset
    
    
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
params = VQ_Wav2vecHyperParam()

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers,)
dev_loader = DataLoader(dev_dataset, batch_size=params.val_batch_size, 
                        shuffle=False, collate_fn=collate_fn,)

# model
encoder = Encoder(5, params.feat_dim, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)],
                  dropout_prob=params.dropout_prob,
                    w2v_large=True if params.model_name=="w2v_large" else False)
context = ContextNetwork(9, params.feat_dim, [(3, 1) for _ in range(9)], 
                         dropout_prob=params.dropout_prob,
                            w2v_large=True if params.model_name=="w2v_large" else False)

vq = VQ(codebook_size=params.codebook_size,
        codebook_dim=params.feat_dim,
        num_groups=params.num_groups,
        share_codebook_variables=params.share_codebook_variables,
        use_gumbel=params.use_gumbel,
        params=params,)

vq_w2v_model = VQ_Wav2VecFeatureExtractor(encoder, context, vq, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(vq_w2v_model, train_loader, dev_loader)
