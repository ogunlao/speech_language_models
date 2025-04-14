import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.encoder import Encoder, ContextNetwork, Wav2VecLoss
from model.hyperparam import VQ_Wav2vecHyperParam
from model.vq_wav2vec import VQ_Wav2VecFeatureExtractor

from vector_quantize_pytorch import VectorQuantize

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
encoder = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                    w2v_large=True if params.model_name=="w2v_large" else False)
context = ContextNetwork(9, [(3, 1) for _ in range(9)], 
                    w2v_large=True if params.model_name=="w2v_large" else False)
vq = VectorQuantize(
    dim = params.feat_dim,
    codebook_size = params.codebook_size,     # codebook size
    ema_update = False,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight = params.commitment_weight,   # the weight on the commitment loss
    learnable_codebook=True,
)
vq_w2v_model = VQ_Wav2VecFeatureExtractor(encoder, context, vq, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(vq_w2v_model, train_loader, dev_loader)
