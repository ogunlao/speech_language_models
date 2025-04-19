import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from model.modules.encoder import Encoder, ContextNetwork
from model.vq_wav2vec import VQ_Wav2VecFeatureExtractor
from model.vq_w2v_discrete_bert import VQw2vDiscreteBert
from model.utils.config import VQ_Wav2vecHyperParam, VQ_w2v_DIscreteBERTHyperParam

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
params = VQ_w2v_DIscreteBERTHyperParam()

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                          shuffle=True, collate_fn=collate_fn, num_workers=params.num_workers,)
dev_loader = DataLoader(dev_dataset, batch_size=params.val_batch_size, 
                        shuffle=False, collate_fn=collate_fn,)


# first initialise a vq-wav2vec model
vq_w2v_params = VQ_Wav2vecHyperParam()
x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
vq_w2v_encoder = Encoder(5, params.feat_dim, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], 
                dropout_prob=vq_w2v_params.dropout_prob, w2v_large=True)
vq_w2v_context = ContextNetwork(12, params.feat_dim, [(i, 1) for i in range(2, 14)], 
                            dropout_prob=vq_w2v_params.dropout_prob, w2v_large=True)

vq = VQ(codebook_size=vq_w2v_params.codebook_size,
        codebook_dim=vq_w2v_params.feat_dim,
        num_groups=1,
        share_codebook_variables=vq_w2v_params.share_codebook_variables,
        use_gumbel=vq_w2v_params.use_gumbel,
        params=vq_w2v_params,)

feat_extractor = VQ_Wav2VecFeatureExtractor(vq_w2v_encoder, vq_w2v_context, 
                                            vq, vq_w2v_params)

# model
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

w2v2_model = VQw2vDiscreteBert(feat_extractor, context, params=params)

# train model
trainer = L.Trainer(max_epochs=params.epochs, 
                    gradient_clip_val=params.gradient_clip_val,
                    accumulate_grad_batches=params.grad_accum,
                    enable_checkpointing=True,
                    )
trainer.fit(w2v2_model, train_loader, dev_loader)
