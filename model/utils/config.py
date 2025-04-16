from dataclasses import dataclass


@dataclass
class VQVAE_HyperParam:
    """Class for keeping track of training and model hyperparameters of wav2vec."""
    model_name: str = 'vq_vae'
    
    # encoder params
    num_conv: int = 6
    filter_size: int = 4
    stride: int = 2
    
    num_speakers: int = 1
    
    # trainer hyperparams
    train_batch_size: int = 2
    val_batch_size: int = 2
    epochs: int = 1000
    warmup_epochs: int = 10
    weight_decay: float = 0.0
    lr: float = 1e-5
    gradient_clip_val: int = 1
    grad_accum: int = 16
    check_val_every_n_epoch: int = 2
    num_workers: int = 8
    dropout_prob: float = 0.05
    
    # model params
    feat_dim: int = 512
    
    codebook_size: int = 320
    num_groups: str = 1 # 2
    commitment_weight: float = 0.25
    
    # gumbel params
    use_gumbel: bool = False # use kmeans or straight through estimator
    share_codebook_variables: bool = True # whether to share codebook for multiple groups
    
    # wav2vec training params
    max_batch_len: int = 150_000
    
    
@dataclass
class Wav2vecHyperParam:
    """Class for keeping track of training and model hyperparameters of wav2vec."""
    model_name: str = 'w2v_small'
    
    # trainer hyperparams
    train_batch_size: int = 4
    val_batch_size: int = 4
    epochs: int = 1000
    warmup_epochs: int = 10
    weight_decay: float = 0.0
    lr: float = 1e-5
    gradient_clip_val: int = 1
    grad_accum: int = 16
    check_val_every_n_epoch: int = 2
    num_workers: int = 8
    dropout_prob: float = 0.05
    
    # wav2vec training params
    k_steps: int = 12
    num_neg: int = 10
    max_batch_len: int = 150_000
    
    # model params
    feat_dim: int = 512
    commitment_weight: float = 0.25


@dataclass
class VQ_Wav2vecHyperParam(Wav2vecHyperParam):
    """Class for keeping track of training and model hyperparameters."""
    model_name: str = 'w2v_large'
    codebook_size: int = 320
    num_groups: str = 2 # 2
    commitment_weight: float = 0.25
    
    # gumbel params
    use_gumbel: bool = False # use kmeans or straight through estimator
    annealing_weight_start : float = 2. # 2 to 0.5
    annealing_weight_end: float = 0.5
    anneal_time: float = 0.7 # 70% of training
    share_codebook_variables: bool = True # whether to share codebook for multiple groups
    
    gradient_clip_val: int = 1 if use_gumbel else None # diable
    grad_accum: int = 16 if use_gumbel else 1 # disable
    
    # wav2vec training params
    k_steps: int = 8
    num_neg: int = 10
    max_batch_len: int = 150_000
    
