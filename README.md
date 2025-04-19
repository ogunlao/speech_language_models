# Speech language models

This library tracks and implements speech language models, with particular focus on recent advances in speech langage modelling using acoustic codes.

The implementation are for research purposes and are intended to follow the papers that proposed them more closely.

## Models currently implemented

1. [vq-vae](https://arxiv.org/abs/1711.00937)(2017): Proposed vector quantization for acoustic code discovery using gumbel softmax for straight-through estimation of latents acoustic code. Also, showed that acoustic codes are closely related to phoneme categories.
1. [wav2vec](https://arxiv.org/abs/1904.05862): Proposed (contrastive) future time step prediction of actual tokens from negative samples.
1. [vq-wav2vec](https://arxiv.org/abs/1910.05453): Applied vector quantization to wav2vec and show promising results for speech language modelling (using masked language modelling), ASR, They also showed higher quality to compression ratio than conventional audio compression algotithms.
1. [wav2vec Discrete and Continuous BERT](): They compared using quantized speeech vs wav2vec features, Filterbanks, and MFCC, showing that quantized features perform better than continuous features for ASR.
1. [wav2vec2](https://arxiv.org/abs/2006.11477): Similar to vq-wav2vec, but with performance improvements and changes in training strategies. Transformer used as the context network to learn long-term dependencies. \i substituted transformers with Conformers in my implementation.

### Special acknowledgements

1. Pytorch lightning: used for multi-gpu training and inference
1. Tabisha: His implementation of the [WavNet vocoder](https://github.com/tabisheva/wavenet-vocoder) is adapted as the VQ-VAE decoder.
