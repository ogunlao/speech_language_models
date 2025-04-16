# Speech language models

This library tracks and implements speech language models, with particular focus on recent advances in speech langage modelling using acoustic codes.

The implementation are for research purposes and are intended to follow the papers that proposed them more closely.

## Models currently implemented

1. [vq-vae](https://arxiv.org/abs/1711.00937)(2017): Proposed vector quantization for acoustic code discovery using gumbel softmax for straight-through estimation of latents. Learned acoustic codes using a WavNet vocoder.
1. [wav2vec](https://arxiv.org/abs/1904.05862): Proposed (contrastive) future time step prediction of actual tokens from negative samples
1. [vq-wav2vec](https://arxiv.org/abs/1910.05453): Applied vector quantization to wav2vec and show promising results for speech language modelling (using masked language modelling) and ASR

### Special acknowledgements

1. Tabisha: His implementation of the [WavNet vocoder](https://github.com/tabisheva/wavenet-vocoder) is adapted as the VQ-VAE decoder.
