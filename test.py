import torch

from implement.wav2vec.model.modules.encoder import Encoder, ContextNetwork, Wav2VecFeatureExtractor

x = torch.rand(2, 1, 16000*5) # Two random noises of 5 seconds 
enc = Encoder(5, [(10, 5), (8, 4), (4, 2), (4, 2), (4, 2)], w2v_large=True)
context = ContextNetwork(9, [(3, 1) for _ in range(9)],)
# context = ContextNetwork(12, [(i, 1) for i in range(2, 14)], w2v_large=True)

w2v = Wav2VecFeatureExtractor(enc, context)
feat_enc, feat_context = w2v(x)

assert torch.equal(torch.tensor([2, 512, 498]), torch.as_tensor(feat_enc.shape))
assert torch.equal(torch.tensor([2, 512, 498]), torch.as_tensor(feat_context.shape))
print("All tests passed")