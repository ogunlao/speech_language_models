import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import bisect


class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 item_length,
                 target_length,
                #  file_location=None,
                 classes=256,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100,
                 num_speakers=1,):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

        self.dataset = dataset
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes

        self.mono = mono
        self.normalize = normalize

        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.start_samples = [0]
        self._length = 0
        self.num_speakers = num_speakers
        print("one hot input")
        # assign every *test_stride*th item to the test set

    def normalize_data(self, x):
        threshold = np.finfo(x.dtype).tiny
        mag = np.abs(x).astype(float)
        length = np.max(mag, axis=0, keepdims=True)
        
        # indices where norm is below the threshold
        small_idx = length < threshold

        xnorm = np.empty_like(x)
        length[small_idx] = 1.0
        xnorm[:] = x / length
        
        return xnorm

    def quantize_audio_file(self, idx):
        file_data = np.array(self.dataset[idx]["audio"]["array"])
        
        if len(file_data) < self._item_length:
            temp = np.zeros_like(self._item_length)
            temp[:len(file_data)] = file_data
            file_data = temp
        else:
            file_data = file_data[0: self._item_length + 1]
            
        file_data = self.normalize_data(file_data)
        quantized_audio = quantize_data(file_data, self.classes).astype(self.dtype)
        
        return file_data, quantized_audio

    def __getitem__(self, idx):
        
        audio, sample = self.quantize_audio_file(idx)
        
        # create one-hot vector for speaker
        speaker_emb = None
        if self.num_speakers > 1:
            if "speaker_id" in self.dataset[idx]:
                speaker_id = self.dataset[idx]["speaker_id"]
            
            if self.num_speakers > 1:
                speaker_emb = torch.zeros(self.num_speakers)
                speaker_emb[speaker_id] = 1.0
        
        # transfor data
        example = torch.from_numpy(sample).type(torch.LongTensor)
        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        one_hot.scatter_(0, example[:self._item_length].unsqueeze(0), 1.)
        target = example[-self.target_length:].unsqueeze(0)

        return (torch.tensor(audio, dtype=torch.float), torch.tensor(sample), 
                one_hot, target, speaker_emb)

    def __len__(self):
        return len(self.dataset)
    

def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s