import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchaudio import FrequencyMasking, TimeMasking, transforms

class SpecAug(nn.Module):
    def __init__(self, freq_len=1, time_len=10, freq_mask=27, time_mask=100):
        super(SpecAug, self).__init__()

        self.freq_len = freq_len
        self.time_len = time_len

        self.freq_mask = freq_mask
        self.time_mask = time_mask

        self.freq_aug = FrequencyMasking(freq_mask_param=self.freq_len)
        self.time_aug = TimeMasking(time_mask_param=self.time_len)
    
    def __call__(self, x):
        x = self.freq_aug(x, mask_value=self.freq_mask)
        x = self.time_aug(x, mask_value=self.time_mask)

        return x

train_audio_transforms = T.Compose([transforms.MelSpectrogram(sample_rate=16000, n_mels=128, normalized=True), SpecAug()])
valid_audio_transforms = transforms.MelSpectrogram(sample_rate=16000, n_mels=128, normalized=True)