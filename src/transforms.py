'''
    Handles transforms from imported audio to spectrogram
    Currently supports: musdb stems
'''

import torch
import torchaudio
import numpy as np
import argparse
import data

'''
waveform, sample_rate = torchaudio.load()
spectrogram = torch.stft(
    waveform, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None)
print(torch.__version__)
'''