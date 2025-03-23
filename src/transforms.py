'''
    Handles data transforms
'''
import torch
from torch import stft
from torch import nn

def stft_transform(waveform, n_fft=1024, hop_length=512, center = True):
    '''
        Converts a waveform to a spectrogram using Short Term Fourier Transform
        Inputs: 
            waveform (tensor), 
            n_fft (size of fourier transform), 
            hop_length (distance between neighboring sliding window frames)
            center: whether to pad input. defaults to true
    '''
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
    spectrogram = stft(
        waveform,
        n_fft = n_fft,
        hop_length = hop_length,
        center = center,
        return_complex = True
    )
    return spectrogram.abs() # Return magnitude spectrogram