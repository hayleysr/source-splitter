'''
    Handles data transforms
'''
import torch
from torch import stft

class STFT:
    '''
        Converts a waveform to a spectrogram using Short Term Fourier Transform
        Inputs: 
            waveform (tensor), 
            n_fft (size of fourier transform), 
            hop_length (distance between neighboring sliding window frames)
            center: whether to pad input. defaults to true
    '''
    def __init__(self, n_fft = 1024, hop_length = 512, center = True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, waveform):
        if isinstance(waveform, tuple): # Handle multiple tensors
            mixture, target = waveform
            return self._compute_stft(mixture), self._compute_stft(target)
        return self._compute_stft(waveform) # Handle single tensor
    
    def _compute_stft(self, waveform):
        return stft(
            waveform,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = self.center,
            window = self.window,
            return_complex = True
        )