'''
    Handles data transforms
'''
import torch
import torch.nn.functional as F
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
    def __init__(self, n_fft = 1024, hop_length = 256, center = True):
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
        complex_stft = stft(
            waveform,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.n_fft,
            center = self.center,
            window = self.window,
            return_complex = True,
            onesided = True
        )
        magnitude = complex_stft.abs()
        phase = complex_stft.angle()
        return magnitude, phase
    
    def reconstruct_waveform(self, predicted_magnitude, original_phase):
        # match frequency dimension to n_fft//2 + 1
        required_freq = self.n_fft // 2 + 1
        current_freq = predicted_magnitude.shape[-2]
        
        if current_freq < required_freq:
            # if too small, pad with zeroes
            pad_amount = required_freq - current_freq
            predicted_magnitude = F.pad(predicted_magnitude, (0, 0, 0, pad_amount))
            original_phase = F.pad(original_phase, (0, 0, 0, pad_amount))
        elif current_freq > required_freq:
            # if too large, truncate
            predicted_magnitude = predicted_magnitude[..., :required_freq, :]
            original_phase = original_phase[..., :required_freq, :]

        mag_cpu = predicted_magnitude.to('cpu')
        phase_cpu = original_phase.to('cpu')
        complex_stft = mag_cpu * torch.exp(1j * phase_cpu)

        '''
        Debug Prints
        print(f"Input magnitude shape: {predicted_magnitude.shape}")
        print(f"Input phase shape: {original_phase.shape}")
        print(f"Complex STFT shape: {complex_stft.shape}")
        '''

        if complex_stft.dim() == 4:
            # process each item in batch separately
            waveforms = []
            for i in range(complex_stft.shape[0]):
                wav = torch.istft(
                    complex_stft[i],  # shape set to [channels, freq, time]
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    center=self.center
                )
                waveforms.append(wav)
            return torch.stack(waveforms)
        else:
            # process one track (no batches)
            return torch.istft(
                complex_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=self.center
            )