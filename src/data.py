'''
    Loads and preprocesses data
'''

# Args imports
import argparse
import random
from typing import Optional

from tqdm import tqdm

import os
from dotenv import load_dotenv

import numpy as np

# Torch imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader #process data in batches

# Internal imports
from transforms import STFT

def load_train_data(parser, args):
    '''
        Input: CLI parser and argument prompts for dataset
        Outputs: train_data, valid_data
    '''
    if args.dataset == 'musdb':
        
        parser.add_argument("--samples-per-track", type=int, default=64)
        parser.add_argument("--root", type=str, default=None)
        parser.add_argument("--is-wav", type=bool, default=False)
        parser.add_argument("--nfft", type=int, default=1024, help="STFT fft size and window size")
        parser.add_argument("--nhop", type=int, default=512, help="STFT hop size")
        args = parser.parse_args()

        load_dotenv()

        train_data = MUSDB(
            root = os.getenv('MUSDB_PATH'),
            split ='train',
            target = 'vocals',
            is_wav = args.is_wav,
            duration = 5.0,
            samples_per_track = args.samples_per_track,
            nfft = args.nfft,
            nhop = args.nhop
            )
        valid_data = MUSDB(
            root = os.getenv('MUSDB_PATH'),
            split = 'valid',
            samples_per_track = 1,
            is_wav = args.is_wav,
            duration = 5.0,
            nfft = args.nfft,
            nhop = args.nhop
            )
        return train_data, valid_data, args
    elif args.dataset == 'sourcefolder':
        raise NotImplementedError
    else:
        print('Invalid command')
        return
    
def load_test_data(parser, args, unet_args):
    '''
        Input: CLI parser and argument prompts for dataset
        Outputs: test_data
    '''
    if args.dataset == 'musdb':
        test_data = MUSDB(
            root = "D:/26 WAYNE/3.2/Deep Learning/acapella-splitter/musdb18",
            subsets ='test',
            target = 'vocals',
            is_wav = unet_args['args']['is_wav'],
            duration = 5.0,
            samples_per_track = unet_args['args']['samples_per_track'],
            nfft = unet_args['args']['nfft'],
            nhop = unet_args['args']['nhop'],
            split = None
            )
        return test_data
    elif args.dataset == 'sourcefolder':
        raise NotImplementedError
    else:
        print('Invalid command')
        return
    
class MUSDB():
    '''
        Initialize MusDB Database Object
        TODO: Integrate all customizations given in the musdb.DB object
    '''
    def __init__(
            self,
            root: str = '/musdb18',
            target: str = 'vocals',
            is_wav: bool = False,
            samples_per_track: int = 4,
            duration: float = 2.0, #TODO: add optional none type
            split: str='train',
            subsets: str='train',
            seed: int = 42,
            nfft: int = 2048,
            nhop: int = 512
    ):
        import musdb
        self.root = root
        self.seed = seed
        self.target = target
        self.is_wav = is_wav
        self.samples_per_track = samples_per_track
        self.duration = duration
        self.mus = musdb.DB(
            root = root,
            is_wav = is_wav,
            split=split,
            subsets=subsets
        )
        self.sample_rate = 44100.0      # sample rate of musdb
        self.num_samples = int(self.duration * self.sample_rate)
        self.nfft = nfft
        self.nhop = nhop

        self.mus.tracks = self.mus.tracks[:1]  #limit to 1 track for testing

    '''
    TODO: reference open-unmix's difference between train and test sets for refining
    '''
    def __getitem__(self, index):
        track_index = index // self.samples_per_track

        # select track at track_index
        track = self.mus.tracks[track_index]

        mixture = track.audio.T #transposed track. Shape: (2 channels, num_samples)
        target = track.targets[self.target].audio.T #transposed target track (ex, vocals for this track). Shape: (2 channels, num_samples)

        # pick a starting point between start and length of the clip
        start = random.randint(0, mixture.shape[1] - self.num_samples)

        # splice data to include only the length of the clip
        mixture = mixture[:, start:start + self.num_samples]
        target = target[:, start:start + self.num_samples]

        # convert to tensors
        mixture = torch.tensor(mixture, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # force length to be divisible by hop_length
        target_length = (mixture.shape[1] // self.nhop) * self.nhop
        mixture = mixture[:, :target_length]
        target = target[:, :target_length]

        # pad to nearest power-of-two (helps UNet)
        def adjust_length(x, n_fft, hop_length):
            # Ensure length is divisible by hop_length
            length = (x.shape[1] // hop_length) * hop_length
            x = x[:, :length]
            
            # Pad to next multiple of n_fft for clean STFT
            pad_len = (n_fft - (length % n_fft)) % n_fft
            return F.pad(x, (0, pad_len))
        
        
        mixture = adjust_length(mixture, self.nfft, self.nhop)
        target = adjust_length(target, self.nfft, self.nhop)

        # apply STFT
        stft_transform = STFT(n_fft = self.nfft, hop_length=self.nhop)
        mixture_mag, mixture_phase = stft_transform(mixture)
        target_mag, _ = stft_transform(target)

        # truncate time to multiples of 4
        mixture_time = mixture_mag.shape[-1]
        target_time = target_mag.shape[-1]
        min_time = min(mixture_time, target_time)
        min_time = min_time - (min_time % 4)
        mixture_mag = mixture_mag[..., :min_time]
        mixture_phase = mixture_phase[..., :min_time]
        target_mag = target_mag[..., :min_time]

        # Ensure frequency dimension is even
        min_freq = min(mixture_mag.shape[-2], target_mag.shape[-2])
        min_freq = min_freq if min_freq % 2 == 0 else min_freq - 1  # Truncate odd to even
        min_time = min(mixture_mag.shape[-1], target_mag.shape[-1])
        
        mixture_mag = mixture_mag[..., :min_freq, :min_time]
        mixture_phase = mixture_phase[..., :min_freq, :]
        target_mag = target_mag[..., :min_freq, :min_time]
        
        assert mixture_mag.shape == target_mag.shape, f"Shape mismatch: {mixture_mag.shape} vs {target_mag.shape}"

        return (mixture_mag, mixture_phase), target_mag 

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


if __name__ == '__main__':
    '''
        Function calls to load dataset
    '''

    # CLI Parser
    parser = argparse.ArgumentParser(description="Source Separation")
    # Parameters: Which dataset to train with
    parser.add_argument('--dataset', 
                        type=str,
                        default='musdb',
                        choices=[
                            'musdb',
                            'sourcefolder' #debug: don't use this yet
                        ],
                        help='Name of dataset, or specify your own')
    args, _ = parser.parse_known_args() #only return dict
    
    train_data, test_data, args = load_train_data(parser, args)

    train_duration = 0
    for i in tqdm(range(len(train_data))): #progress marker
        x, y = train_data[i]
        train_duration += x.shape[1] / train_data.sample_rate #count length of clip trained
    
    print("Total training duration (h): ", train_duration / 3600)
    print("Number of train samples: ", len(train_data))
    print("Number of validation samples: ", len(test_data))

    loaders = {
    'train': DataLoader(train_data, 
                        batch_size = 8, #formerly 100
                        shuffle = True, 
                        num_workers = 2), #for multi-core processor
    'test': DataLoader(test_data, 
                       batch_size = 8, 
                       shuffle = True, 
                       num_workers = 2)
    }

    for x, y in tqdm(loaders['train']):
        print(x.shape)