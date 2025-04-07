'''
    Loads and preprocesses data
'''

# Args imports
import argparse
import random
from typing import Optional

from tqdm import tqdm

# Torch imports
import torch
from torch.utils.data import DataLoader #process data in batches

def load_data(parser, args):
    '''
        Input: CLI parser and argument prompts for dataset
        Outputs: train_data, test_data
    '''
    if args.dataset == 'musdb':
        
        parser.add_argument("--samples-per-track", type=int, default=64)
        parser.add_argument("--root", type=str, default=None)
        args = parser.parse_args()

        train_data = MUSDB(
            root = "D:/26 WAYNE/3.2/Deep Learning/acapella-splitter/musdb18",
            split ='train',
            target = 'vocals',
            as_wav = False,
            duration = 5.0,
            samples_per_track = args.samples_per_track
            )
        test_data = MUSDB(
            root = "D:/26 WAYNE/3.2/Deep Learning/acapella-splitter/musdb18",
            split = 'valid',
            samples_per_track = 1,
            duration = 5.0
            )
        return train_data, test_data, args
    elif args.dataset == 'sourcefolder':
        print('Not implemented- use MUSDB')
        return
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
            as_wav: bool = False,
            samples_per_track: int = 4,
            duration: float = 2.0, #TODO: add optional none type
            split: str='train',
            subsets: str='train',
            seed: int = 42,
    ):
        import musdb
        self.root = root
        self.seed = seed
        self.target = target
        self.as_wav = as_wav
        self.samples_per_track = samples_per_track
        self.duration = duration
        self.mus = musdb.DB(
            root = root,
            is_wav = as_wav,
            split=split,
            subsets=subsets
        )
        self.sample_rate = 44100.0      # sample rate of musdb
        self.num_samples = int(self.duration * self.sample_rate)

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

        return mixture, target

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
    
    train_data, test_data, args = load_data(parser, args)

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