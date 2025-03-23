'''
    Train model from CLI
'''
# Arg imports
import argparse
from pathlib import Path

# Torch imports
import torch
from torch.utils.data import DataLoader

# Internal imports
import data
import transforms

def train():
    # use the code with optimizers and loss
    return 0

def test():
    # use the same code but the test side
    return 0

def main():
    # CLI Configuration
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
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    args, _ = parser.parse_known_args() #only return dict

    # Initialize torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets from data
    train_data, test_data, args = data.load_data(parser, args)

    # Set output path
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it does not yet exist

    # Call dataloader from torch
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
    
    for x, y in loaders['train']:
        print(f"Mixture spectrogram shape: {x.shape}, Target spectrogram shape: {y.shape}")

    # send encoder to device
    # separator config?? 
    # initialize our model (use an if/else if someone's feeding in their model)
    # set optimizer to adam
    # checkpointing? or initialize arrays of losses and times, also start tqdm
    # call train and test for each epoch and append to array
    # then call utils.save_checkpoint() with state_dict()s
    # save params

if __name__ == "__main__":
    main()