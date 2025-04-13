'''
    Run source splitter on test data and generate results
    Inputs: 
    Outputs: 
'''
# Arg imports
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import tqdm as tq
import os

# Torch imports
import torch
import torchaudio
import torch.optim as optim
from torch.utils.data import DataLoader

# Internal imports
import data
import model
from model import load
from transforms import STFT
from utils import get_device, to_device

def main():
    # CLI Configuration
    parser = argparse.ArgumentParser(description="Source Separation")

    parser.add_argument('--dataset', 
                        type=str,
                        default='musdb',
                        choices=[
                            'musdb',
                            'sourcefolder' #debug: don't use this yet
                        ],
                        help='Name of dataset, or specify your own')
    parser.add_argument("--target",
                        type=str,
                        default="vocals",
                        help="Name of .pth file")
    parser.add_argument("--path",
                        type=str,
                        default="output/",
                        help="Path to .pth file")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate defaults 0.001")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    args, _ = parser.parse_known_args() #only return dict

    # Initialize torch
    device = get_device()

    # Load model
    unet, unet_args = load(args.path, args.target)
    to_device(unet, device)
    loss_fn = model.SDRLoss()

    # Load test audio
    test_data = data.load_test_data(parser, args, unet_args)

    test_duration = 0
    for i in tqdm(range(len(test_data)), desc="Loading datasets"): #progress marker
        (x_mag, x_phase), y_mag = test_data[i]
        test_duration += x_mag.shape[1] / test_data.sample_rate

    # Set output path
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it does not yet exist

    # Call dataloader from torch
    loaders = {"test": DataLoader(test_data, 
                    batch_size = 8, #formerly 100
                    shuffle = True, 
                    num_workers = 2), #for multi-core processor
            }
    
    # Separate Audio
    pbar = tqdm(loaders["test"], desc="Separating")
    test_losses = []
    stft_transform = STFT(n_fft = unet_args['args']['nfft'], hop_length=unet_args['args']['nhop'])

    for (x_mag, x_phase), y_mag in pbar:
        with torch.no_grad():
            x_mag, x_phase, y_mag = to_device(x_mag, device), to_device(x_phase, device), to_device(y_mag, device)
            y_pred = unet(x_mag)
            loss = loss_fn(y_pred, y_mag)
            test_losses.append(loss.item())
    
            # Reconstruct and save
            waveform = stft_transform.reconstruct_waveform(predicted_magnitude=y_pred.squeeze(1), original_phase=x_phase.squeeze(1))
            
            # Save output file
            output_path = os.path.join(output_dir, f"track_{i}.wav")
            torchaudio.save(output_path, waveform, sample_rate=44100)

            print(f"Saved {output_path}")

        print("Separation complete.")
    

if __name__ == "__main__":
    main()