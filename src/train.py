'''
    Train model from CLI commands
    Inputs: MUSDB18 dataset
    Arguments: dataset, target, nfft, nhop, lr, num-epochs, samples-per-track, root, is-wav
    Outputs: ML model in pth and chkpnt files
'''
# Arg imports
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import tqdm as tq
import json

# Torch imports
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Internal imports
import data
from transforms import STFT
import model
from utils import get_device, to_device

def train(unet, device, loader, loss_fn, optimizer):
    unet.train()
    losses = []
    pbar = tqdm(loader, desc="Training")
    for (x_mag, x_phase), y_mag in pbar:
        x_mag, y_mag = x_mag.to(device), y_mag.to(device)
        optimizer.zero_grad()
        #print(f"Input shape: {x_mag.shape}, Target shape: {y_mag.shape}")
        y_pred = unet(x_mag)
        #print(f"Pred shape: {y_pred.shape}")
        assert y_pred.shape == y_mag.shape, f"Network output mismatch: {y_pred.shape} vs {y_mag.shape}"
        loss = loss_fn(y_pred, y_mag)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pbar.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)

def valid(unet, device, loader, loss_fn):
    unet.eval()
    losses = []

    with torch.no_grad():
        for (x_mag, x_phase), y_mag in loader:
            x_mag, y_mag = x_mag.to(device), y_mag.to(device)
            y_pred = unet(x_mag)
            loss = loss_fn(y_pred, y_mag)
            losses.append(loss.item())

    return sum(losses) / len(losses)

def main():
    # CLI Configuration
    parser = argparse.ArgumentParser(description="Source Separation")

    # Parameters
    parser.add_argument('--dataset', 
                        type=str,
                        default='musdb',
                        choices=[
                            'musdb',
                            'sourcefolder' #debug: don't use this yet
                        ],
                        help='Name of dataset, or specify your own')
    
    # ML Paramaters
    parser.add_argument("--target",
                        type=str,
                        default="vocals",
                        help="Target source")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate defaults 0.001")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")

    args, _ = parser.parse_known_args() #only return dict

    # Initialize torch
    device = get_device()

    # Load datasets from data
    train_data, valid_data, args = data.load_train_data(parser, args)

    train_duration = 0
    for i in tqdm(range(len(train_data)), desc="Loading datasets"): #progress marker
        (x_mag, x_phase), y_mag = train_data[i]
        train_duration += x_mag.shape[1] / train_data.sample_rate #count length of clip trained

    # Set output path
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it does not yet exist

    # Call dataloader from torch
    loaders = {
        'train': DataLoader(train_data, 
                            batch_size = 8, #formerly 100
                            shuffle = True, 
                            num_workers = 2), #for multi-core processor
        'valid': DataLoader(valid_data, 
                            batch_size = 8, 
                            shuffle = True, 
                            num_workers = 2)
        }
    
    # Initialize model, loss, and optimizer
    unet = model.UNet().to(device)
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    loss_fn = model.SDRLoss()

    train_losses = []
    valid_losses = []

    tqdm_range = tq.trange(1, args.num_epochs + 1)

    # Training loop
    for epoch in tqdm_range:
        train_loss = train(unet, device, loaders["train"], loss_fn, optimizer)
        valid_loss = valid(unet, device, loaders["valid"], loss_fn)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        tqdm_range.set_postfix(train_loss=train_loss)
        
        model.save({"state_dict": unet.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict()},
                    path= output_dir,
                    target=args.target,
                        checkpoint=True)
    
    # Save model
    model.save({"state_dict": unet.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict()},
                path= output_dir,
                target=args.target,
                checkpoint=False)
    
    params = {
        "args": vars(args)
    }
    
    with open(Path(output_dir, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

if __name__ == "__main__":
    main()