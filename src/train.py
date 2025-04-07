'''
    Train model from CLI
'''
# Arg imports
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import tqdm as tq

# Torch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Internal imports
import data
from transforms import STFT
import model

def train(unet, device, loader, loss_fn, optimizer):
    unet.train()
    losses = []
    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = unet(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pbar.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)

def valid(unet, device, loader, loss_fn):
    unet.eval()
    losses = []

    with torch.no_grad():
        for x, y, in loader:
            x, y = x.to(device), y.to(device)
            y_pred = unet(x)
            loss = loss_fn(y_pred, y)
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
    parser.add_argument("--target",
                        type=str,
                        default="vocals",
                        help="Target source")
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate defaults 0.001")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    args, _ = parser.parse_known_args() #only return dict

    # Initialize torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets from data
    train_data, test_data, args = data.load_data(parser, args)

    train_duration = 0
    for i in tqdm(range(len(train_data)), desc="Loading datasets"): #progress marker
        x, y = train_data[i]
        train_duration += x.shape[1] / train_data.sample_rate #count length of clip trained

    # Set output path
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it does not yet exist

    # Apply STFT transform
    stft_transform = STFT(n_fft = 1024, hop_length=512)
    train_data = [stft_transform(x) for x in tqdm(train_data, desc="Applying STFT")]
    test_data = [stft_transform(x) for x in tqdm(test_data, desc="Applying STFT")]

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
    
    # Initialize model, loss, and optimizer
    unet = model.UNet().to(device)
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    loss_fn = model.SDRLoss()

    train_losses = []
    valid_losses = []

    tqdm_range = tq.trange(1, args.num_epochs + 1)

    # Training loop
    for epoch in tqdm_range:
       # for x, y in loaders["train"]:
            #print("Input to UNet (x):", x)
            #print("Contains NaN:", torch.isnan(x).any())
        train_loss = train(unet, device, loaders["train"], loss_fn, optimizer)
        valid_loss = valid(unet, device, loaders["test"], loss_fn)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        tqdm_range.set_postfix(train_loss=train_loss, val_loss=valid_loss)
        
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

if __name__ == "__main__":
    main()