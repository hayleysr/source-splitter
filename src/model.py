# Torch imports
import torch
import torch.nn as nn
import torchaudio

# Util imports
import os
import json

# Internal imports
from utils import get_device, to_device

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        '''
            Define architecture
            Encoder uses 2D convolution layers with RELU and max pooling
            Decoder undoes this with 2D deconvolution
            Sequential feeds information from one step to the next- syntactic sugar to make 
                the forward function more readable
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=2, stride=2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
         with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class SDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SDRLoss, self).__init__()
        self.eps = eps # avoid div by 0

    def forward(self, pred, target):
        '''
            Define SDR loss
            Inputs:
                pred: predicted data, (batch, channels, time)
                target: actual data, (batch, channels, time)
            Outputs: SDR loss
            Formula: 10log_10(target^2/error^2)
        '''
        # match dimensions
        min_freq = min(pred.shape[-2], target.shape[-2])
        min_time = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_freq, :min_time]
        target = target[..., :min_freq, :min_time] 

        # calculate loss
        target_pow = target.pow(2).sum((1,2)) + self.eps
        error_pow = (target - pred).pow(2).sum((1,2)) + self.eps
        sdr = 10 * (target_pow / error_pow).log10()
        
        return -sdr.mean()

    
def save(
        state: dict, 
        path: str, 
        target: str,
        checkpoint: bool
        ):
    '''
        Input: state dict, path, target file name, if saved as checkpoint or path
        Outputs: .chkpnt or .pth file
    '''
    if checkpoint:
        torch.save(state, os.path.join(path, target + ".chkpnt"))
    torch.save(state["state_dict"], os.path.join(path, target + ".pth"))

def load(path: str, target: str):
    '''
        Input: path, target file name
        Outputs: pre-trained model, args
    '''
    device = get_device()

    model_path = os.path.join(path, target + ".pth")
    json_path = os.path.join(path, target + ".json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path ${model_path} does not exist in your files!")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Json path ${json_path} does not exist in your files!")
    
    with open(json_path, "r") as stream:
        json_args = json.load(stream)

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, json_args