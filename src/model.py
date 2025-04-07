# Torch imports
import torch
import torch.nn as nn
import torchaudio

# Util imports
import os

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
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=2, stride=2, output_padding=1)
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
        '''
        print("Input:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())
    
        x = self.encoder[0](x)  # First Conv2d
        conv1 = self.encoder[0]
        print("Conv1 Weights Min:", conv1.weight.min().item(), "Max:", conv1.weight.max().item())
        print("After Conv1:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())
        
        x = self.encoder[1](x)  # ReLU
        print("After ReLU1:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.encoder[2](x)  # MaxPool
        print("After Pool1:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.encoder[3](x)  # Second Conv2d
        print("After Conv2:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.encoder[4](x)  # ReLU
        print("After ReLU2:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.encoder[5](x)  # MaxPool
        print("After Pool2:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.decoder[0](x)  # First ConvTranspose2d
        print("After Deconv1:", x.shape, "NaNs:", torch.isnan(x).any().item(),"Min:", x.min().item(), "Max:", x.max().item())

        x = self.decoder[1](x)  # ReLU
        print("After ReLU3:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        x = self.decoder[2](x)  # Second ConvTranspose2d
        print("After Deconv2:", x.shape, "NaNs:", torch.isnan(x).any().item(), "Min:", x.min().item(), "Max:", x.max().item())

        return x
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class SDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SDRLoss, self).__init__()
        self.eps = eps # Avoids div by 0

    def forward(self, pred, target):
        '''
            Define SDR loss
            Inputs:
                pred: predicted data, (batch, channels, time)
                target: actual data, (batch, channels, time)
            Outputs: SDR loss
            Formula: 10log_10(target^2/error^2)
        '''
        # Match dimensions
        min_freq_dim = min(target.shape[1], pred.shape[1])  
        min_time_dim = min(target.shape[-1], pred.shape[-1])  

        target = target[:, :min_freq_dim, :min_time_dim]  
        pred = pred[:, :min_freq_dim, :min_time_dim]  

        # Calculate loss
        numerator = torch.sum(pow(target, 2), dim=(1,2))
        denominator = torch.sum(pow((target - pred), 2), dim=(1,2))
        sdr = 10 * torch.log10(numerator / (denominator + self.eps))
        #if torch.isnan(-1 * torch.mean(sdr)):
            #print(f"Not a number! Numerator: {numerator}, Denominator: {denominator}")
        return -1 * torch.mean(sdr)

    
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