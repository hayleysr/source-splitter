import torch
import torch.nn as nn
import torchaudio

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        '''
            Define architecture
            Encoder uses 2D convolution layers with RELU and max pooling
            Decoder undoes this with 2D deconvolution
            Sequential feeds information from one step to the next- syntactic sugar to make 
                the forward function more readable
            FIXME: Assumes stereo audio (2 channels in)
            FIXME: Consider using max unpooling? Probably not, if we want to reference unet stuff.
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x