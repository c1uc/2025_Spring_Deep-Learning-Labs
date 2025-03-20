# ref: https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
    

class DounConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DounConv, self).__init__()
        
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        ]

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        ]

        self.nn = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.nn(x)


if __name__ == '__main__':
    x = torch.randn(1, 1, 572, 572)
    model = DoubleConv(1, 64)
    print(model(x).shape)
    assert model(x).shape == torch.Size([1, 64, 568, 568])