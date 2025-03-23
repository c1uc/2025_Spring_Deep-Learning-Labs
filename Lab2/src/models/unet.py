# ref: https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()

        layers = [nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels)]

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        down_channels: list[int] = [64, 128, 256, 512],
        up_channels: list[int] = [1024, 512, 256, 128, 64],
        out_channels: int = 1,
    ):
        super(UNet, self).__init__()

        self.in_conv = DoubleConv(in_channels, down_channels[0])

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        for i in range(len(down_channels) - 1):
            self.down.append(DownConv(down_channels[i], down_channels[i + 1]))

        self.mid = DownConv(down_channels[-1], up_channels[0])

        for i in range(len(up_channels) - 1):
            self.up.append(UpConv(up_channels[i], up_channels[i + 1]))

        self.out = nn.Sequential(
            nn.Conv2d(up_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.in_conv(x)

        x_rec = [x]
        for down in self.down:
            x_rec.append(down(x))
            x = x_rec[-1]

        x = self.mid(x)

        for up in self.up:
            x = up(x, x_rec.pop())
        return self.out(x)


if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    model = DoubleConv(1, 64)
    x = model(x)
    print(x.shape)
    assert x.shape == torch.Size([1, 64, 256, 256])

    model = DownConv(64, 128)
    x_down = model(x)
    print(x_down.shape)
    assert x_down.shape == torch.Size([1, 128, 128, 128])

    model = UpConv(128, 64)
    x = model(x_down, x)
    print(x.shape)
    assert x.shape == torch.Size([1, 64, 256, 256])

    model = UNet()
    x = torch.randn(1, 1, 256, 256)
    x = model(x)
    print(x.shape)
    assert x.shape == torch.Size([1, 1, 256, 256])
