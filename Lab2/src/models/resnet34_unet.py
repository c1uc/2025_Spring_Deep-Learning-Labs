import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UpConv


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down_sample: bool = False):
        super(ResBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2 if down_sample else 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    stride=2 if down_sample else 1,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if down_sample
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))

    @classmethod
    def make_layer(
        cls, in_channels: int, out_channels: int, blocks: int, down_sample: bool = False
    ):
        layers = [cls(in_channels, out_channels, down_sample)]
        for _ in range(1, blocks):
            layers.append(cls(out_channels, out_channels))
        return nn.Sequential(*layers)


class ResNet34Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        blocks: list[int] = [3, 4, 6, 3],
        down_channels: list[int] = [64, 64, 128, 256, 512],
        up_channels: list[int] = [1024, 512, 256, 128, 64],
    ):
        super(ResNet34Unet, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                down_channels[0],
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(down_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.down = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.down.append(
                ResBlock.make_layer(
                    down_channels[i],
                    down_channels[i + 1],
                    blocks[i],
                    down_sample=(down_channels[i] != down_channels[i + 1]),
                )
            )

        self.mid = ResBlock.make_layer(
            down_channels[-1],
            up_channels[0],
            1,
            down_sample=(down_channels[-1] != up_channels[0]),
        )

        self.up = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.up.append(UpConv(up_channels[i], up_channels[i + 1]))

        self.out = nn.Sequential(
            nn.ConvTranspose2d(
                up_channels[-1], up_channels[-1], kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(up_channels[-1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                up_channels[-1], up_channels[-1], kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(up_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(up_channels[-1], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.in_conv(x)
        x_rec = []
        for layer in self.down:
            x = layer(x)
            x_rec.append(x)
        x = self.mid(x)
        for layer in self.up:
            x = layer(x, x_rec.pop())
        return self.out(x)


if __name__ == "__main__":
    model = ResBlock(64, 64)
    x = torch.randn(1, 64, 256, 256)
    x = model(x)
    print(x.shape)
    assert x.shape == torch.Size([1, 64, 256, 256])

    model = ResBlock.make_layer(64, 128, 3, down_sample=True)
    x = torch.randn(1, 64, 256, 256)
    x = model(x)
    print(x.shape)
    assert x.shape == torch.Size([1, 128, 128, 128])
