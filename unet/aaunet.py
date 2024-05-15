import torch
from torch import nn
import einops as E
from .blocks import *


class AAUNet(nn.Module):
    def __init__(self, channels: list, in_channels=1, out_channels=1, act="ReLU"):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        for channel in channels:
            self.encoder.append(HAAM(in_channels, channel, act))
            in_channels = channel

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # channels[-2::-1]
        for i in range(len(channels) - 2, -1, -1):
            self.upsamples.append(Upsample(channels[i + 1], channels[i]))
            self.decoder.append(HAAM(channels[i] * 2, channels[i], act))

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        pass_through = []
        # b,c,h,w
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            if i != len(self.encoder) - 1:
                pass_through.append(x)
                x = self.downsample(x)
        for i, decoder in enumerate(self.decoder):
            x = self.upsamples[i](x)
            passx = pass_through.pop()
            x = torch.cat([x, passx], dim=1)
            x = decoder(x)

        x = self.out(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.BatchNorm1d(in_channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = x1 + x2
        # b,c,h,w
        alpha = self.layer(x)
        # b,c
        alpha = E.rearrange(alpha, "b c -> b c 1 1")
        return alpha * x1, (1 - alpha) * x2


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, in_channels, 1, 1)
        self.conv2 = ConvBNAct(in_channels, in_channels, 1, 1)
        self.layer = ConvBNAct(in_channels, 1, 1, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.relu(x1 + x2)
        x = self.layer(x)
        beta = torch.sigmoid(x)
        x1 = beta * x1
        x2 = beta * x2
        return x1, x2


class HAAM(nn.Module):
    # Hybrid adaptive attention module
    def __init__(self, in_channels, out_channels, act="LeakyReLU"):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, 1, act)
        self.conv2 = ConvBNAct(in_channels, out_channels, 5, 1, act)
        self.conv3 = ConvBNAct(in_channels, out_channels, 3, 3, act)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(out_channels)
        self.out = ConvBNAct(out_channels, out_channels, 1, 1, act)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x2, x3 = self.ca(x2, x3)
        x23 = x2 + x3
        x1, x23 = self.sa(x1, x23)

        out = self.out(x1 + x23)
        return out
