import torch
from torch import nn
from .blocks import *


class AttentionUNet(nn.Module):
    def __init__(self, channels: list, in_channels=1, out_channels=1, act="ReLU"):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.encoder = nn.ModuleList()
        for channel in channels:
            block = nn.Sequential(
                ConvBNAct(in_channels, channel, 3, act=act),
                ConvBNAct(channel, channel, 3, act=act),
            )
            self.encoder.append(block)
            in_channels = channel

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.attn = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoder) - 2, -1, -1):
            self.upsamples.append(Upsample(channels[i + 1], channels[i]))
            block = nn.Sequential(
                ConvBNAct(channels[i] * 2, channels[i], 3, act=act),
                ConvBNAct(channels[i], channels[i], 3, act=act),
            )
            self.decoder.append(block)
            attn = AttentionGate(channels[i])
            self.attn.append(attn)

        self.downsample = nn.MaxPool2d(2, 2)
        self.out = nn.Conv2d(channels[0], out_channels, 1)

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
            passx = self.attn[i](x, passx)
            x = torch.cat([x, passx], dim=1)
            x = decoder(x)

        x = self.out(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels, r=2):
        super().__init__()
        self.convx = ConvBNAct(in_channels, in_channels // r, 1, act="")
        self.convskip = ConvBNAct(in_channels, in_channels // r, 1, act="")

        self.conv = ConvBNAct(in_channels // r, in_channels, 1, act="Sigmoid")

    def forward(self, x, skip):
        # x和skip的长宽一样
        pass
        x = self.convx(x)
        skip_new = self.convskip(skip)
        merge = torch.relu(x + skip_new)
        alpha = self.conv(merge)
        return alpha * skip
