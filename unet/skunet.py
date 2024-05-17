import einops
import torch.nn

from .blocks import *


class SKUNet(nn.Module):
    def __init__(self, channels: list, in_channels=1, out_channels=1, act="ReLU"):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.encoder = nn.ModuleList()
        for channel in channels:
            block = nn.Sequential(
                SKBlock(in_channels, channel, act),
                SKBlock(channel, channel, act),
            )
            self.encoder.append(block)
            in_channels = channel

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoder) - 2, -1, -1):
            self.upsamples.append(Upsample(channels[i + 1], channels[i]))
            block = nn.Sequential(
                SKBlock(channels[i] * 2, channels[i], act),
                SKBlock(channels[i], channels[i], act),
            )
            self.decoder.append(block)
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
            x = torch.cat([x, passx], dim=1)
            x = decoder(x)

        x = self.out(x)
        return x


class SKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act, r=32):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, dilation=2, act=act)
        self.conv2 = ConvBNAct(in_channels, out_channels, 3, dilation=1, act=act)
        d = max(32, out_channels // r)
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, d),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Linear(d, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        alpha = self.layer(x1 + x2)
        alpha = einops.rearrange(alpha, "b c -> b c 1 1")
        return alpha * x1 + (1 - alpha) * x2
