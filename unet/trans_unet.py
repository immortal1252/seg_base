import torch
import einops as E
from .blocks import *


class TransUNet(nn.Module):
    def __init__(self, channels: list, num_layers, nhead, dim_feedforward, in_channels=1, out_channels=1, act="ReLU"):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        for channel in channels:
            block = nn.Sequential(
                ConvBNAct(in_channels, channel, 3, 1, act),
                ConvBNAct(channel, channel, 3, 1, act)
            )
            self.encoder.append(block)
            in_channels = channel

        layer = nn.TransformerEncoderLayer(in_channels, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.tranformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoder) - 2, -1, -1):
            self.upsamples.append(Upsample(channels[i + 1], channels[i]))
            block = nn.Sequential(
                ConvBNAct(channels[i] * 2, channels[i], 3, 1, act),
                ConvBNAct(channels[i], channels[i], 3, 1, act),
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
        b, c, h, w = x.shape
        x = E.rearrange(x, "b c h w -> b (h w) c")
        x = self.tranformer(x)
        x = E.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        for i, decoder in enumerate(self.decoder):
            x = self.upsamples[i](x)
            passx = pass_through.pop()
            x = torch.cat([x, passx], dim=1)
            x = decoder(x)

        x = self.out(x)
        return x
