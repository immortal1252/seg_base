from torch import nn
import torch


class Unet(nn.Module):
    def __init__(self, channels: list, in_channels=1, out_channels=1, act="ReLU"):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        self.encoder = nn.ModuleList()
        for channel in channels:
            block = nn.Sequential(
                ConvBNAct(in_channels, channel, 3, 1, act),
                ConvBNAct(channel, channel, 3, 1, act)
            )
            self.encoder.append(block)
            in_channels = channel

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
        for i, decoder in enumerate(self.decoder):
            x = self.upsamples[i](x)
            passx = pass_through.pop()
            x = torch.cat([x, passx], dim=1)
            x = decoder(x)

        x = self.out(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels, r=2):
        super().__init__()
        self.convx = ConvBNAct(in_channels, in_channels // r, 1, 1, "")
        self.convskip = ConvBNAct(in_channels, in_channels // r, 1, 1, "")

        self.conv = ConvBNAct(in_channels // r, in_channels, 1, 1, "Sigmoid")

    def forward(self, x, skip):
        # x和skip的长宽一样
        pass
        x = self.convx(x)
        skip_new = self.convskip(skip)
        merge = torch.relu(x + skip_new)
        merge = self.conv(merge)
        return merge * skip


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, act="ReLU"):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if hasattr(nn, act):
            self.act = getattr(nn, act)()


class Upsample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
