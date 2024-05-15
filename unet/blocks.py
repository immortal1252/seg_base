from torch import nn
import torch


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
