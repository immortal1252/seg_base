from torch import nn
import torch


class ConvBNAct(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, dilation=1, stride=1, act="ReLU"
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_act(act)


class Upsample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


def get_act(act: str):
    if hasattr(nn, act):
        activation_class = getattr(nn, act)
        if 'inplace' in activation_class.__init__.__code__.co_varnames:
            instance = activation_class(inplace=True)
        else:
            instance = activation_class()
        return instance
    return nn.Identity()
