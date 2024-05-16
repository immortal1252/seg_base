from typing import Union, Tuple

import einops as E

import spgutils.utils
from unet.blocks import *
from .cross_conv import CrossConv2d
from .vmap import Vmap, vmap


def as_2tuple(val: Union[int, Tuple]):
    if isinstance(val, int):
        return val, val
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


class CrossConv2dBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, act: str):
        super().__init__()

        self.cross_conv = CrossConv2d(
            two_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlin = get_act(act)

    def forward(self, u, v):
        interaction = self.cross_conv(u, v).squeeze(dim=1)
        interaction = vmap(self.bn, interaction)
        interaction = vmap(self.nonlin, interaction)
        u_avg = interaction.mean(dim=1, keepdim=True)
        return u_avg, interaction


class CrossBlock(nn.Module):
    def __init__(self, in_channels, cross_features, out_channels, act):
        super().__init__()

        self.c1 = CrossConv2dBNAct(in_channels, cross_features, 3, stride=1, act=act)
        self.c2 = Vmap(ConvBNAct(cross_features, out_channels, 3, act=act))
        self.c1_ = Vmap(ConvBNAct(cross_features, out_channels, 3, act=act))

    def forward(self, u, v):
        u, v = self.c1(u, v)
        u = self.c1_(u)
        v = self.c2(v)
        return u, v


class UniverSeg(nn.Module):
    def __init__(self, channels, in_channels=(3, 4), out_channels=1, act="LeakyReLU", init=True):
        super().__init__()
        self.channels = channels

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.first_conv = CrossConv2dBNAct(
            in_channels, 16, kernel_size=7, stride=2, act=act
        )
        in_channels = 16
        self.encoders = nn.ModuleList()
        # Encoder
        for channel in channels:
            block = CrossBlock(as_2tuple(in_channels), channel, channel, act)
            in_channels = channel
            self.encoders.append(block)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoders) - 2, -1, -1):
            self.upsamples.append(nn.UpsamplingBilinear2d(scale_factor=2))
            block = CrossBlock((2 * channels[i + 1], 2 * channels[i]), channels[i], channels[i], act)
            self.decoders.append(block)

        self.out_conv = ConvBNAct(channels[0], out_channels, kernel_size=1, act="")
        if init:
            if act == "LeakyReLU":
                spgutils.utils.init_params(self, "leaky_relu")
            elif act == "ReLU":
                spgutils.utils.init_params(self, "relu")
            else:
                raise Exception(f"no support init {act}")

    def forward(self, u, v, vy):
        """
        Args:
            u:  (b,c,h,w)
            v:  (b,s,c,h,w)
            vy: (b,s,1,h,w)
        """

        u = E.rearrange(u, f"B C H W -> B 1 C H W")
        v = torch.cat([v, vy], dim=2)
        u, v = self.first_conv(u, v)
        pass_through = []
        for i, encoder_block in enumerate(self.encoders):
            u, v = encoder_block(u=u, v=v)
            if i != len(self.channels) - 1:
                pass_through.append((u, v))
                u = vmap(self.downsample, u)
                v = vmap(self.downsample, v)

        for decoder_block in self.decoders:
            u_skip, v_skip = pass_through.pop()
            u = torch.cat([vmap(self.upsample, u), u_skip], dim=2)
            v = torch.cat([vmap(self.upsample, v), v_skip], dim=2)
            u, v = decoder_block(u, v)

        u = E.rearrange(u, "B 1 C H W -> B C H W")
        u = self.out_conv(u)
        u = self.upsample(u)

        return u
