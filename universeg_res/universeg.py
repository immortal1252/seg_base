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


class CrossBlock(nn.Module):
    def __init__(self, in_channels, cross_features, out_channels, kernel_size=3, stride=1, num_layer=1, act="ReLU"):
        super().__init__()

        self.c1 = CrossConv2d(in_channels, cross_features, kernel_size=kernel_size, stride=stride,
                              num_layer=num_layer,
                              act=act)
        self.c2 = Vmap(ConvBNAct(cross_features, out_channels, 3, act=act))
        self.c1_ = Vmap(ConvBNAct(cross_features, out_channels, 3, act=act))

    def forward(self, u, v):
        """
        Args:
            u: b,1,c1,h,w
            v: b,s,c2,h,w
        Returns:

        """
        interaction = self.c1(u, v).squeeze(1)  # b 1 s c0 h w -> b s c0 h w
        u_avg = interaction.mean(dim=1, keepdim=True)  # b 1 c0 h w
        u = self.c1_(u_avg)
        v = self.c2(interaction)
        return u, v


class UniverSeg(nn.Module):
    def __init__(self, channels, num_layers=None, in_channels=(1, 2), out_channels=1, act="LeakyReLU", init=True):
        super().__init__()
        self.channels = channels
        if num_layers is None:
            num_layers = [1] * len(channels)

        assert len(num_layers) == len(channels)
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.first_conv = CrossBlock(
            in_channels, 16, 16, kernel_size=7, stride=2, act=act
        )
        in_channels = 16
        self.encoders = nn.ModuleList()
        # Encoder
        for num_layer, channel in zip(num_layers, channels):
            block = CrossBlock(as_2tuple(in_channels), channel, channel, num_layer=num_layer, act=act)
            in_channels = channel
            self.encoders.append(block)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoders) - 2, -1, -1):
            self.upsamples.append(nn.UpsamplingBilinear2d(scale_factor=2))
            block = CrossBlock((channels[i + 1] + channels[i], channels[i + 1] + channels[i]), channels[i], channels[i],
                               act=act)
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
