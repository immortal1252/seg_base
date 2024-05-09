from typing import Optional

import einops as E
import torch
from torch import nn

from .cross_conv import CrossConv2d
from .vmap import Vmap, vmap
from typing import Union, Tuple

import spgutils.utils


def as_2tuple(val: Union[int, Tuple]):
    if isinstance(val, int):
        return val, val
    assert isinstance(val, (list, tuple)) and len(val) == 2
    return tuple(val)


def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


class Conv2dBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        nonlinearity: Optional[str],
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            padding_mode="zeros",
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlin = get_nonlinearity(nonlinearity)


class CrossConv2dBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        nonlinearity: Optional[str],
    ):
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
        self.nonlin = get_nonlinearity(nonlinearity)

    def forward(self, u, v):
        interaction = self.cross_conv(u, v).squeeze(dim=1)
        interaction = vmap(self.bn, interaction)
        interaction = vmap(self.nonlin, interaction)
        u_avg = interaction.mean(dim=1, keepdim=True)
        return u_avg, interaction


class CrossBlock(nn.Module):
    def __init__(self, in_channels, cross_features, out_channels, nonlinearity):
        super().__init__()

        self.c1 = CrossConv2dBNAct(in_channels, cross_features, 3, 1, nonlinearity)
        self.c2 = Vmap(Conv2dBNAct(cross_features, out_channels, 3, 1, nonlinearity))
        self.c1_ = Vmap(Conv2dBNAct(cross_features, out_channels, 3, 1, nonlinearity))

    def forward(self, u, v):
        u, v = self.c1(u, v)
        u = self.c1_(u)
        v = self.c2(v)
        return u, v


class UniverSeg(nn.Module):
    def __init__(
        self,
        encoder_blocks,
        decoder_blocks=None,
        in_ch=(3, 4),
        out_channels=1,
        nonlinearity="LeakyReLU",
        init=True,
    ):
        super().__init__()
        self.encoder_blocks = encoder_blocks

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.first_conv = CrossConv2dBNAct(
            in_ch, 16, kernel_size=7, stride=2, nonlinearity=nonlinearity
        )
        encoder_blocks = list(map(as_2tuple, encoder_blocks))
        decoder_blocks = decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        in_ch = 16
        # Encoder
        skip_outputs = []
        for cross_ch, out_ch in encoder_blocks:
            block = CrossBlock(as_2tuple(in_ch), cross_ch, out_ch, nonlinearity)
            in_ch = out_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, out_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(
                as_2tuple(in_ch + skip_ch), cross_ch, out_ch, nonlinearity
            )
            in_ch = out_ch
            self.dec_blocks.append(block)

        self.out_conv = Conv2dBNAct(
            in_ch,
            out_channels,
            1,
            1,
            nonlinearity=None,
        )
        if init:
            if nonlinearity == "LeakyReLU":
                spgutils.utils.init_params(self, "leaky_relu")
            elif nonlinearity == "ReLU":
                spgutils.utils.init_params(self, "relu")
            else:
                raise Exception(f"no support init {nonlinearity}")

    def forward(self, u, v, vy):
        """
        Args:
            u:  (b,c,h,w)
            v:  (b,s,c,h,w)
            vy: (b,s,h,w)
        """

        u = E.rearrange(u, f"B C H W -> B 1 C H W")
        v = torch.cat([v, vy], dim=2)
        u, v = self.first_conv(u, v)
        pass_through = []
        for i, encoder_block in enumerate(self.enc_blocks):
            u, v = encoder_block(u=u, v=v)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((u, v))
            u = vmap(self.downsample, u)
            v = vmap(self.downsample, v)

        for decoder_block in self.dec_blocks:
            u_skip, v_skip = pass_through.pop()
            u = torch.cat([vmap(self.upsample, u), u_skip], dim=2)
            v = torch.cat([vmap(self.upsample, v), v_skip], dim=2)
            u, v = decoder_block(u, v)

        u = E.rearrange(u, "B 1 C H W -> B C H W")
        u = self.out_conv(u)
        u = self.upsample(u)

        return u
