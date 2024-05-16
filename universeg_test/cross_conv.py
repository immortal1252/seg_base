import einops as E
import torch
import torch.nn as nn
from unet.blocks import *


class CrossConv2d(nn.Module):

    def __init__(
            self,
            two_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias: bool = True,
    ):
        super().__init__()
        if isinstance(two_channels, (list, tuple)):
            concat_channels = sum(two_channels)
        else:
            raise Exception("need tuple")
        self.proj_u = ConvBNAct(two_channels[0], two_channels[0], 1)
        self.proj_v = ConvBNAct(two_channels[1], two_channels[1], 1)
        self.conv = nn.Conv2d(
            in_channels=concat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: b,s1,c1,h,w 
            v: b,s2,c2,h,w
        """
        B, Sx, *_ = u.shape
        _, Sy, *_ = v.shape
        u = E.rearrange(u, "b s c h w -> (b s) c h w")
        v = E.rearrange(v, "b s c h w -> (b s) c h w")
        u = self.proj_u(u)
        v = self.proj_v(v)

        us = E.repeat(u, "(b sx) c h w -> b sx sy c h w", b=B, sx=Sx, sy=Sy)
        vs = E.repeat(v, "(b sy) c h w -> b sx sy c h w", b=B, sx=Sx, sy=Sy)

        xy = torch.cat([us, vs], dim=3)

        batched_xy = E.rearrange(xy, "B Sx Sy C2 H W -> (B Sx Sy) C2 H W")
        batched_output = self.conv(batched_xy)

        output = E.rearrange(
            batched_output, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )
        return output
