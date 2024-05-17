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
            act="ReLU"
    ):
        super().__init__()
        if isinstance(two_channels, (list, tuple)):
            concat_channels = sum(two_channels)
        else:
            raise Exception("need tuple")
        if concat_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = ConvBNAct(concat_channels, out_channels, 1, stride=stride, act="")
        # print(f"concat:{concat_channels},out:{out_channels}")
        self.conv = ConvBNAct(concat_channels, out_channels, kernel_size, stride=stride, act="")
        self.act = get_act(act)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: b,1,c1,h,w
            v: b,s,c2,h,w
        Returns:
            output: b 1 s co h w
        """
        B, *_ = u.shape
        _, S, *_ = v.shape
        # u = E.rearrange(u, "b s c h w -> (b s) c h w")
        # v = E.rearrange(v, "b s c h w -> (b s) c h w")
        # u = self.proj_u(u)
        # v = self.proj_v(v)

        # us = E.repeat(u, "(b sx) c h w -> b sx sy c h w", b=B, sx=Sx, sy=S)
        # vs = E.repeat(v, "(b sy) c h w -> b sx sy c h w", b=B, sx=Sx, sy=S)

        us = E.repeat(u, "b 1 c h w -> b 1 s c h w", b=B, s=S)
        vs = E.repeat(v, "b s c h w -> b 1 s c h w", b=B, s=S)
        xy = torch.cat([us, vs], dim=3)

        batched_xy = E.rearrange(xy, "b 1 s c h w -> (b s) c h w")
        batched_output = self.conv(batched_xy) + self.skip(batched_xy)
        batched_output = self.act(batched_output)

        output = E.rearrange(
            batched_output, "(b s) c h w -> b 1 s c h w", b=B, s=S
        )
        return output
