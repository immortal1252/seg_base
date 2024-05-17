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
            num_layer=1,
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
        conv = []
        for i in range(num_layer):
            # 一般卷积核的参数
            act_t, kernel_size_t, stride_t, in_channels_t = act, 3, 1, out_channels
            # 等残差连接后再激活
            if i == num_layer - 1:
                act_t = ""
            ## 只有第一层，改变通道数以及尺寸
            if i == 0:
                stride_t = stride
                kernel_size_t = kernel_size
                in_channels_t = concat_channels

            block = ConvBNAct(in_channels_t, out_channels, kernel_size=kernel_size_t, stride=stride_t, act=act_t)
            conv.append(block)
        self.conv = nn.Sequential(*conv)

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
