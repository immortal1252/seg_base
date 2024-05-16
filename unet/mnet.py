from .blocks import *
import einops as E


class MNet(nn.Module):
    def __init__(self, channels: list, nhead=8, embeding_dim=2048, in_channels=1, out_channels=1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        # self.base_size = base_size
        self.encoder = nn.ModuleList()
        for i, channel in enumerate(channels):
            basic_block = BasicBlock(channel, nhead, embeding_dim)
            patch = 8 if i == 0 else 2
            block = PatchEmbed(in_channels, channel, patch, basic_block)
            self.encoder.append(block)
            in_channels = channel

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # [-2::-1]
        for i in range(len(self.encoder) - 2, -1, -1):
            self.upsamples.append(Upsample(channels[i + 1], channels[i]))
            patch = 2
            basic_block = BasicBlock(channels[i], nhead, embeding_dim)
            block = PatchEmbed(2 * channels[i], channels[i], patch, basic_block)

            self.decoder.append(block)

        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.downsample = nn.MaxPool2d(2, 2)
        self.out = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                 ConvBNAct(channels[0], channels[0], 1),
                                 nn.UpsamplingBilinear2d(scale_factor=2),
                                 ConvBNAct(channels[0], channels[0], 1),
                                 nn.UpsamplingBilinear2d(scale_factor=2),
                                 ConvBNAct(channels[0], out_channels, 1))

    def forward(self, q, k, v):
        pass_through = []
        # b,c,h,w
        for i, encoder in enumerate(self.encoder):
            q, k, v = encoder(q, k, v)
            if i != len(self.encoder) - 1:
                pass_through.append([q, k, v])
                # q = self.downsample(q)
                # k = self.downsample(k)
                # v = self.downsample(v)
        for i, decoder in enumerate(self.decoder):
            q = self.upsamples[i](q)
            k = self.upsamples[i](k)
            v = self.upsamples[i](v)
            passq, passk, passv = pass_through.pop()
            q = torch.cat([q, passq], dim=1)
            k = torch.cat([k, passk], dim=1)
            v = torch.cat([v, passv], dim=1)
            q, k, v = decoder(q, k, v)
            q = self.upscale(q)
            k = self.upscale(k)
            v = self.upscale(v)
        #
        q = self.out(q)
        return q
        pass


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, patch, block):
        super().__init__()
        self.block = block
        self.proj_q = ConvBNAct(in_channels, out_channels, patch, stride=patch)
        self.proj_k = ConvBNAct(in_channels, out_channels, patch, stride=patch)
        self.proj_v = ConvBNAct(in_channels, out_channels, patch, stride=patch)

    def forward(self, q, k, v):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        b, c, h, w = q.shape
        q = E.rearrange(q, "b c h w -> b (h w) c")
        k = E.rearrange(k, "b c h w -> b (h w) c")
        v = E.rearrange(v, "b c h w -> b (h w) c")
        q, k, v = self.block(q, k, v)
        q = E.rearrange(q, "b (h w) c -> b c h w", h=h, w=w)
        k = E.rearrange(k, "b (h w) c -> b c h w", h=h, w=w)
        v = E.rearrange(v, "b (h w) c -> b c h w", h=h, w=w)
        return q, k, v


class BasicBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.activation = nn.ReLU()
        # Implementation of Feedforward model
        self.ff_q = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ff_k = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ff_v = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1_q = nn.LayerNorm(d_model)
        self.norm2_q = nn.LayerNorm(d_model)

        self.norm1_k = nn.LayerNorm(d_model)
        self.norm2_k = nn.LayerNorm(d_model)

        self.norm1_v = nn.LayerNorm(d_model)
        self.norm2_v = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        q = q + self.attn(self.norm1_q(q), self.norm1_k(k), self.norm1_v(v))[0]

        q = q + self.ff_q(self.norm2_q(q))
        k = k + self.ff_k(self.norm2_k(k))
        v = v + self.ff_v(self.norm2_v(v))
        return q, k, v
        # if self.norm_first:
        #     x = x + self._sa_block(self.norm1(x))
        #     x = x + self._ff_block(self.norm2(x))
        # else:
        #     x = self.norm1(x + self._sa_block(x))
        #     x = self.norm2(x + self._ff_block(x))
