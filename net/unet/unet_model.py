""" Full assembly of the parts to form the complete network """

from .unet_parts import *


def init_params(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, out_channels=None, act=nn.ReLU):
        super(UNet, self).__init__()

        if out_channels is None:
            out_channels = [32, 64, 128, 256, 512]
        assert len(out_channels) == 5

        self.inc = DoubleConv(in_channels, out_channels[0], act=act)
        self.down1 = MaxpoolConv(out_channels[0], out_channels[1], act=act)
        self.down2 = MaxpoolConv(out_channels[1], out_channels[2], act=act)
        self.down3 = MaxpoolConv(out_channels[2], out_channels[3], act=act)

        self.down4 = MaxpoolConv(out_channels[3], out_channels[4], act=act)
        self.up1 = Up(out_channels[4], out_channels[3], act=act)
        self.up2 = Up(out_channels[3], out_channels[2], act=act)
        self.up3 = Up(out_channels[2], out_channels[1], act=act)
        self.up4 = Up(out_channels[1], out_channels[0], act=act)
        self.outc = nn.Conv2d(out_channels[0], n_classes, 1, 1)
        init_params(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
