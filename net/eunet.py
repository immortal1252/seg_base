import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def init_params(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class Upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv3x3(ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class convblock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(convblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, ch_in, r=8):
        super(SEBlock, self).__init__()
        self.averagePool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_in // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // r, ch_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.averagePool(x).view(b, c)
        y = self.linear(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def resnet18():
    net = models.resnet18()
    return net


class UNetComposedLossSupervised(nn.Module):
    def __init__(self, ch_out,sv=False):
        super(UNetComposedLossSupervised, self).__init__()
        self.sv = sv
        backbone = resnet18()
        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.down_layer1 = backbone.layer1
        self.down_layer2 = backbone.layer2
        self.down_layer3 = backbone.layer3
        self.down_layer4 = backbone.layer4

        if sv:
            self.logix_pixel5 = nn.Sequential(
                conv3x3(512, 128),
                conv3x3(128, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                conv3x3(32, 1),
            )
            self.logix_pixel4 = nn.Sequential(
                conv3x3(256, 64),
                conv3x3(64, 16),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                conv3x3(16, 1)
            )
            self.logix_pixel3 = nn.Sequential(
                conv3x3(128, 32),
                conv3x3(32, 8),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                conv3x3(8, 1)
            )
            self.logix_pixel2 = nn.Sequential(
                conv3x3(64, 16),
                conv3x3(16, 4),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                conv3x3(4, 1)
            )
        self.fuse = conv3x3(4, 1)

        self.se_block = SEBlock(ch_in=512, r=32)

        self.up4 = up_conv(512, 256)
        self.up4_block = convblock(512, 256)

        self.up3 = up_conv(256, 128)
        self.up3_block = convblock(256, 128)

        self.up2 = up_conv(128, 64)
        self.up2_block = convblock(128, 64)

        self.up1 = conv1x1(64, ch_out)
        init_params(self)
        backbone.load_state_dict(torch.load(
            'pretrained/resnet18-f37072fd.pth'))

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.down_layer1(x)
        x2 = self.down_layer2(x1)
        x3 = self.down_layer3(x2)
        x4 = self.down_layer4(x3)

        # 通道 attention
        x4 = self.se_block(x4)
        if self.training and self.sv:
            o4 = self.logix_pixel5(x4)
            o4 = F.interpolate(o4, scale_factor=32, mode='bilinear', align_corners=True)
        d4 = self.up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up4_block(d4)

        if self.training and self.sv:
            o3 = self.logix_pixel4(d4)
            o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=True)
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up3_block(d3)

        if self.training and self.sv:
            o2 = self.logix_pixel3(d3)
            o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=True)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up2_block(d2)

        if self.training and self.sv:
            o1 = self.logix_pixel2(d2)
            o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=True)

        d1 = self.up1(d2)
        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=True)

        if self.training and self.sv:
            return out, o1, o2, o3, o4
        else:
            return out
