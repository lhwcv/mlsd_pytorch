import torch
import torch.nn as nn
from torch.nn import functional as F


class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x