import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from . import FromRGB
from . import DiscBlock
from . import DecisionBlock

class Discriminator(nn.Module):

    def __init__(self, bias: bool = False):
        super().__init__()

        self.layers = nn.ModuleList([
            DiscBlock(in_channels=256, out_channels=256, bias=bias),
            DiscBlock(in_channels=256, out_channels=256, bias=bias),
            DiscBlock(in_channels=256, out_channels=256, bias=bias),
            DiscBlock(in_channels=128, out_channels=256, bias=bias),
            DiscBlock(in_channels=64, out_channels=128, bias=bias),
        ])

        self.from_rgb = nn.ModuleList([
            FromRGB(out_channels=256),
            FromRGB(out_channels=256),
            FromRGB(out_channels=256),
            FromRGB(out_channels=128),
            FromRGB(out_channels=64)
        ])

        self.outp = DecisionBlock(in_channels=256)

    def forward(self, inp: tensor, step: int, alpha: float) -> tensor:
        out = self.from_rgb[step-1](inp)

        for i in range(step-1, 0, -1):
            out = self.layers[i](out)
            if i == step-1 and alpha < 1:
                downscaled = F.avg_pool2d(inp, 2)
                downscaled = self.from_rgb[i-1](downscaled)
                out = (out * alpha) + ((1 - alpha) * downscaled)

        out = self.outp(out)
        print(out.shape)
        return out