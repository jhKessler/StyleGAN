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
            DiscBlock(in_channels=128, out_channels=128, bias=bias),
            DiscBlock(in_channels=128, out_channels=128, bias=bias),
            DiscBlock(in_channels=128, out_channels=128, bias=bias),
            DiscBlock(in_channels=128, out_channels=256, bias=bias),
        ])

        self.from_rgb = FromRGB(out_channels=128)

        self.outp = DecisionBlock(in_channels=256)

    def forward(self, inp: tensor) -> tensor:
        out = self.from_rgb(inp)

        for lyr in self.layers:
            out = lyr(out)

        out = self.outp(out)
        return out