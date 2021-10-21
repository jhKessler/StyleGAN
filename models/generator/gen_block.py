import torch
from torch import tensor
import torch.nn as nn

from .modules import BaseConstant
from .modules import NoiseInjection
from .modules import AdaptiveInstanceNormalization

class GenBlock(nn.Module):
    """Generator Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool, kernel_size: int = 3, padding: int = 1, initial: bool = False, style_dim: int = 256) -> None:
        super().__init__()

        self.style_dim = style_dim
        self.initial = initial

        # add learned constant if first block
        if initial:
            self.conv1 = BaseConstant(in_channels)
        # add upsample
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
                )

        # noise injection
        self.noise1 = NoiseInjection(out_channels)
        # AdaIN
        self.ada1 = AdaptiveInstanceNormalization(out_channels, self.style_dim)
        # relu
        self.leaky1 = nn.LeakyReLU(0.1)
        # conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)
        # noise injection
        self.noise2 = NoiseInjection(out_channels)
        # AdaIN
        self.ada2 = AdaptiveInstanceNormalization(out_channels, self.style_dim)
        # relu
        self.leaky2 = nn.LeakyReLU(0.1)

    def forward(self, inp: tensor, style: tensor) -> tensor:
        if not self.initial:
            batch_size, _, x, y = inp.shape
        else:
            batch_size, x, y = inp, 4, 4

        style1, style2 = torch.split(style, 1, dim=1)

        out = self.conv1(batch_size if self.initial else inp)
        out = self.noise1(out, torch.randn(batch_size, 1, out.shape[2], out.shape[2]))
        out = self.ada1(out, style1)
        out = self.leaky1(out)
        out = self.conv2(out)
        out = self.noise2(out, torch.randn(batch_size, 1, out.shape[2], out.shape[2]))
        out = self.ada2(out, style2)
        out = self.leaky2(out)

        return out