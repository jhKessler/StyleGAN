import torch
from torch import tensor
import torch.nn as nn


class DiscBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        
        self.normalization = nn.InstanceNorm2d

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            self.normalization(in_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            self.normalization(out_channels),
            nn.LeakyReLU(0.1),
            # downsample
            nn.AvgPool2d(kernel_size=2)
        ])

    def forward(self, inp: tensor) -> tensor:
        out = inp
        for layer in self.layers:
            out = layer(out)
        return out
