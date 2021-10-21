import torch
from torch import tensor
import torch.nn as nn

class FromRGB(nn.Module):

    def __init__(self, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1, bias=False)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, inp: tensor) -> tensor:
        out = self.conv(inp)
        out = self.leaky(out)
        return out