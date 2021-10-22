import torch
from torch import tensor
import torch.nn as nn

class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, in_channels: int, style_dim: int) -> None:
        super().__init__()

        self.in_channels = in_channels

        self.norm = nn.InstanceNorm2d(in_channels)
        self.style = nn.Linear(style_dim, in_channels*2)

        self.style.bias.data[:in_channels] = 1
        self.style.bias.data[in_channels:] = 0


    def forward(self, inp: tensor, style: tensor) -> tensor:
        style = style.squeeze()
        batch_size, style_dim = style.shape
        
        style = self.style(style).reshape(batch_size, self.in_channels*2, 1, 1)
        gamma, beta = torch.split(style, self.in_channels, dim=1)
        out = self.norm(inp)
        out = gamma * out + beta
        return out