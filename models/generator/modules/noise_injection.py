import torch
from torch import tensor
import torch.nn as nn

class NoiseInjection(nn.Module):

    def __init__(self, n_channels: int) -> None: 
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n_channels, 1, 1))

    def forward(self, image: tensor, noise: tensor) -> tensor:
        return image + (self.weights * noise)