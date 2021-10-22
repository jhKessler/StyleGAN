import torch
from torch import tensor
import torch.nn as nn

class BaseConstant(nn.Module):
    """Learned Base Constant for Generator Network"""

    def __init__(self, n_channels: int, size: int = 4) -> None:
        super().__init__()
        # learned constant
        self.base = nn.Parameter(torch.randn(1, n_channels, size, size))

    def forward(self, inp: tensor) -> tensor:
        """Returns base constant for every noise batch element"""
        batch_size = inp.shape[0]
        return self.base.repeat(batch_size, 1, 1, 1)