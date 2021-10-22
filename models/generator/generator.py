import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from . import GenBlock
from . import MappingNetwork

class Generator(nn.Module):
    """Generator class for GAN"""

    def __init__(self, style_dim: int, bias: bool = False) -> None:
        """Constructor for Generator Class"""
        super().__init__()
        
        self.style_dim = style_dim
        self.mapping_network = MappingNetwork(n_layer=8, layer_perceptrons=style_dim)

        self.base = GenBlock(in_channels=256, out_channels=256, bias=bias, initial=True, style_dim=style_dim) # 4x4

        self.layers = nn.ModuleList([
            GenBlock(in_channels=256, out_channels=256, bias=bias, style_dim=style_dim), # 8x8
            GenBlock(in_channels=256, out_channels=256, bias=bias, style_dim=style_dim), # 16x16
            GenBlock(in_channels=256, out_channels=128, bias=bias, style_dim=style_dim), # 32x32
            GenBlock(in_channels=128, out_channels=64, bias=bias, style_dim=style_dim), # 64x64
        ])

        self.to_rgb = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, bias=bias)


    def forward(self, noise: tensor, device: torch.device) -> tensor:
        gen_block_count, batch_size, conv_block_count, style_dim = noise.shape
        
        out = self.base(noise[0], self.mapping_network(noise[0]), device)
        noise = noise[1:]

        for i in range(gen_block_count-1):
            style = self.mapping_network(noise[i])
            out = self.layers[i](out, style, device)

        out = self.to_rgb(out)   
        return out
