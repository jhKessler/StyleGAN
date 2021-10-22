import torch
from torch import tensor
import torch.nn as nn

class DecisionBlock(nn.Module):

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, bias=False),
            nn.Flatten(),
            nn.Linear(in_features=in_channels, out_features=1, bias=False)
        ])

    def forward(self, inp: tensor) -> tensor:
        # minibatch stdev
        out_std = torch.sqrt(inp.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(inp.size(0), 1, 4, 4)
        print(inp.shape, mean_std.shape)
        inp = torch.cat([inp, mean_std], 1)

        out = inp

        for layer in self.layers:
            out = layer(out)
        return out.view(-1)


