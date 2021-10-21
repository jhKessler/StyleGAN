from torch import tensor
import torch.nn as nn

class MappingNetwork(nn.Module):
    """Network from Mapping Latent Space Z -> W so that the Network is not bound by the Training Data Distribution"""

    def __init__(self, n_layer: int, layer_perceptrons: int, bias: bool = False) -> None:
        """Constructor for Mapping network Class""" 
        super().__init__()

        self.layers = nn.ModuleList() # modulelist for storing layers
        
        # fill module list
        for i in range(n_layer):
            # add linear layer
            self.layers.append(nn.Linear(in_features=layer_perceptrons, out_features=layer_perceptrons, bias=bias))

            # add relu activation
            if i < (n_layer-1):
                self.layers.append(nn.LeakyReLU(0.1))

    def forward(self, noise: tensor) -> tensor:
        mapping = noise
        for layer in self.layers:
            mapping = layer(mapping)
        return mapping