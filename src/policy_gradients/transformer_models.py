import torch
import torch.nn as nn

"""
This module defines the transformer models.
"""

# utility models

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(nn.Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            lin = ls

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP_TwoLayers_Norm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# main model

class SceneTransformer(nn.Module):
    """
    Contains an encoding linear layer for scenes, a transformer layer
    for relating the different objects, and policy and value
    linear heads.
    """
    def __init__(self, ):

