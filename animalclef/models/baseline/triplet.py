from typing import List

import torch
import torch.nn as nn


class TripletHead(nn.Module):
    """
    A simple MLP head for use in triplet learning.

    Takes as input...
    - in_dim: input dimension
    - out_dim: output dimension (number of classes/identities)
    - hidden_dims: list of hidden dimensions to use between input and output layers. Defaults to bottleneck shrinking strategy.
    - dropout: the dropout rate to use before the output dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int] = [2048, 1024, 256],  # final hidden is a bottleneck
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        layers = []

        prev_dim = in_dim
        for idx, curr_dim in enumerate(hidden_dims):
            layers += [
                nn.Linear(prev_dim, curr_dim),
                nn.BatchNorm1d(curr_dim),
                nn.ReLU(),
            ]

            if idx == len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))

            prev_dim = curr_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.layers(X)

        return out
