from typing import List

import torch
import torch.nn as nn


class TripletHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int] = [2048]):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dims[0]))

        for idx in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[idx - 1], hidden_dims[idx]))

        layers.append(nn.Linear(hidden_dims[-1], out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.layers(X)

        return out
