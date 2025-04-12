from typing import List

import torch
import torch.nn as nn


class SimpleEmbedder(nn.Module):
    '''
    Simple embedding projection for our minimal viable solution.

    Attributes:
        __version__: version of the embedding projection.
        in_dim: input embedding dimension.
        out_dim: output embedding dimension.
        proj: a simple linear layer for projecting the input embedding to a different size.
    '''

    def __init__(self, in_dim: int, out_dim: int):
        '''
        Initializes an instance of the SimpleEmbedder class.

        Inputs:
            in_dim: input embedding dimension.
            out_dim: output embedding dimension.
        '''

        self.__version__ = '0.1.0'

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Projects the input embedding.

        Inputs:
            x: the input embedding (likely from the DINO backbone).
        
        Returns:
            out: the projected embedding with new length (as specified by out_dim).
        '''

        out = self.proj(x)

        return out
