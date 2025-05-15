import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjectionHead(nn.Module):
    """
    Simple projection head to transform embeddings.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.fc(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class NonlinearProjectionHead(nn.Module):
    """
    Simple projection head to transform embeddings.

    Tried out nonlinearity (GELU) with more capacity (hidden_dim).
    Normalization is useful because it links the cosine distance to
    euclidean distance via inner product.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.projection(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class TripletLoss(nn.Module):
    """
    Triplet loss with a margin.
    Takes embeddings of an anchor, a positive and a negative sample.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.pairwise_distance(anchor, positive)
        neg_dist = torch.pairwise_distance(anchor, negative)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class MRLTripletLoss(nn.Module):
    """
    Matryoshka Representation Learning with Triplet Loss.
    Applies triplet loss at multiple embedding dimensionalities (prefixes).
    """

    def __init__(
        self,
        margin: float = 1.0,
        nested_dims: list = [],
        loss_weights: list = None,
    ):
        super().__init__()
        self.margin = margin
        self.base_triplet_loss = TripletLoss(margin=margin)
        self.nested_dims = nested_dims

        # Weights for each loss component (full + nested).
        # If not provided, defaults to equal weighting (1.0 for each).
        num_losses = 1 + len(self.nested_dims)
        self.loss_weights = loss_weights
        if self.loss_weights is None:
            self.loss_weights = [1.0] * num_losses
        assert len(self.loss_weights) == num_losses
        self.loss_weight_sum = sum(self.loss_weights)

    def forward(self, anchor_full, positive_full, negative_full):
        total_loss = 0.0

        loss_full = self.base_triplet_loss(anchor_full, positive_full, negative_full)
        total_loss += self.loss_weights[0] * loss_full

        for i, d_prefix in enumerate(self.nested_dims):
            if d_prefix >= anchor_full.shape[-1]:
                continue

            anchor_prefix = anchor_full[:, :d_prefix]
            positive_prefix = positive_full[:, :d_prefix]
            negative_prefix = negative_full[:, :d_prefix]

            loss_prefix = self.base_triplet_loss(
                anchor_prefix, positive_prefix, negative_prefix
            )
            total_loss += self.loss_weights[i + 1] * loss_prefix

        return total_loss / self.loss_weight_sum
