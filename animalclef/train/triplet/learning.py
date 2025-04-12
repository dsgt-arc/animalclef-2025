"""
Triplet learning module for animal re-identification.

This module provides a PyTorch Lightning implementation of triplet learning
using a frozen DINO backbone and a projector (SimpleEmbedder).

Generated with Claude Sonnet 3.7
"""

import logging, sys
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from animalclef.dataset import split_reid_data
from .embedder import SimpleEmbedder
from animalclef.embed.transform import get_dino_processor_and_model

logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    """
    Triplet loss with random triplet selection.
    
    Computes loss based on the distances between anchor-positive and 
    anchor-negative pairs, applying a margin to ensure separation.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize the triplet loss with a margin.
        
        Args:
            margin: Minimum desired distance between anchor-negative and anchor-positive pairs
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss using random triplet selection.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size) containing label indices
            
        Returns:
            Average triplet loss for valid triplets
        """
        # Compute pairwise distances between all embeddings
        dist_matrix = self._pairwise_distances(embeddings)
        
        # Get random positive and negative pairs for each anchor
        positive_dist = self._get_random_positive_dist(dist_matrix, labels)
        negative_dist = self._get_random_negative_dist(dist_matrix, labels)
        
        # Calculate triplet loss with margin
        triplet_loss = F.relu(positive_dist - negative_dist + self.margin)
        
        # Count number of valid triplets (those with positive loss)
        valid_triplets = torch.sum(triplet_loss > 1e-16).item()
        
        if valid_triplets == 0:
            # Return 0 loss if no valid triplets found
            return torch.tensor(0.0, device=embeddings.device)
        
        # Return mean of positive losses
        return torch.mean(triplet_loss)

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix of pairwise distances between embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Matrix of pairwise squared Euclidean distances
        """
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        distances = F.relu(distances)
        return distances
    
    def _get_random_positive_dist(self, dist_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        For each anchor, randomly select a positive example.
        
        Args:
            dist_matrix: Pairwise distance matrix
            labels: Tensor of shape (batch_size) containing label indices
            
        Returns:
            Tensor of random positive distances for each anchor
        """
        batch_size = dist_matrix.size(0)
        
        # Create a 2D mask for positive pairs (same label)
        mask_positives = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
        mask_positives.fill_diagonal_(False)
        
        # For each anchor, get indices of all valid positives
        positive_indices = [torch.where(mask_positives[i])[0] for i in range(batch_size)]
        
        # Randomly select one positive for each anchor that has positives
        random_positive_dist = torch.zeros(batch_size, device=dist_matrix.device)
        for i, pos_indices in enumerate(positive_indices):
            if len(pos_indices) > 0:
                random_idx = torch.randint(0, len(pos_indices), (1,))
                random_positive_dist[i] = dist_matrix[i, pos_indices[random_idx]]
                
        return random_positive_dist
    
    def _get_random_negative_dist(self, dist_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        For each anchor, randomly select a negative example.
        
        Args:
            dist_matrix: Pairwise distance matrix
            labels: Tensor of shape (batch_size) containing label indices
            
        Returns:
            Tensor of random negative distances for each anchor
        """
        batch_size = dist_matrix.size(0)
        
        # Create a 2D mask for negative pairs (different label)
        mask_negatives = ~labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
        
        # For each anchor, get indices of all valid negatives
        negative_indices = [torch.where(mask_negatives[i])[0] for i in range(batch_size)]
        
        # Randomly select one negative for each anchor that has negatives
        random_negative_dist = torch.zeros(batch_size, device=dist_matrix.device)
        for i, neg_indices in enumerate(negative_indices):
            if len(neg_indices) > 0:
                random_idx = torch.randint(0, len(neg_indices), (1,))
                random_negative_dist[i] = dist_matrix[i, neg_indices[random_idx]]
                
        return random_negative_dist


class TripletLearningModule(pl.LightningModule):
    """
    PyTorch Lightning module for triplet learning with a frozen DINO backbone.
    
    Uses a SimpleEmbedder to project DINO embeddings and trains using triplet loss
    with online hard triplet mining.
    """
    
    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-base",
        embedding_dim: int = 768,
        projection_dim: int = 128,
        learning_rate: float = 1e-4,
        margin: float = 1.0,
        embed_type: str = "cls",
    ):
        """
        Initialize the triplet learning module.
        
        Args:
            dino_model_name: Name of the DINO model to use as backbone
            embedding_dim: Input dimension from DINO model
            projection_dim: Output dimension after projection
            learning_rate: Learning rate for optimizer
            margin: Margin for triplet loss
            embed_type: Type of embedding to use from DINO ('cls' or 'avg_patch')
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the embedder
        self.projector = SimpleEmbedder(in_dim=embedding_dim, out_dim=projection_dim)
        
        # Initialize triplet loss
        self.triplet_loss = TripletLoss(margin=margin)
        
        # We don't need to initialize DINO model here as it will be used through the dataloader
        self.learning_rate = learning_rate
        self.embed_type = embed_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projector.
        
        Args:
            x: Tensor of DINO embeddings
            
        Returns:
            Projected embeddings
        """
        return self.projector(x)

    def _step(self, batch: List[Tuple], batch_idx: int, prefix: str) -> Dict[str, torch.Tensor]:
        # Extract embeddings and labels from batch
        embeddings = []
        labels = []
        
        # Process each sample in the batch
        for sample in batch:
            # Each sample is a triplet of (anchor, positive, negative)
            # But we'll just collect all anchors, positives, and negatives together
            # and let the loss function handle the triplet mining
            anchor_embedding, anchor_label, _ = sample[0]
            positive_embedding, positive_label, _ = sample[1]
            negative_embedding, negative_label, _ = sample[2]
            
            embeddings.extend([anchor_embedding, positive_embedding, negative_embedding])
            labels.extend([anchor_label, positive_label, negative_label])
        
        # Stack embeddings and labels
        embeddings = torch.stack(embeddings)
        labels = torch.stack(labels).squeeze()
        
        # Apply the projector to get final embeddings
        projected_embeddings = self(embeddings)
        
        # Calculate triplet loss
        loss = self.triplet_loss(projected_embeddings, labels)
        
        # Log training loss
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {f'{prefix}_loss': loss}
    
    def training_step(self, batch: List[Tuple], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Execute a training step on a batch of data.
        
        Args:
            batch: Batch of data containing embeddings and labels
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with the training loss
        """
        
        return self._step(batch, batch_idx, prefix='train')
    
    def validation_step(self, batch: List[Tuple], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Execute a validation step on a batch of data.
        
        Args:
            batch: Batch of data containing embeddings and labels
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with the validation loss
        """
        
        return self._step(batch, batch_idx, prefix='val')
    
    def test_step(self, batch: List[Tuple], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Execute a test step on a batch of data.
        
        Args:
            batch: Batch of data containing embeddings and labels
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with the test loss
        """
        # Reuse validation step logic
        return self._step(batch, batch_idx, prefix='test')
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.
        
        Returns:
            Optimizer instance
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)