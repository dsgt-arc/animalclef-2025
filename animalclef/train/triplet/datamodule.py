"""
Updated data handling module for the AnimalCLEF dataset with triplet support.

This module provides streamlined dataset classes and a PyTorch Lightning DataModule
for loading and preprocessing animal images for triplet learning.

Generated with Claude Sonnet 3.7
"""

import logging
import os
from typing import Any, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from animalclef.dataset import split_reid_data
from embed.transform import get_dino_processor_and_model

logger = logging.getLogger(__name__)


class AnimalEmbeddingDataset(Dataset):
    """
    Dataset for animal image embeddings with individual ID as target.
    
    Uses a DINO model to extract embeddings from images.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str,
        processor: Any,
        model: Any,
        embed_type: str = "cls",
        target_col: str = "individual_id",
        img_id_col: str = "image_id",
    ):
        """
        Initialize the dataset.
        
        Args:
            metadata: DataFrame containing image metadata
            img_dir: Directory containing images
            processor: DINO image processor
            model: DINO model for feature extraction
            embed_type: Type of embedding to use ('cls' or 'avg_patch')
            target_col: Column name for the target/label
            img_id_col: Column name for the image ID
        """
        self.metadata = metadata
        self.img_dir = img_dir
        self.processor = processor
        self.model = model
        self.embed_type = embed_type
        self.target_col = target_col
        self.img_id_col = img_id_col
        self.device = next(model.parameters()).device

        # Create mapping from individual ID to integer class
        self.classes = sorted(metadata[target_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (embedding, label_index, original_label)
        """
        row = self.metadata.iloc[idx]
        
        img_id = row[self.img_id_col]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        # Load and process the image
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Extract embedding using DINO
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state
                
                if self.embed_type == "cls":
                    embedding = features[:, 0, :].squeeze(0)
                else:  # avg_patch
                    embedding = features[:, 1:, :].mean(dim=1).squeeze(0)
                
                # Move embedding to CPU
                embedding = embedding.cpu()
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            # Return a zero tensor as fallback
            embedding = torch.zeros(self.model.config.hidden_size)
        
        target = row[self.target_col]
        target_int = self.class_to_idx[target]
        label = torch.tensor(target_int, dtype=torch.long)
        
        return embedding, label, target


class TripletBatchSampler:
    """
    Batch sampler that creates batches for triplet learning.
    
    Ensures each batch contains multiple instances of the same class
    to allow for triplet mining.
    """
    
    def __init__(
        self,
        dataset: AnimalEmbeddingDataset,
        batch_size: int = 32,
        drop_last: bool = False,
    ):
        """
        Initialize the triplet batch sampler.
        
        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.labels = []
        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            self.labels.append(label.item())
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create label to indices mapping
        self.label_to_indices = {}
        for i, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)
        
        # Filter classes with only one instance
        self.labels_with_multiple_instances = [
            label for label in self.label_to_indices
            if len(self.label_to_indices[label]) > 1
        ]
    
    def __iter__(self):
        """
        Generate batches for triplet learning.
        
        Yields:
            List of indices for each batch
        """
        import random
        
        # Shuffle the data
        indices = list(range(len(self.labels)))
        random.shuffle(indices)
        
        # Create batches
        for start_idx in range(0, len(indices), self.batch_size):
            if self.drop_last and start_idx + self.batch_size > len(indices):
                break
                
            batch = indices[start_idx:start_idx + self.batch_size]
            
            # Ensure batch has at least some classes with multiple instances
            classes_in_batch = set([self.labels[i] for i in batch])
            classes_with_multiple = [
                label for label in classes_in_batch
                if len(self.label_to_indices[label]) > 1
            ]
            
            if len(classes_with_multiple) < 1 and len(self.labels_with_multiple_instances) > 0:
                # Add at least one class with multiple instances
                additional_class = random.choice(self.labels_with_multiple_instances)
                instances = random.sample(self.label_to_indices[additional_class], 2)
                
                # Replace two random elements in batch
                if len(batch) >= 2:
                    replace_indices = random.sample(range(len(batch)), 2)
                    for i, instance in zip(replace_indices, instances):
                        batch[i] = instance
            
            yield batch
    
    def __len__(self):
        """
        Get the number of batches.
        
        Returns:
            Number of batches
        """
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


class AnimalTripletDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AnimalCLEF triplet learning.
    
    Handles data loading, preprocessing, and splitting into train/val/test sets.
    Integrates with DINO models for feature extraction.
    """

    def __init__(
        self,
        metadata_path: str,
        img_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_name: str = "facebook/dinov2-base",
        embed_type: str = "cls",
    ):
        """
        Initialize the data module.
        
        Args:
            metadata_path: Path to the metadata CSV file
            img_dir: Directory containing images
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            model_name: Name of the DINO model to use
            embed_type: Type of embedding to use ('cls' or 'avg_patch')
        """
        super().__init__()
        
        assert embed_type in ["cls", "avg_patch"], f'Error: embed_type = "{embed_type}" is invalid!'
        
        self.metadata_path = metadata_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.embed_type = embed_type
        
        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0
        self.class_names = []
        self.processor = None
        self.model = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        # Load the DINO processor and model to ensure they're downloaded
        get_dino_processor_and_model(self.model_name)

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for training, validation and test.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        
        # Split dataset
        train_meta, val_meta, test_meta = split_reid_data(df=metadata)
        
        # Log split information
        logger.info(f"Train samples: {len(train_meta)}")
        logger.info(f"Val samples: {len(val_meta)}")
        logger.info(f"Test samples: {len(test_meta)}")
        
        # Get class information
        all_classes = sorted(metadata["individual_id"].unique())
        self.num_classes = len(all_classes)
        self.class_names = all_classes
        logger.info(f"Number of individual classes: {self.num_classes}")
        
        # Set up DINO model and processor
        self.processor, self.model = get_dino_processor_and_model(self.model_name)
        
        # Create datasets
        kwargs = dict(
            img_dir=self.img_dir,
            processor=self.processor,
            model=self.model,
            embed_type=self.embed_type,
        )
        
        self.train_dataset = AnimalEmbeddingDataset(metadata=train_meta, **kwargs)
        self.val_dataset = AnimalEmbeddingDataset(metadata=val_meta, **kwargs)
        self.test_dataset = AnimalEmbeddingDataset(metadata=test_meta, **kwargs)

    def train_dataloader(self):
        """
        Create the training dataloader with triplet batch sampling.
        
        Returns:
            DataLoader instance for training
        """
        sampler = TripletBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Create the validation dataloader.
        
        Returns:
            DataLoader instance for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Create the test dataloader.
        
        Returns:
            DataLoader instance for testing
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )