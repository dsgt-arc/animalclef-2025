"""
Data handling module for the AnimalCLEF dataset.

This module provides dataset classes and a PyTorch Lightning DataModule for loading
and preprocessing animal images with individual ID labels. It supports working with
raw images or pre-extracted features.

Classes:
    AnimalImageDataset: Dataset for animal images with individual ID as target
    AnimalFeatureDataset: Dataset for pre-extracted animal image features
    AnimalDataModule: PyTorch Lightning DataModule for the AnimalCLEF dataset
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.io import read_image

from dataset import split_reid_data
from embed.transform import get_dino_processor_and_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)


class AnimalImageDataset(Dataset):
    """
    Dataset for animal images with individual ID as target.

    Loads images from disk and applies transformations as needed.
    Provides error handling for missing or corrupted images.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        img_dir: str,
        transform=None,
        target_col: str = "individual_id",
        img_id_col: str = "image_id",
    ):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform
        self.target_col = target_col
        self.img_id_col = img_id_col

        # Create mapping from individual ID to integer class
        self.classes = sorted(metadata[target_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        img = row[self.img_id_col]
        img_path = os.path.join(self.img_dir, f"{img}.jpg")

        target = row[self.target_col]
        target_int = self.class_to_idx[target]

        X = read_image(img_path).float()
        y = torch.tensor(target_int).float()

        if self.transform:
            X = self.transform(X)

        return X, y, target


class AnimalTripletImageDataset(AnimalImageDataset):
    """
    Dataset for animal image triplets with individual ID as target.

    Loads images from disk and applies transformations as needed.
    Provides error handling for missing or corrupted images.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize using inheritance.
        """

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        """
        Returns a triplet with anchor at passed index.

        Inputs:
            idx: anchor index in the metadata.

        Returns:
            X_anchor, y_anchor, target_anchor: anchor image returns.
            X_positive, y_positive, target_positive: positive image returns.
            X_negative, y_negative, target_negative: negative image returns.
        """

        X_anchor, y_anchor, target_anchor = super().__getitem__(idx)

        positive_tmp = self.metadata.loc[
            (self.metadata[self.target_col] == target_anchor)
            & (
                self.metadata[self.img_id_col]
                != self.metadata.iloc[idx][self.img_id_col]
            )
        ]
        negative_tmp = self.metadata.loc[
            (self.metadata[self.target_col] != target_anchor)
            & (
                self.metadata[self.img_id_col]
                != self.metadata.iloc[idx][self.img_id_col]
            )
        ]

        positive_idx = positive_tmp.sample(n=1, random_state=42).index.iloc[0]
        negative_idx = negative_tmp.sample(n=1, random_state=42).index.iloc[0]

        return (
            (X_anchor, y_anchor, target_anchor),
            super().__getitem__(positive_idx),
            super().__getitem__(negative_idx),
        )


class AnimalFeatureTransform(nn.Module):
    """
    Simple transformation that turns image Tensors into embeddign Tensors.
    """

    def __init__(self, processor: Any, model: Any, embed_type: str = "cls"):
        """
        Inputs:
            processor: output processor from embed.transform.get_dino_processor_and_model function.
            model: output model from embed.transform.get_dino_processor_and_model function.
            embed_type: string indicating the type of embedding to return (must be either 'cls' or 'avg_patch'; default = 'cls').
        """

        super().__init__()

        assert embed_type in ["cls", "avg_patch"], (
            f'Error: embed_type = "{embed_type}" is invalid!'
        )

        self.processor = processor
        self.model = model
        self.embed_type = embed_type

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Mostly taken from embed/transform/WrappedDino._make_predict_fn.predict function implementation.

        Inputs:
            X: batch of images.

        Returns:
            features: batch of image embeddings.
        """

        with torch.no_grad():
            processed_inputs = self.processor(
                images=X,  # write a test for this
                return_tensors="pt",
            )
            outputs = self.model(**processed_inputs)
            features = outputs.last_hidden_state

            if self.embed_type == "cls":
                features = features[:, 0, :]
            else:
                features = features[:, 1:, :].mean(dim=1)

        return features


class AnimalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AnimalCLEF dataset.

    Handles data loading, preprocessing, and splitting into train/val/test sets.
    Integrates with vision transformer models by providing appropriate image transforms.
    """

    def __init__(
        self,
        metadata_path: str,
        img_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_name: str = "facebook/dinov2-base",
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        extract_features: bool = True,
        embed_type: str = "cls",
    ):
        super().__init__()

        assert embed_type in ["cls", "avg_patch"], (
            f'Error: embed_type = "{embed_type}" is invalid!'
        )

        self.metadata_path = metadata_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.extract_features = extract_features
        self.embed_type = embed_type

        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0
        self.class_names = []
        self.feature_extractor = None

    def _create_dataloader(self, dataset, shuffle=False):
        """Helper method to create a dataloader with consistent parameters."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        # Load the image processor
        AutoImageProcessor.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for training, validation and test."""
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)

        # Split dataset
        # train_val_meta, test_meta = train_test_split(
        #     metadata,
        #     test_size=self.test_size,
        #     stratify=metadata["individual_id"],
        #     random_state=self.seed,
        # )

        # train_meta, val_meta = train_test_split(
        #     train_val_meta,
        #     test_size=self.val_size / (1 - self.test_size),
        #     stratify=train_val_meta["individual_id"],
        #     random_state=self.seed,
        # )

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

        # Set up processor for transforming images
        # self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)

        self.processor, self.embedder = get_dino_processor_and_model(self.model_name)
        self.transform = AnimalFeatureTransform(
            self.processor, self.embedder, embed_type=self.embed_type
        )

        kwargs = dict(
            img_dir=self.img_dir,
            transform=self.transform,
        )

        self.train_dataset = AnimalTripletImageDataset(metadata=train_meta, **kwargs)
        self.val_dataset = AnimalTripletImageDataset(metadata=val_meta, **kwargs)
        self.test_dataset = AnimalTripletImageDataset(metadata=test_meta, **kwargs)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)
