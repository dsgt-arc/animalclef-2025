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

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, Optional
from PIL import Image
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split
import logging

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

        # Create list of (image_id, class_idx) pairs
        self.samples = [
            (row[self.img_id_col], self.class_to_idx[row[self.target_col]])
            for _, row in metadata.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, target = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        try:
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                # Handle both callable transforms and processor objects with __call__ method
                if hasattr(self.transform, "preprocess"):
                    # For HF processors that use preprocess
                    processed = self.transform.preprocess(img, return_tensors="pt")
                    img = processed.pixel_values.squeeze(0)
                else:
                    # For standard transforms
                    img = self.transform(img)

            return img, target

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder black image
            if self.transform:
                placeholder = torch.zeros(3, 224, 224)
            else:
                placeholder = Image.new("RGB", (224, 224), (0, 0, 0))
            return placeholder, target


# TODO: this should be a transform that takes a read image and converts it
# to dino embeddings directly. take advantage of get_dino_processor_and_model
class AnimalFeatureDataset(Dataset):
    """
    Dataset for pre-extracted animal image features.

    Used when working with pre-computed features instead of raw images.
    """

    def __init__(self, features: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # TODO: use the animalclef.embed.transform.WrappedDino._make_predict_fn
        # this should return a function that converts an image into a dictionary of results
        # it might be useful to extract this functionality a bit so we don't have to
        # instantiate the model
        return self.features[idx], self.labels[idx]


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
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.extract_features = extract_features

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
        train_val_meta, test_meta = train_test_split(
            metadata,
            test_size=self.test_size,
            stratify=metadata["individual_id"],
            random_state=self.seed,
        )

        train_meta, val_meta = train_test_split(
            train_val_meta,
            test_size=self.val_size / (1 - self.test_size),
            stratify=train_val_meta["individual_id"],
            random_state=self.seed,
        )

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
        self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)
        kwargs = dict(
            img_dir=self.img_dir,
            transform=self.feature_extractor,
        )
        self.train_dataset = AnimalImageDataset(metadata=train_meta, **kwargs)
        self.val_dataset = AnimalImageDataset(metadata=val_meta, **kwargs)
        self.test_dataset = AnimalImageDataset(metadata=test_meta, **kwargs)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)
