import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import logging
from typing import Dict, Any

from animalclef.data import AnimalDataModule
from animalclef.models.baseline import DinoClassifierWithExtractor

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_baseline_model(config_path: str):
    """
    Train baseline model using PyTorch Lightning.

    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize data module
    data_module = AnimalDataModule(
        metadata_path=config["data"]["metadata_path"],
        img_dir=config["data"]["img_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        model_name=config["model"]["dino_model_name"],
        val_size=config["data"]["val_split"],
        test_size=config["data"]["test_split"],
        seed=config["data"]["random_seed"],
        extract_features=False,  # Extract features on the fly
    )

    # Initialize model
    model = DinoClassifierWithExtractor(
        num_classes=config["model"]["num_classes"],
        dino_model_name=config["model"]["dino_model_name"],
        hidden_dim=config["model"]["params"]["hidden_dim"],
        dropout_rate=config["model"]["params"]["dropout_rate"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        freeze_backbone=config["model"]["freeze_backbone"],
        use_cls_token=config["model"]["params"]["use_cls_token"],
    )

    # Set up logging
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=save_dir, name="lightning_logs", default_hp_metric=False
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="dino-classifier-{epoch:02d}-{val_acc:.4f}",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=config["training"].get("early_stopping_patience", 10),
        min_delta=config["training"].get("early_stopping_min_delta", 0.001),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)

    return trainer, model, data_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model for AnimalCLEF")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    args = parser.parse_args()

    train_baseline_model(args.config)
