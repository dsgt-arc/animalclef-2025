"""
Training script for triplet learning with PyTorch Lightning.

This script demonstrates how to train a triplet learning model for animal re-identification
using a frozen DINO backbone and a SimpleEmbedder projection.

Generated using Claude Sonnet 3.7
"""

import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch.multiprocessing as mp
from .datamodule import AnimalTripletDataModule
from .learning import TripletLearningModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a triplet learning model for animal re-identification')
    
    # Data arguments
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    
    # Model arguments
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-base', 
                      help='DINO model name')
    parser.add_argument('--embed_type', type=str, default='cls', choices=['cls', 'avg_patch'],
                      help='Type of embedding to use from DINO')
    parser.add_argument('--embedding_dim', type=int, default=768, 
                      help='Input dimension from DINO model')
    parser.add_argument('--projection_dim', type=int, default=128, 
                      help='Output dimension after projection')
    parser.add_argument('--margin', type=float, default=1.0, 
                      help='Margin for triplet loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Number of workers for data loading')
    parser.add_argument('--max_epochs', type=int, default=50, 
                      help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                      help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=5, 
                      help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                      help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', 
                      help='Directory to save logs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up data module
    data_module = AnimalTripletDataModule(
        metadata_path=args.metadata_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.dino_model,
        embed_type=args.embed_type,
    )
    
    # Set up model
    model = TripletLearningModule(
        dino_model_name=args.dino_model,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        margin=args.margin,
        embed_type=args.embed_type,
    )
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )
    
    # Set up logger
    logger_tb = TensorBoardLogger(
        save_dir=args.log_dir,
        name='animal_reid',
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger_tb,
        log_every_n_steps=10,
        accelerator='auto',  # Automatically detect GPU
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    logger.info("Testing the model...")
    trainer.test(model, data_module)
    
    logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    # setup basic logging
    mp.set_start_method('spawn')
    
    logging.basicConfig(level=logging.INFO)
    main()