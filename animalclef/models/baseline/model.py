import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import AutoModel


class DinoClassifierWithExtractor(pl.LightningModule):
    """
    A PyTorch Lightning module that combines DINOv2 vision transformer feature extraction
    with a classification head for image classification tasks.

    This model uses a pre-trained DINO vision transformer as a backbone and adds a
    classification layer on top. It supports using either the CLS token or the mean
    of patch embeddings as input to the classifier.

    The backbone can be either frozen (transfer learning) or fine-tuned (full training).
    """

    def __init__(
        self,
        num_classes: int,
        dino_model_name: str = "facebook/dinov2-base",
        hidden_dim: int | None = 512,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        freeze_backbone: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor (DINO model)
        self.feature_extractor = AutoModel.from_pretrained(dino_model_name)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Build classifier
        embedding_dim = self.feature_extractor.config.hidden_size
        self.classifier = self._build_classifier(
            embedding_dim, hidden_dim, dropout_rate, num_classes
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Configuration
        self.use_cls_token = use_cls_token

    def _build_classifier(self, embedding_dim, hidden_dim, dropout_rate, num_classes):
        """Helper method to build the classifier network."""
        if hidden_dim is not None:
            return nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            return nn.Linear(embedding_dim, num_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def _extract_embeddings(self, pixel_values):
        outputs = self.feature_extractor(pixel_values, output_hidden_states=True)
        if self.use_cls_token:
            # Use pooler_output for CLS
            return outputs.pooler_output
        else:
            # mean of patch embeddings
            return outputs.hidden_states[-1][:, 1:, :].mean(dim=1)

    def forward(self, pixel_values):
        """Complete forward pass from images to class predictions."""
        embeddings = self._extract_embeddings(pixel_values)
        return self.classifier(embeddings)

    def _step(self, batch, metric, prefix, prefix_kwargs={}):
        """Shared step logic for train/val/test."""
        images, targets = batch
        logits = self(images)
        loss = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        metric(preds, targets)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", metric, prog_bar=True, **prefix_kwargs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(
            batch,
            self.train_acc,
            "train",
            prefix_kwargs={"on_step": False, "on_epoch": True},
        )

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.val_acc, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, self.test_acc, "test")

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        logits = self(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return {"probs": probs, "preds": preds}
