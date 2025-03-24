import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from animalclef.models.baseline import DinoClassifier
from animalclef.train.baseline import DinoFeatureDataset

logger = logging.getLogger(__name__)


def evaluate_model(
    model: DinoClassifier,
    test_features: Dict[str, np.ndarray],
    test_labels: np.ndarray,
    class_names: List[str] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier model on test data.

    Args:
        model: Trained DinoClassifier model
        test_features: Dict with 'cls' and 'avg_patch' features for testing
        test_labels: Labels for test data
        class_names: Optional list of class names for visualization
        config: Configuration dictionary

    Returns:
        Dictionary with evaluation results
    """
    if config is None:
        config = {}

    default_config = {
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "experiments/baseline/evaluation",
    }

    for k, v in config.items():
        default_config[k] = v
    config = default_config

    # Create test dataset and loader
    test_dataset = DinoFeatureDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Set model to evaluation mode
    model = model.to(config["device"])
    model.eval()

    # Collect predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = {k: v.to(config["device"]) for k, v in features.items()}
            outputs = model(features)
            _, preds = torch.max(outputs["logits"], 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = np.mean(all_preds == all_labels)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Save results
    os.makedirs(config["save_dir"], exist_ok=True)

    # Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config["save_dir"], "confusion_matrix.png"))
    plt.close()

    # Log results
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(
        f"Classification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}"
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = None
) -> Tuple[DinoClassifier, Dict[str, Any]]:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device to load the model to

    Returns:
        Tuple of (loaded model, config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = DinoClassifier(
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_rate=config["dropout_rate"],
        use_cls_token=config["use_cls_token"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config
