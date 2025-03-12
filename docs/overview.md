# overview

## module structure

The project is organized to support experimentation with multiple model architectures and training approaches. The main structure is:

```
animalclef/
├── preprocess/      # Data preprocessing pipelines
├── embed/           # Feature extraction modules
├── models/          # Model definitions
│   ├── baseline/    # Cross-entropy baseline models
│   ├── triplet/     # Models with triplet loss
│   ├── attention/   # Models with attention mechanisms
│   └── utils/       # Shared model utilities
├── train/           # Training workflows
├── evaluate/        # Evaluation scripts
└── experiments/     # Experiment configurations and results
    ├── baseline/
    ├── triplet/
    └── attention/
```

### Experiment Structure

Each experiment can be organized as follows:

1. **Configuration**: Define model hyperparameters, training settings, and data splits in a standardized configuration format (YAML or JSON).
2. **Model Definition**: Implement model architecture in the appropriate models/ subdirectory.
3. **Training Workflow**: Create a training script that loads configuration, builds the model, and manages the training loop.
4. **Evaluation**: Run standardized evaluation metrics on validation/test data.
5. **Results Tracking**: Store metrics, model checkpoints, and visualizations in the experiments directory.

### Running Different Models

To implement and compare the different model types:

#### Baseline with Cross Entropy

The baseline model would use a standard classification approach with cross-entropy loss. This serves as a performance benchmark.

```python
# Example usage
from animalclef.models.baseline import CrossEntropyClassifier
from animalclef.train.trainer import train_model

config = load_config("configs/baseline.yaml")
model = CrossEntropyClassifier(config)
train_model(model, data_loader, config)
```

#### Triplet Loss Models

For models using triplet loss, the training pipeline would need to handle triplet generation/mining:

```python
# Example usage
from animalclef.models.triplet import TripletModel
from animalclef.train.triplet_trainer import train_with_triplets

config = load_config("configs/triplet.yaml")
model = TripletModel(config)
train_with_triplets(model, data_loader, triplet_miner, config)
```

#### Attention-Based Models

Attention models would incorporate various attention mechanisms:

```python
# Example usage
from animalclef.models.attention import AttentionClassifier
from animalclef.train.trainer import train_model

config = load_config("configs/attention.yaml")
model = AttentionClassifier(config)
train_model(model, data_loader, config)
```

### Experiment Workflow

A typical experiment workflow would consist of:

1. Preprocess raw data using the existing pipeline
2. Extract embeddings using the DINOv2 (or other) feature extractor
3. Train the specific model variant
4. Evaluate on test data
5. Compare results across different model architectures

The CLI could be extended to support this workflow:

```bash
# Example CLI usage
python -m animalclef train baseline --config configs/baseline.yaml
python -m animalclef train triplet --config configs/triplet.yaml
python -m animalclef train attention --config configs/attention.yaml
python -m animalclef evaluate --model-path experiments/baseline/model.pt
```

This structure enables systematic comparison of different approaches while maintaining code organization and reusability.
