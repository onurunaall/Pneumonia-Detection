from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PathConfig:
    """File path configurations"""
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models/saved")
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")

@dataclass
class DataConfig:
    """Data processing configurations"""
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    batch_size: int = 32
    num_workers: int = 4
    validation_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42

@dataclass
class ModelConfig:
    """Model architecture configurations"""
    architecture: str = "vgg16"
    pretrained: bool = True
    num_classes: int = 1  # Binary classification
    dropout_rate: float = 0.5
    freeze_base: bool = True
    unfreeze_layers: int = 0

@dataclass
class TrainingConfig:
    """Training configurations"""
    epochs: int = 50
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    metrics: List[str] = None
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    class_weights: bool = True  # Handle imbalance

@dataclass
class InferenceConfig:
    """Inference configurations"""
    threshold: float = 0.51
    allowed_modality: str = "DX"
    allowed_body_part: str = "CHEST"
    allowed_positions: List[str] = None

    def __post_init__(self):
        if self.allowed_positions is None:
            self.allowed_positions = ["PA", "AP"]

@dataclass
class Config:
    """Main configuration class"""
    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()

    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.paths.data_dir, self.paths.processed_dir, 
                     self.paths.model_dir, self.paths.log_dir,
                     self.paths.checkpoint_dir]:
            path.mkdir(parents=True, exist_ok=True)

# Singleton instance
config = Config()
