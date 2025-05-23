"""Model configuration classes."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    name: str
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class MobileNetConfig(ModelConfig):
    """Configuration for MobileNetV2 transfer learning."""
    name: str = "mobilenet_v2"
    dropout_rate: float = 0.2
    freeze_base: bool = True
    fine_tune_from: Optional[int] = None
    fine_tune_epochs: int = 20
    fine_tune_learning_rate: float = 0.0001
    dense_units: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class CustomCNNConfig(ModelConfig):
    """Configuration for custom CNN."""
    name: str = "custom_cnn"
    filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    dense_units: List[int] = field(default_factory=lambda: [512, 256])


@dataclass
class EnsembleConfig(ModelConfig):
    """Configuration for ensemble models."""
    name: str = "ensemble"
    base_model_configs: List[ModelConfig] = field(default_factory=list)
    ensemble_method: str = "average"  # 'average', 'weighted', 'stacking'
    meta_learner_units: List[int] = field(default_factory=lambda: [32])


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2
    
    # Data augmentation
    augmentation_enabled: bool = True
    horizontal_flip: bool = True
    rotation_range: float = 0.1
    zoom_range: float = 0.1
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    
    # Training parameters
    shuffle: bool = True
    random_seed: int = 42
    num_workers: int = 4
    prefetch_buffer: int = 2
    
    # Class weights
    use_class_weights: bool = True
    class_weight_strategy: str = "balanced"  # 'balanced' or 'custom'
    custom_class_weights: Optional[Dict[int, float]] = None
    
    # Callbacks
    use_tensorboard: bool = True
    use_model_checkpoint: bool = True
    use_early_stopping: bool = True
    use_reduce_lr: bool = True
    
    # Validation
    validation_frequency: int = 1
    save_best_only: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# Preset configurations
PRESET_CONFIGS = {
    "mobilenet_quick": MobileNetConfig(
        epochs=10,
        batch_size=64,
        freeze_base=True,
        dropout_rate=0.2
    ),
    
    "mobilenet_full": MobileNetConfig(
        epochs=50,
        batch_size=32,
        freeze_base=True,
        fine_tune_from=100,
        fine_tune_epochs=20,
        dropout_rate=0.3
    ),
    
    "custom_cnn_small": CustomCNNConfig(
        filters=[16, 32, 64, 128],
        epochs=30,
        batch_size=64,
        dropout_rate=0.2
    ),
    
    "custom_cnn_large": CustomCNNConfig(
        filters=[64, 128, 256, 512],
        epochs=50,
        batch_size=32,
        dropout_rate=0.4
    ),
    
    "ensemble_simple": EnsembleConfig(
        ensemble_method="average",
        epochs=10  # For fine-tuning ensemble weights
    ),
    
    "ensemble_stacking": EnsembleConfig(
        ensemble_method="stacking",
        epochs=20,
        meta_learner_units=[64, 32]
    )
}


def get_preset_config(preset_name: str) -> ModelConfig:
    """Get a preset model configuration.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        ModelConfig instance
        
    Raises:
        ValueError: If preset not found
    """
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    
    return PRESET_CONFIGS[preset_name]


def load_config_from_yaml(yaml_path: str) -> ModelConfig:
    """Load model configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        ModelConfig instance
    """
    import yaml
    from pathlib import Path
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Determine model type
    model_type = config_dict.get('model_type', 'mobilenet')
    
    if model_type == 'mobilenet':
        return MobileNetConfig(**config_dict)
    elif model_type == 'custom_cnn':
        return CustomCNNConfig(**config_dict)
    elif model_type == 'ensemble':
        return EnsembleConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_config_to_yaml(config: ModelConfig, yaml_path: str):
    """Save model configuration to YAML file.
    
    Args:
        config: ModelConfig instance
        yaml_path: Path to save YAML file
    """
    import yaml
    from pathlib import Path
    
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    config_dict['model_type'] = config.__class__.__name__.replace('Config', '').lower()
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)