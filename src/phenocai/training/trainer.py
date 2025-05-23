"""Main training pipeline for PhenoCAI models."""

import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import yaml
from datetime import datetime
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from ..models.architectures import (
    create_mobilenet_model, 
    create_custom_cnn,
    compile_model,
    get_model_summary
)
from ..models.config import ModelConfig, TrainingConfig
from ..data.dataloader import create_data_loaders
from .callbacks import (
    create_callbacks,
    TrainingMonitor,
    ConfusionMatrixCallback,
    LearningRateLogger
)


class ModelTrainer:
    """Main class for training PhenoCAI models."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        output_dir: str
    ):
        """Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            output_dir: Directory for outputs
        """
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.config_dir = self.output_dir / 'config'
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and data
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_info = None
        
    def build_model(self) -> tf.keras.Model:
        """Build model based on configuration."""
        if hasattr(self.model_config, 'filters'):  # Custom CNN
            model = create_custom_cnn(
                input_shape=self.model_config.input_shape,
                num_classes=self.model_config.num_classes,
                filters=self.model_config.filters,
                dropout_rate=self.model_config.dropout_rate,
                use_batch_norm=self.model_config.use_batch_norm
            )
        else:  # MobileNet
            model = create_mobilenet_model(
                input_shape=self.model_config.input_shape,
                num_classes=self.model_config.num_classes,
                dropout_rate=self.model_config.dropout_rate,
                freeze_base=self.model_config.freeze_base,
                fine_tune_from=self.model_config.fine_tune_from
            )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=self.model_config.learning_rate,
            num_classes=self.model_config.num_classes
        )
        
        self.model = model
        return model
    
    def prepare_data(self, csv_path: str):
        """Prepare datasets for training.
        
        Args:
            csv_path: Path to dataset CSV
        """
        train_dataset, val_dataset, test_dataset, info = create_data_loaders(
            csv_path=csv_path,
            batch_size=self.model_config.batch_size,
            image_size=self.model_config.input_shape[:2],
            train_size=self.training_config.train_split,
            val_size=self.training_config.val_split,
            test_size=self.training_config.test_split,
            augment_train=self.training_config.augmentation_enabled,
            cache=True,
            random_state=self.training_config.random_seed
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.dataset_info = info
        
        # Log dataset information
        print("\nDataset Information:")
        print(f"  Training samples: {info['num_train_samples']} "
              f"({info['train_snow_percentage']:.1%} with snow)")
        print(f"  Validation samples: {info['num_val_samples']} "
              f"({info['val_snow_percentage']:.1%} with snow)")
        print(f"  Test samples: {info['num_test_samples']} "
              f"({info['test_snow_percentage']:.1%} with snow)")
        
        if self.training_config.use_class_weights:
            print(f"\nClass weights: {info['class_weights']}")
    
    def save_configs(self):
        """Save all configurations."""
        # Save model config
        model_config_path = self.config_dir / 'model_config.yaml'
        with open(model_config_path, 'w') as f:
            yaml.dump(self.model_config.to_dict(), f, default_flow_style=False)
        
        # Save training config
        training_config_path = self.config_dir / 'training_config.yaml'
        with open(training_config_path, 'w') as f:
            yaml.dump(self.training_config.to_dict(), f, default_flow_style=False)
        
        # Save dataset info
        if self.dataset_info:
            dataset_info_path = self.config_dir / 'dataset_info.json'
            with open(dataset_info_path, 'w') as f:
                json.dump(self.dataset_info, f, indent=2, cls=NumpyEncoder)
        
        # Save model summary
        if self.model:
            summary = get_model_summary(self.model)
            summary_path = self.config_dir / 'model_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
            # Save model architecture plot
            try:
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=self.config_dir / 'model_architecture.png',
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB',
                    expand_nested=True,
                    dpi=150
                )
            except Exception as e:
                print(f"Warning: Could not plot model architecture: {e}")
    
    def train(
        self,
        csv_path: str,
        initial_epoch: int = 0,
        fine_tune: bool = False
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            csv_path: Path to dataset CSV
            initial_epoch: Starting epoch (for resuming)
            fine_tune: Whether this is fine-tuning phase
            
        Returns:
            Training history
        """
        # Prepare data if not already done
        if self.train_dataset is None:
            self.prepare_data(csv_path)
        
        # Build model if not already done
        if self.model is None:
            self.build_model()
        
        # Save configurations
        self.save_configs()
        
        # Adjust for fine-tuning
        if fine_tune and hasattr(self.model_config, 'fine_tune_from'):
            print("\nFine-tuning model...")
            # Unfreeze layers
            base_model = self.model.layers[4]  # Assuming standard architecture
            base_model.trainable = True
            
            # Freeze early layers
            for layer in base_model.layers[:self.model_config.fine_tune_from]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            self.model = compile_model(
                self.model,
                learning_rate=self.model_config.fine_tune_learning_rate,
                num_classes=self.model_config.num_classes
            )
        
        # Create callbacks
        callbacks = create_callbacks(
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            monitor=self.training_config.monitor_metric,
            mode=self.training_config.monitor_mode,
            patience=self.model_config.early_stopping_patience,
            reduce_lr_patience=self.model_config.reduce_lr_patience,
            save_best_only=self.training_config.save_best_only
        )
        
        # Add custom callbacks
        callbacks.extend([
            TrainingMonitor(str(self.log_dir)),
            LearningRateLogger(str(self.log_dir))
        ])
        
        # Add confusion matrix callback for validation
        if self.val_dataset:
            callbacks.append(
                ConfusionMatrixCallback(
                    self.val_dataset,
                    self.dataset_info['class_names'],
                    str(self.log_dir),
                    frequency=5
                )
            )
        
        # Get class weights if needed
        class_weight = None
        if self.training_config.use_class_weights and self.dataset_info:
            class_weight = self.dataset_info['class_weights']
        
        # Train model
        print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        history = self.model.fit(
            self.train_dataset,
            epochs=self.model_config.epochs,
            initial_epoch=initial_epoch,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Save final model (weights only to avoid pickle issues)
        final_model_path = self.output_dir / 'final_model.weights.h5'
        self.model.save_weights(final_model_path)
        
        # Also save the full model in Keras format
        keras_model_path = self.output_dir / 'final_model.keras'
        try:
            self.model.save(keras_model_path)
            print(f"\nFull model saved to {keras_model_path}")
        except:
            print(f"\nModel weights saved to {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2, cls=NumpyEncoder)
        
        return history.history
    
    def evaluate(self, dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            dataset: Dataset to evaluate on (uses test set if None)
            
        Returns:
            Dictionary of metrics
        """
        if dataset is None:
            dataset = self.test_dataset
        
        if dataset is None:
            raise ValueError("No dataset available for evaluation")
        
        print("\nEvaluating model...")
        results = self.model.evaluate(dataset, verbose=1)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(results[i])
        
        # Save evaluation results
        eval_path = self.output_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        
        return metrics
    
    def predict_on_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions on test set.
        
        Returns:
            Tuple of (y_true, y_pred)
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset available")
        
        y_true = []
        y_pred = []
        
        print("\nGenerating predictions on test set...")
        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            
            if self.model_config.num_classes == 2:
                y_pred.extend(predictions.flatten())
            else:
                y_pred.extend(np.argmax(predictions, axis=1))
            
            y_true.extend(labels.numpy())
        
        return np.array(y_true), np.array(y_pred)


def train_model(
    dataset_csv: str,
    model_type: str,
    output_dir: str,
    preset: Optional[str] = None,
    config_file: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """High-level function to train a model.
    
    Args:
        dataset_csv: Path to dataset CSV
        model_type: Type of model ('mobilenet' or 'custom_cnn')
        output_dir: Output directory
        preset: Preset configuration name
        config_file: Path to config file
        **kwargs: Additional configuration overrides
        
    Returns:
        Training history
    """
    from ..models.config import (
        MobileNetConfig, 
        CustomCNNConfig,
        TrainingConfig,
        get_preset_config,
        load_config_from_yaml
    )
    
    # Load or create model config
    if config_file:
        model_config = load_config_from_yaml(config_file)
    elif preset:
        model_config = get_preset_config(preset)
    else:
        # Create default config
        if model_type == 'mobilenet':
            model_config = MobileNetConfig()
        elif model_type == 'custom_cnn':
            model_config = CustomCNNConfig()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    
    # Create training config
    training_config = TrainingConfig()
    
    # Create trainer
    trainer = ModelTrainer(model_config, training_config, output_dir)
    
    # Train model
    history = trainer.train(dataset_csv)
    
    # Evaluate on test set
    trainer.evaluate()
    
    return history