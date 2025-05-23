"""Model training commands."""

import click
from pathlib import Path
from typing import Optional
import json

from ...training import train_model, ModelTrainer
from ...models.config import (
    MobileNetConfig, 
    CustomCNNConfig, 
    TrainingConfig,
    PRESET_CONFIGS
)
from ...config.setup import config as app_config


@click.group()
def train():
    """Train machine learning models."""
    pass


@train.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--model-type', type=click.Choice(['mobilenet', 'custom_cnn']), 
              default='mobilenet', help='Model architecture to train')
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory for models')
@click.option('--preset', type=click.Choice(list(PRESET_CONFIGS.keys())), 
              help='Use a preset configuration')
@click.option('--config-file', type=click.Path(exists=True), 
              help='Path to configuration YAML file')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--batch-size', type=int, help='Batch size for training')
@click.option('--learning-rate', type=float, help='Initial learning rate')
@click.option('--fine-tune', is_flag=True, help='Enable fine-tuning for transfer learning')
@click.option('--resume-from', type=click.Path(exists=True), 
              help='Resume training from checkpoint')
def model(
    dataset_path: str,
    model_type: str,
    output_dir: Optional[str],
    preset: Optional[str],
    config_file: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    fine_tune: bool,
    resume_from: Optional[str]
):
    """Train a model on the specified dataset.
    
    Examples:
        # Quick training with preset
        phenocai train model dataset.csv --preset mobilenet_quick
        
        # Full training with custom parameters
        phenocai train model dataset.csv --model-type custom_cnn --epochs 50 --batch-size 32
        
        # Fine-tune a pre-trained model
        phenocai train model dataset.csv --preset mobilenet_full --fine-tune
    """
    dataset_path = Path(dataset_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = app_config.model_output_dir / f"{model_type}_{dataset_path.stem}"
    output_dir = Path(output_dir)
    
    click.echo(f"\n=== Training {model_type} Model ===")
    click.echo(f"Dataset: {dataset_path}")
    click.echo(f"Output directory: {output_dir}")
    
    # Build configuration
    config_overrides = {}
    if epochs is not None:
        config_overrides['epochs'] = epochs
    if batch_size is not None:
        config_overrides['batch_size'] = batch_size
    if learning_rate is not None:
        config_overrides['learning_rate'] = learning_rate
    
    try:
        # Train model
        history = train_model(
            dataset_csv=str(dataset_path),
            model_type=model_type,
            output_dir=str(output_dir),
            preset=preset,
            config_file=config_file,
            **config_overrides
        )
        
        # Display final metrics
        final_epoch = len(history['loss']) - 1
        click.echo(f"\n=== Training Complete ===")
        click.echo(f"Final epoch {final_epoch + 1} metrics:")
        for metric, values in history.items():
            if values:  # Check if list is not empty
                click.echo(f"  {metric}: {values[-1]:.4f}")
        
        click.echo(f"\nModel saved to: {output_dir}")
        click.echo(f"View training progress with: tensorboard --logdir {output_dir / 'logs'}")
        
    except Exception as e:
        click.echo(f"\n❌ Training failed: {str(e)}", err=True)
        raise click.Abort()


@train.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--epochs', type=int, default=20, help='Number of fine-tuning epochs')
@click.option('--learning-rate', type=float, default=0.0001, 
              help='Learning rate for fine-tuning')
@click.option('--unfreeze-from', type=int, default=100, 
              help='Layer index to unfreeze from')
def fine_tune(
    model_dir: str,
    dataset_path: str,
    epochs: int,
    learning_rate: float,
    unfreeze_from: int
):
    """Fine-tune a pre-trained model on new data.
    
    This unfreezes later layers of the model and trains with a lower
    learning rate to adapt to new data while preserving learned features.
    """
    model_dir = Path(model_dir)
    dataset_path = Path(dataset_path)
    
    click.echo(f"\n=== Fine-tuning Model ===")
    click.echo(f"Model: {model_dir}")
    click.echo(f"Dataset: {dataset_path}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Unfreezing from layer: {unfreeze_from}")
    
    try:
        # Load existing configuration
        config_path = model_dir / 'config' / 'model_config.yaml'
        if not config_path.exists():
            raise ValueError(f"Model configuration not found at {config_path}")
        
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update for fine-tuning
        config_dict['fine_tune_from'] = unfreeze_from
        config_dict['fine_tune_epochs'] = epochs
        config_dict['fine_tune_learning_rate'] = learning_rate
        
        # Create model config
        if 'filters' in config_dict:
            model_config = CustomCNNConfig(**config_dict)
        else:
            model_config = MobileNetConfig(**config_dict)
        
        # Create trainer
        training_config = TrainingConfig()
        output_dir = model_dir.parent / f"{model_dir.name}_finetuned"
        
        trainer = ModelTrainer(model_config, training_config, str(output_dir))
        
        # Load pre-trained weights
        model_path = model_dir / 'best_model.h5'
        if not model_path.exists():
            model_path = model_dir / 'final_model.h5'
        
        trainer.build_model()
        trainer.model.load_weights(str(model_path))
        
        # Fine-tune
        history = trainer.train(str(dataset_path), fine_tune=True)
        
        click.echo(f"\n✓ Fine-tuning complete!")
        click.echo(f"Fine-tuned model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"\n❌ Fine-tuning failed: {str(e)}", err=True)
        raise click.Abort()


@train.command()
def list_models():
    """List available trained models."""
    model_dir = app_config.model_output_dir
    
    if not model_dir.exists():
        click.echo("No models directory found")
        return
    
    click.echo(f"\n=== Trained Models in {model_dir} ===")
    
    # Look for model directories
    model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        click.echo("No trained models found")
        return
    
    for model_path in sorted(model_dirs):
        click.echo(f"\n📁 {model_path.name}")
        
        # Check for model files
        h5_files = list(model_path.glob("*.h5"))
        if h5_files:
            for h5_file in h5_files:
                size_mb = h5_file.stat().st_size / (1024 * 1024)
                click.echo(f"  • {h5_file.name} ({size_mb:.1f} MB)")
        
        # Check for evaluation results
        eval_file = model_path / 'evaluation_results.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results = json.load(f)
            click.echo("  • Evaluation metrics:")
            for metric, value in results.items():
                click.echo(f"    - {metric}: {value:.4f}")
        
        # Check for config
        config_file = model_path / 'config' / 'model_config.yaml'
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            click.echo(f"  • Model type: {config.get('name', 'unknown')}")
            click.echo(f"  • Epochs: {config.get('epochs', 'unknown')}")


@train.command()
def list_presets():
    """List available training presets."""
    click.echo("\n=== Available Training Presets ===")
    
    for name, config in PRESET_CONFIGS.items():
        click.echo(f"\n{name}:")
        click.echo(f"  Model: {config.name}")
        click.echo(f"  Epochs: {config.epochs}")
        click.echo(f"  Batch size: {config.batch_size}")
        
        if hasattr(config, 'filters'):
            click.echo(f"  Filters: {config.filters}")
        if hasattr(config, 'freeze_base'):
            click.echo(f"  Freeze base: {config.freeze_base}")
        if hasattr(config, 'ensemble_method'):
            click.echo(f"  Ensemble method: {config.ensemble_method}")


@train.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--output-file', '-o', type=click.Path(),
              help='Output file for analysis results')
@click.option('--sample-size', type=int, default=100,
              help='Number of images to sample for analysis')
def analyze_dataset(dataset_path: str, output_file: Optional[str], sample_size: int):
    """Analyze dataset suitability for training.
    
    Checks class balance, quality issues, and provides recommendations.
    """
    import pandas as pd
    from collections import Counter
    
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    
    click.echo(f"\n=== Dataset Analysis: {dataset_path.name} ===")
    click.echo(f"Total samples: {len(df)}")
    
    # Class distribution
    if 'has_snow' in df.columns:
        snow_count = df['has_snow'].sum()
        no_snow_count = len(df) - snow_count
        snow_pct = snow_count / len(df) * 100
        
        click.echo(f"\nClass distribution:")
        click.echo(f"  Snow: {snow_count} ({snow_pct:.1f}%)")
        click.echo(f"  No snow: {no_snow_count} ({100-snow_pct:.1f}%)")
        
        # Balance assessment
        if snow_pct < 20 or snow_pct > 80:
            click.echo("\n⚠️  WARNING: Highly imbalanced dataset!")
            click.echo("  Consider using class weights or data augmentation")
        elif snow_pct < 30 or snow_pct > 70:
            click.echo("\n⚠️  Dataset is moderately imbalanced")
            click.echo("  Class weights recommended")
        else:
            click.echo("\n✓ Dataset is reasonably balanced")
    
    # Quality flags analysis
    if 'has_flags' in df.columns:
        flagged = df['has_flags'].sum()
        flagged_pct = flagged / len(df) * 100
        
        click.echo(f"\nQuality issues:")
        click.echo(f"  Images with flags: {flagged} ({flagged_pct:.1f}%)")
        
        if flagged_pct > 50:
            click.echo("\n⚠️  WARNING: Many images have quality issues!")
            click.echo("  Consider filtering dataset before training")
    
    # Temporal distribution
    if 'month' in df.columns:
        click.echo("\nTemporal distribution:")
        month_counts = df['month'].value_counts().sort_index()
        for month, count in month_counts.items():
            pct = count / len(df) * 100
            click.echo(f"  Month {month}: {count} ({pct:.1f}%)")
    
    # Station distribution
    if 'station_name' in df.columns:
        click.echo("\nStation distribution:")
        station_counts = df['station_name'].value_counts()
        for station, count in station_counts.items():
            pct = count / len(df) * 100
            click.echo(f"  {station}: {count} ({pct:.1f}%)")
    
    # Recommendations
    click.echo("\n=== Recommendations ===")
    
    min_samples = 1000
    if len(df) < min_samples:
        click.echo(f"⚠️  Dataset size ({len(df)}) is below recommended minimum ({min_samples})")
        click.echo("  Consider collecting more data or using data augmentation")
    else:
        click.echo(f"✓ Dataset size ({len(df)}) is sufficient for training")
    
    # Save detailed analysis if requested
    if output_file:
        analysis = {
            'dataset': str(dataset_path),
            'total_samples': len(df),
            'columns': list(df.columns),
            'class_distribution': {
                'snow': int(snow_count) if 'has_snow' in df.columns else None,
                'no_snow': int(no_snow_count) if 'has_snow' in df.columns else None,
                'balance_ratio': float(snow_pct / 100) if 'has_snow' in df.columns else None
            },
            'quality_issues': {
                'flagged_images': int(flagged) if 'has_flags' in df.columns else None,
                'flagged_percentage': float(flagged_pct) if 'has_flags' in df.columns else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        click.echo(f"\nDetailed analysis saved to: {output_file}")