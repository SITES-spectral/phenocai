"""Model evaluation commands."""

import click
from pathlib import Path
import tensorflow as tf
import json
import pandas as pd
from typing import Optional

from ...evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_samples,
    analyze_errors
)
from ...data.dataloader import load_dataset_from_csv, create_data_loaders
from ...config.setup import config as app_config


@click.group()
def evaluate():
    """Evaluate trained models."""
    pass


@evaluate.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--split', type=click.Choice(['train', 'val', 'test', 'all']), 
              default='test', help='Dataset split to evaluate')
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory for results')
@click.option('--batch-size', type=int, default=32, 
              help='Batch size for evaluation')
@click.option('--save-predictions', is_flag=True, 
              help='Save individual predictions')
@click.option('--plot-samples', type=int, default=0,
              help='Number of sample predictions to plot')
@click.option('--analyze-errors', is_flag=True,
              help='Perform detailed error analysis')
def model(
    model_path: str,
    dataset_path: str,
    split: str,
    output_dir: Optional[str],
    batch_size: int,
    save_predictions: bool,
    plot_samples: int,
    analyze_errors_flag: bool
):
    """Evaluate a trained model on dataset.
    
    Examples:
        # Evaluate on test set
        phenocai evaluate model model.h5 dataset.csv
        
        # Evaluate with full analysis
        phenocai evaluate model model_dir/final_model.h5 dataset.csv \\
            --output-dir results --save-predictions --plot-samples 16
    """
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = model_path.parent / 'evaluation' / split
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n=== Evaluating Model ===")
    click.echo(f"Model: {model_path}")
    click.echo(f"Dataset: {dataset_path}")
    click.echo(f"Split: {split}")
    click.echo(f"Output: {output_dir}")
    
    try:
        # Load model
        click.echo("\nLoading model...")
        if model_path.is_dir():
            # Assume it's a saved model directory
            model = tf.keras.models.load_model(model_path)
        else:
            # Assume it's an H5 file
            model = tf.keras.models.load_model(str(model_path))
        
        # Get dataset based on split
        if split == 'all':
            # Load entire dataset
            dataset, info = load_dataset_from_csv(
                str(dataset_path),
                batch_size=batch_size,
                shuffle=False,
                augment=False
            )
            class_names = info.get('class_names', ['no_snow', 'snow'])
        else:
            # Create train/val/test splits
            train_dataset, val_dataset, test_dataset, info = create_data_loaders(
                str(dataset_path),
                batch_size=batch_size,
                augment_train=False,  # No augmentation for evaluation
                cache=True
            )
            
            # Select appropriate split
            if split == 'train':
                dataset = train_dataset
                num_samples = info['num_train_samples']
            elif split == 'val':
                dataset = val_dataset
                num_samples = info['num_val_samples']
            else:  # test
                dataset = test_dataset
                num_samples = info['num_test_samples']
            
            class_names = info.get('class_names', ['no_snow', 'snow'])
            click.echo(f"Evaluating on {num_samples} samples")
        
        # Evaluate model
        click.echo("\nRunning evaluation...")
        results = evaluate_model(
            model=model,
            test_dataset=dataset,
            output_dir=str(output_dir),
            class_names=class_names,
            save_predictions=save_predictions
        )
        
        # Display results
        click.echo("\n=== Evaluation Results ===")
        for metric, value in results.items():
            if isinstance(value, float):
                click.echo(f"{metric}: {value:.4f}")
            else:
                click.echo(f"{metric}: {value}")
        
        # Plot sample predictions if requested
        if plot_samples > 0:
            click.echo(f"\nPlotting {plot_samples} sample predictions...")
            plot_prediction_samples(
                model=model,
                dataset=dataset,
                num_samples=plot_samples,
                class_names=class_names,
                save_path=output_dir / 'sample_predictions.png'
            )
        
        # Analyze errors if requested
        if analyze_errors_flag:
            click.echo("\nAnalyzing errors...")
            errors = analyze_errors(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                class_names=class_names
            )
            click.echo(f"Found {len(errors)} errors")
        
        click.echo(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"\n❌ Evaluation failed: {str(e)}", err=True)
        raise click.Abort()


@evaluate.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('predictions_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Output directory for comparison')
@click.option('--threshold', type=float, default=0.5,
              help='Threshold for binary predictions')
def compare(
    dataset_path: str,
    predictions_path: str,
    output_dir: Optional[str],
    threshold: float
):
    """Compare predictions with ground truth.
    
    Loads predictions from a JSON file and compares with ground truth labels.
    """
    dataset_path = Path(dataset_path)
    predictions_path = Path(predictions_path)
    
    if output_dir is None:
        output_dir = predictions_path.parent / 'comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n=== Comparing Predictions ===")
    click.echo(f"Ground truth: {dataset_path}")
    click.echo(f"Predictions: {predictions_path}")
    
    try:
        # Load ground truth
        df = pd.read_csv(dataset_path)
        y_true = df['has_snow'].values
        
        # Load predictions
        with open(predictions_path, 'r') as f:
            pred_data = json.load(f)
        
        if 'y_pred' in pred_data:
            y_pred = pred_data['y_pred']
        elif 'y_pred_proba' in pred_data:
            y_pred_proba = pred_data['y_pred_proba']
            y_pred = [1 if p > threshold else 0 for p in y_pred_proba]
        else:
            raise ValueError("Predictions file must contain 'y_pred' or 'y_pred_proba'")
        
        # Ensure same length
        if len(y_true) != len(y_pred):
            click.echo(f"⚠️  Warning: Length mismatch (truth: {len(y_true)}, pred: {len(y_pred)})")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # Calculate metrics
        from ...evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Display results
        click.echo("\n=== Comparison Results ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"{metric}: {value:.4f}")
            else:
                click.echo(f"{metric}: {value}")
        
        # Generate confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=['no_snow', 'snow'],
            save_path=output_dir / 'comparison_confusion_matrix.png'
        )
        
        # Find disagreements
        disagreements = []
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                disagreements.append({
                    'index': i,
                    'file_path': df.iloc[i]['file_path'] if 'file_path' in df.columns else None,
                    'true_label': int(true),
                    'pred_label': int(pred)
                })
        
        # Save disagreements
        with open(output_dir / 'disagreements.json', 'w') as f:
            json.dump({
                'total_disagreements': len(disagreements),
                'disagreement_rate': len(disagreements) / len(y_true),
                'samples': disagreements[:100]  # First 100
            }, f, indent=2)
        
        click.echo(f"\nFound {len(disagreements)} disagreements ({len(disagreements)/len(y_true):.1%})")
        click.echo(f"Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"\n❌ Comparison failed: {str(e)}", err=True)
        raise click.Abort()


@evaluate.command()
@click.option('--models-dir', type=click.Path(exists=True), 
              help='Directory containing models')
@click.option('--dataset-path', type=click.Path(exists=True), required=True, 
              help='Dataset for evaluation')
@click.option('--output-path', '-o', type=click.Path(), 
              help='Output path for report')
@click.option('--batch-size', type=int, default=32,
              help='Batch size for evaluation')
def benchmark(
    models_dir: Optional[str],
    dataset_path: str,
    output_path: Optional[str],
    batch_size: int
):
    """Benchmark multiple models on same dataset.
    
    Evaluates all models in a directory and generates a comparison report.
    """
    dataset_path = Path(dataset_path)
    
    if models_dir is None:
        models_dir = app_config.model_output_dir
    models_dir = Path(models_dir)
    
    if output_path is None:
        output_path = models_dir / 'benchmark_results.csv'
    output_path = Path(output_path)
    
    click.echo(f"\n=== Benchmarking Models ===")
    click.echo(f"Models directory: {models_dir}")
    click.echo(f"Dataset: {dataset_path}")
    
    if not models_dir.exists():
        click.echo("❌ Models directory not found", err=True)
        raise click.Abort()
    
    try:
        # Find all model files
        model_files = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Look for model files in subdirectories
                for model_file in model_dir.glob('*.h5'):
                    model_files.append(model_file)
                # Also check for SavedModel format
                if (model_dir / 'saved_model.pb').exists():
                    model_files.append(model_dir)
        
        if not model_files:
            click.echo("No models found in directory")
            return
        
        click.echo(f"Found {len(model_files)} models to benchmark")
        
        # Load test dataset once
        _, _, test_dataset, info = create_data_loaders(
            str(dataset_path),
            batch_size=batch_size,
            augment_train=False,
            cache=True
        )
        class_names = info.get('class_names', ['no_snow', 'snow'])
        
        # Benchmark each model
        results = []
        for model_file in model_files:
            click.echo(f"\nEvaluating: {model_file.name}")
            
            try:
                # Load model
                model = tf.keras.models.load_model(str(model_file))
                
                # Evaluate
                metrics = evaluate_model(
                    model=model,
                    test_dataset=test_dataset,
                    class_names=class_names,
                    save_predictions=False
                )
                
                # Add model info
                metrics['model_name'] = model_file.parent.name if model_file.suffix == '.h5' else model_file.name
                metrics['model_path'] = str(model_file)
                
                # Get model size
                if model_file.is_file():
                    metrics['model_size_mb'] = model_file.stat().st_size / (1024 * 1024)
                else:
                    # Estimate size for SavedModel
                    total_size = sum(f.stat().st_size for f in model_file.rglob('*') if f.is_file())
                    metrics['model_size_mb'] = total_size / (1024 * 1024)
                
                results.append(metrics)
                
                # Display key metrics
                click.echo(f"  Accuracy: {metrics['accuracy']:.4f}")
                click.echo(f"  F1 Score: {metrics['f1_score']:.4f}")
                if 'auc' in metrics:
                    click.echo(f"  AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                click.echo(f"  ❌ Failed: {str(e)}", err=True)
                continue
        
        if results:
            # Create comparison DataFrame
            df = pd.DataFrame(results)
            
            # Sort by accuracy
            df = df.sort_values('accuracy', ascending=False)
            
            # Save results
            df.to_csv(output_path, index=False)
            click.echo(f"\n=== Benchmark Summary ===")
            
            # Display top models
            click.echo("\nTop 5 Models by Accuracy:")
            for idx, row in df.head(5).iterrows():
                click.echo(f"{idx+1}. {row['model_name']}: {row['accuracy']:.4f}")
            
            # Best model overall
            best_model = df.iloc[0]
            click.echo(f"\n🏆 Best Model: {best_model['model_name']}")
            click.echo(f"   Accuracy: {best_model['accuracy']:.4f}")
            click.echo(f"   F1 Score: {best_model['f1_score']:.4f}")
            if 'auc' in best_model:
                click.echo(f"   AUC: {best_model['auc']:.4f}")
            
            click.echo(f"\nFull results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"\n❌ Benchmarking failed: {str(e)}", err=True)
        raise click.Abort()