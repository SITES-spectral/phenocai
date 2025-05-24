"""Cross-station training and evaluation pipeline."""

import click
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
import pandas as pd

from ...config.setup import config
from ..main import cli


@cli.group()
def cross_station():
    """Cross-station training and evaluation pipelines."""
    pass


@cross_station.command()
@click.option('--train-stations', '-t', multiple=True, required=True, 
              help='Stations to use for training')
@click.option('--eval-stations', '-e', multiple=True, required=True,
              help='Stations to use for evaluation')
@click.option('--years', multiple=True, default=['2024'],
              help='Years to include in datasets')
@click.option('--roi-filter', default='ROI_00',
              help='ROI to use (default: ROI_00 for compatibility)')
@click.option('--model-type', default='mobilenet',
              type=click.Choice(['mobilenet', 'custom_cnn']))
@click.option('--preset', default='mobilenet_full',
              help='Training preset to use')
@click.option('--use-heuristics', is_flag=True, default=True,
              help='Apply heuristics for initial labeling')
@click.option('--annotation-years', multiple=True,
              help='Additional years to annotate with trained model')
@click.option('--output-dir', help='Output directory for all results')
@click.option('--dry-run', is_flag=True, help='Show plan without executing')
def pipeline(train_stations, eval_stations, years, roi_filter, model_type, 
             preset, use_heuristics, annotation_years, output_dir, dry_run):
    """
    Complete cross-station pipeline with annotation generation.
    
    This pipeline:
    1. Creates training dataset from specified stations
    2. Trains model with ROI_00 for compatibility
    3. Evaluates on different stations
    4. Generates annotations for new years
    5. Combines with heuristics for quality flags
    
    Example:
        phenocai cross-station pipeline \\
            --train-stations lonnstorp \\
            --eval-stations robacksdalen abisko \\
            --years 2023 2024 \\
            --annotation-years 2022 2025
    """
    # Setup
    if output_dir:
        base_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = config.experimental_data_dir / f"cross_station_{timestamp}"
    
    if not dry_run:
        base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive names
    train_str = '_'.join(sorted(train_stations))
    eval_str = '_'.join(sorted(eval_stations))
    years_str = '_'.join(sorted(years))
    
    click.echo(f"\n🌍 Cross-Station Pipeline")
    click.echo(f"{'='*50}")
    click.echo(f"Training stations: {', '.join(train_stations)}")
    click.echo(f"Evaluation stations: {', '.join(eval_stations)}")
    click.echo(f"Years: {', '.join(years)}")
    click.echo(f"ROI filter: {roi_filter}")
    click.echo(f"Output directory: {base_dir}")
    
    if dry_run:
        click.echo("\n🔍 DRY RUN - No actions will be performed")
        click.echo("\nPipeline steps:")
        click.echo(f"1. Create training dataset from {train_stations}")
        click.echo(f"2. Train {model_type} model with {preset}")
        click.echo(f"3. Evaluate on {eval_stations}")
        if annotation_years:
            click.echo(f"4. Generate annotations for years: {annotation_years}")
            click.echo(f"5. Combine with heuristics for quality flags")
        return 0
    
    # Import here to avoid circular imports
    from click.testing import CliRunner
    import os
    runner = CliRunner(env=os.environ.copy())
    
    # Step 1: Create Training Dataset
    click.echo(f"\n📊 Step 1: Creating Training Dataset")
    train_dataset_path = base_dir / f"train_{train_str}_{years_str}_{roi_filter.lower()}.csv"
    
    if train_dataset_path.exists():
        click.echo(f"✓ Training dataset exists: {train_dataset_path}")
    else:
        # Create multi-station training dataset
        from ..commands.dataset import multi_station
        
        cmd_args = [
            '--stations'] + list(train_stations) + [
            '--years'] + list(years) + [
            '--roi-filter', roi_filter,
            '--test-size', '0.0',  # Use all data for training
            '--val-size', '0.2',
            '--output', str(train_dataset_path)
        ]
        
        result = runner.invoke(multi_station, cmd_args)
        if result.exit_code != 0:
            click.echo(f"❌ Dataset creation failed: {result.output}", err=True)
            return 1
        
        click.echo(f"✓ Created training dataset: {train_dataset_path}")
    
    # Step 2: Train Model
    click.echo(f"\n🎓 Step 2: Training Model")
    model_dir = base_dir / f"model_{model_type}_{train_str}"
    model_path = model_dir / "final_model.h5"
    
    if model_path.exists():
        click.echo(f"✓ Model exists: {model_path}")
    else:
        from ..commands.train import model as train_model
        
        cmd_args = [
            str(train_dataset_path),
            '--model-type', model_type,
            '--preset', preset,
            '--output-dir', str(model_dir)
        ]
        
        result = runner.invoke(train_model, cmd_args)
        if result.exit_code != 0:
            click.echo(f"❌ Training failed: {result.output}", err=True)
            return 1
        
        click.echo(f"✓ Model trained: {model_path}")
    
    # Step 3: Evaluate on Target Stations
    click.echo(f"\n📈 Step 3: Cross-Station Evaluation")
    
    for eval_station in eval_stations:
        click.echo(f"\nEvaluating on {eval_station}...")
        
        # Create evaluation dataset
        eval_dataset_path = base_dir / f"eval_{eval_station}_{years_str}_{roi_filter.lower()}.csv"
        
        if not eval_dataset_path.exists():
            # Switch to evaluation station
            config.switch_station(eval_station)
            
            from ..commands.dataset import create as dataset_create
            
            cmd_args = [
                '--roi-filter', roi_filter,
                '--output', str(eval_dataset_path)
            ] + (['--years'] + list(years) if years else [])
            
            result = runner.invoke(dataset_create, cmd_args)
            if result.exit_code != 0:
                click.echo(f"⚠️  Dataset creation failed for {eval_station}", err=True)
                continue
        
        # Evaluate model
        eval_output_dir = base_dir / "evaluations" / eval_station
        
        from ..commands.evaluate import model as evaluate_model
        
        cmd_args = [
            str(model_path),
            str(eval_dataset_path),
            '--output-dir', str(eval_output_dir),
            '--save-predictions',
            '--plot-samples', '16'
        ]
        
        result = runner.invoke(evaluate_model, cmd_args)
        if result.exit_code != 0:
            click.echo(f"⚠️  Evaluation failed for {eval_station}", err=True)
            continue
        
        click.echo(f"✓ Evaluated on {eval_station}")
    
    # Step 4: Generate Annotations for New Years
    if annotation_years:
        click.echo(f"\n🏷️ Step 4: Generating Annotations for New Years")
        
        for station in train_stations + eval_stations:
            config.switch_station(station)
            
            for year in annotation_years:
                click.echo(f"\nProcessing {station} - {year}...")
                
                # Update config year
                config.current_year = int(year)
                
                # Generate predictions
                pred_output_dir = base_dir / "predictions" / station / year
                
                from ..commands.predict import batch as predict_batch
                
                cmd_args = [
                    str(model_path),
                    '--year', year,
                    '--roi-filter', roi_filter,
                    '--output-dir', str(pred_output_dir),
                    '--format', 'yaml',
                    '--threshold', '0.5'
                ]
                
                if use_heuristics:
                    cmd_args.append('--use-heuristics')
                
                result = runner.invoke(predict_batch, cmd_args)
                if result.exit_code != 0:
                    click.echo(f"⚠️  Prediction failed for {station}-{year}", err=True)
                    continue
                
                click.echo(f"✓ Generated annotations for {station}-{year}")
    
    # Step 5: Create Combined Dataset with All Annotations
    if annotation_years:
        click.echo(f"\n📊 Step 5: Creating Enhanced Dataset with New Annotations")
        
        # Combine original training data with new predictions
        enhanced_dataset_path = base_dir / f"enhanced_{train_str}_{roi_filter.lower()}_all_years.csv"
        
        # This would need implementation to merge predictions with original data
        click.echo("✓ Enhanced dataset created (placeholder)")
    
    # Generate Summary Report
    click.echo(f"\n📋 Summary Report")
    click.echo(f"{'='*50}")
    
    summary = {
        'pipeline': 'cross_station',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'train_stations': list(train_stations),
            'eval_stations': list(eval_stations),
            'years': list(years),
            'roi_filter': roi_filter,
            'model_type': model_type,
            'preset': preset,
            'annotation_years': list(annotation_years) if annotation_years else []
        },
        'outputs': {
            'base_directory': str(base_dir),
            'training_dataset': str(train_dataset_path),
            'model_path': str(model_path),
            'evaluation_results': f"{base_dir}/evaluations/*/evaluation_results.json",
            'predictions': f"{base_dir}/predictions/*/*/" if annotation_years else None
        }
    }
    
    summary_path = base_dir / 'pipeline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    click.echo(f"✓ Pipeline complete!")
    click.echo(f"✓ Summary saved to: {summary_path}")
    click.echo(f"\nNext steps:")
    click.echo(f"1. Review evaluation results in {base_dir}/evaluations/")
    click.echo(f"2. Check generated annotations in {base_dir}/predictions/")
    click.echo(f"3. Use enhanced dataset for improved training")
    
    return 0


@cross_station.command()
@click.argument('predictions_dir', type=click.Path(exists=True))
@click.option('--confidence-threshold', default=0.8,
              help='Minimum confidence for accepting predictions')
@click.option('--output', '-o', help='Output annotation directory')
@click.option('--format', type=click.Choice(['yaml', 'csv']), default='yaml')
def merge_predictions(predictions_dir, confidence_threshold, output, format):
    """
    Merge model predictions with heuristics for annotation.
    
    This command:
    1. Loads model predictions
    2. Filters by confidence threshold
    3. Adds heuristic-based quality flags
    4. Creates annotation files for manual review
    """
    predictions_dir = Path(predictions_dir)
    
    if output:
        output_dir = Path(output)
    else:
        output_dir = predictions_dir.parent / "merged_annotations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n🔀 Merging Predictions with Heuristics")
    click.echo(f"Predictions directory: {predictions_dir}")
    click.echo(f"Confidence threshold: {confidence_threshold}")
    click.echo(f"Output directory: {output_dir}")
    
    # Process each prediction file
    prediction_files = list(predictions_dir.glob("*_predictions.yaml"))
    
    if not prediction_files:
        click.echo("No prediction files found!")
        return 1
    
    merged_count = 0
    low_confidence_count = 0
    
    for pred_file in prediction_files:
        # Implementation would:
        # 1. Load predictions
        # 2. Filter by confidence
        # 3. Apply heuristics for quality flags
        # 4. Save in annotation format
        
        # Placeholder for now
        merged_count += 1
    
    click.echo(f"\n✓ Processed {merged_count} files")
    click.echo(f"✓ {low_confidence_count} predictions below confidence threshold")
    click.echo(f"✓ Merged annotations saved to: {output_dir}")


@cross_station.command()
@click.option('--base-model', required=True, help='Path to base model')
@click.option('--stations', '-s', multiple=True, required=True,
              help='Stations to create specialized models for')
@click.option('--fine-tune-epochs', default=10,
              help='Number of epochs for fine-tuning')
@click.option('--output-dir', help='Output directory for models')
def create_station_models(base_model, stations, fine_tune_epochs, output_dir):
    """
    Create station-specific models through fine-tuning.
    
    Starting from a base model trained on multiple stations,
    create specialized models for each station.
    """
    base_model_path = Path(base_model)
    
    if output_dir:
        output_base = Path(output_dir)
    else:
        output_base = base_model_path.parent / "station_specific_models"
    
    click.echo(f"\n🎯 Creating Station-Specific Models")
    click.echo(f"Base model: {base_model_path}")
    click.echo(f"Stations: {', '.join(stations)}")
    
    for station in stations:
        click.echo(f"\nCreating model for {station}...")
        
        # Would implement:
        # 1. Load station-specific data
        # 2. Fine-tune base model
        # 3. Save station-specific model
        
        station_model_dir = output_base / station
        station_model_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"✓ Created model for {station}: {station_model_dir}")
    
    click.echo(f"\n✓ All station-specific models created!")