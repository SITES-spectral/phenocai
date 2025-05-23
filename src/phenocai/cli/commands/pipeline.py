"""
Complete pipeline commands for end-to-end processing
"""
import click
import time
from pathlib import Path
from datetime import datetime
import logging

from ...config.setup import config
from .dataset import create as dataset_create
from .train import model as train_model
from .evaluate import model as evaluate_model
from .predict import batch as predict_batch

logger = logging.getLogger(__name__)


@click.group()
def pipeline():
    """Complete end-to-end pipeline commands."""
    pass


@pipeline.command()
@click.option('--station', '-s', help='Station to process (overrides current)')
@click.option('--instrument', '-i', help='Instrument to use (overrides current)')
@click.option('--year', '-y', help='Year to process (overrides current)')
@click.option('--prediction-years', '-p', multiple=True, default=['2023', '2024'], 
              help='Years to generate predictions for')
@click.option('--model-type', default='mobilenet', help='Model architecture to use')
@click.option('--preset', default='mobilenet_full', help='Training preset to use')
@click.option('--test-size', default=0.2, help='Test set fraction')
@click.option('--val-size', default=0.1, help='Validation set fraction')
@click.option('--clean-only', is_flag=True, help='Use only clean data (no flags)')
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.option('--output-dir', help='Base output directory for all results')
def full(station, instrument, year, prediction_years, model_type, preset, 
         test_size, val_size, clean_only, dry_run, output_dir):
    """
    Run the complete pipeline: dataset creation → training → evaluation → prediction.
    
    This command orchestrates the entire workflow from raw annotations to
    production predictions for the specified years.
    """
    start_time = time.time()
    
    # Setup output directory
    if output_dir:
        output_base = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = config.experimental_data_dir / f"pipeline_run_{timestamp}"
    
    if not dry_run:
        output_base.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n🚀 PhenoCAI Complete Pipeline")
    click.echo(f"={'='*50}")
    
    # Step 0: Setup and validation
    click.echo(f"\n📋 Step 0: Setup and Validation")
    
    if station:
        if dry_run:
            click.echo(f"Would switch to station: {station}")
        else:
            try:
                config.switch_station(station, instrument)
                click.echo(f"✓ Switched to station: {config.current_station}")
            except ValueError as e:
                click.echo(f"❌ Error switching station: {e}", err=True)
                return 1
    
    if instrument:
        if dry_run:
            click.echo(f"Would switch to instrument: {instrument}")
        else:
            try:
                config.switch_instrument(instrument)
                click.echo(f"✓ Using instrument: {config.current_instrument}")
            except ValueError as e:
                click.echo(f"❌ Error switching instrument: {e}", err=True)
                return 1
    
    if year:
        if not dry_run:
            config.current_year = year
        click.echo(f"✓ Processing year: {year or config.current_year}")
    
    current_station = config.current_station
    current_instrument = config.current_instrument
    current_year = config.current_year
    
    # Generate filenames
    available_instruments = config.list_available_instruments()
    inst_suffix = ""
    if len(available_instruments) > 1:
        inst_parts = current_instrument.split('_')
        if len(inst_parts) >= 4:
            inst_suffix = f"_{inst_parts[-1]}"
    
    dataset_name = f"{current_station}{inst_suffix}_dataset_{current_year}_splits_{int(test_size*100)}_{int(val_size*100)}.csv"
    if clean_only:
        filtered_name = f"{current_station}{inst_suffix}_dataset_{current_year}_splits_{int(test_size*100)}_{int(val_size*100)}_clean_filtered.csv"
    else:
        filtered_name = dataset_name
    
    model_dir = output_base / f"models_{model_type}"
    
    click.echo(f"✓ Station: {current_station}")
    click.echo(f"✓ Instrument: {current_instrument}")
    click.echo(f"✓ Training year: {current_year}")
    click.echo(f"✓ Prediction years: {', '.join(prediction_years)}")
    click.echo(f"✓ Dataset: {dataset_name}")
    click.echo(f"✓ Model directory: {model_dir}")
    click.echo(f"✓ Output directory: {output_base}")
    
    if dry_run:
        click.echo(f"\n🔍 DRY RUN - No actions will be performed")
        click.echo(f"\nPipeline would execute:")
        click.echo(f"1. Create dataset: {dataset_name}")
        if clean_only:
            click.echo(f"2. Filter to clean data: {filtered_name}")
        click.echo(f"3. Train {model_type} model with preset {preset}")
        click.echo(f"4. Evaluate model on test set")
        for pred_year in prediction_years:
            click.echo(f"5. Generate predictions for {pred_year}")
        click.echo(f"6. Export results")
        return 0
    
    # Create CliRunner at the beginning for use throughout
    from click.testing import CliRunner
    import os
    
    # Preserve environment for CliRunner
    env = os.environ.copy()
    env['PHENOCAI_CURRENT_STATION'] = config.current_station
    env['PHENOCAI_CURRENT_INSTRUMENT'] = config.current_instrument
    env['PHENOCAI_CURRENT_YEAR'] = str(config.current_year)
    
    runner = CliRunner(env=env)
    
    # Step 1: Dataset Creation
    click.echo(f"\n📊 Step 1: Dataset Creation")
    try:
        dataset_path = config.experimental_data_dir / dataset_name
        if dataset_path.exists():
            click.echo(f"✓ Dataset already exists: {dataset_path}")
        else:
            click.echo(f"Creating dataset with {test_size:.0%} test, {val_size:.0%} val...")
            
            # Call dataset create command with explicit output path
            cmd_args = [
                '--test-size', str(test_size),
                '--val-size', str(val_size),
                '--output', str(dataset_path)
            ]
            if instrument:
                cmd_args.extend(['--instrument', instrument])
            
            result = runner.invoke(dataset_create, cmd_args)
            if result.exit_code != 0:
                click.echo(f"❌ Dataset creation failed: {result.output}", err=True)
                return 1
            
            click.echo(f"✓ Dataset created: {dataset_path}")
        
        # Step 1b: Filter dataset if requested
        if clean_only:
            filtered_path = config.experimental_data_dir / filtered_name
            if filtered_path.exists():
                click.echo(f"✓ Clean dataset already exists: {filtered_path}")
                dataset_path = filtered_path  # Use filtered dataset
            else:
                click.echo(f"Filtering to clean data only...")
                from .dataset import filter as dataset_filter
                
                result = runner.invoke(dataset_filter, [str(dataset_path), '--no-flags'])
                if result.exit_code != 0:
                    click.echo(f"❌ Dataset filtering failed: {result.output}", err=True)
                    return 1
                
                click.echo(f"✓ Clean dataset created: {filtered_path}")
                dataset_path = filtered_path  # Use filtered dataset
    
    except Exception as e:
        click.echo(f"❌ Error in dataset creation: {e}", err=True)
        return 1
    
    # Step 2: Model Training
    click.echo(f"\n🎓 Step 2: Model Training")
    try:
        model_path = model_dir / "final_model.h5"
        if model_path.exists():
            click.echo(f"✓ Model already exists: {model_path}")
        else:
            click.echo(f"Training {model_type} model with preset {preset}...")
            
            # Call train command
            cmd_args = [
                str(dataset_path),  # Use the actual dataset path
                '--preset', preset,
                '--output-dir', str(model_dir)
            ]
            
            result = runner.invoke(train_model, cmd_args)
            if result.exit_code != 0:
                click.echo(f"❌ Model training failed: {result.output}", err=True)
                return 1
            
            click.echo(f"✓ Model trained: {model_path}")
    
    except Exception as e:
        click.echo(f"❌ Error in model training: {e}", err=True)
        return 1
    
    # Step 3: Model Evaluation
    click.echo(f"\n📈 Step 3: Model Evaluation")
    try:
        eval_output = output_base / "evaluation_results.txt"
        click.echo(f"Evaluating model performance...")
        
        # Call evaluate command
        cmd_args = [
            str(model_path),
            str(dataset_path),  # Use the actual dataset path
            '--save-predictions',
            '--generate-plots'
        ]
        
        result = runner.invoke(evaluate_model, cmd_args)
        if result.exit_code != 0:
            click.echo(f"❌ Model evaluation failed: {result.output}", err=True)
            return 1
        
        click.echo(f"✓ Model evaluated, results saved")
    
    except Exception as e:
        click.echo(f"❌ Error in model evaluation: {e}", err=True)
        return 1
    
    # Step 4: Generate Predictions
    click.echo(f"\n🔮 Step 4: Generate Predictions")
    try:
        for pred_year in prediction_years:
            click.echo(f"Generating predictions for {pred_year}...")
            
            pred_output_dir = output_base / f"predictions_{pred_year}"
            
            # Call predict batch command
            cmd_args = [
                str(model_path),
                '--year', pred_year,
                '--output-dir', str(pred_output_dir),
                '--format', 'yaml',
                '--use-heuristics'
            ]
            
            result = runner.invoke(predict_batch, cmd_args)
            if result.exit_code != 0:
                click.echo(f"❌ Prediction failed for {pred_year}: {result.output}", err=True)
                continue
            
            click.echo(f"✓ Predictions generated for {pred_year}: {pred_output_dir}")
    
    except Exception as e:
        click.echo(f"❌ Error in prediction generation: {e}", err=True)
        return 1
    
    # Step 5: Export Summary
    click.echo(f"\n📋 Step 5: Generate Summary")
    try:
        summary_file = output_base / "pipeline_summary.txt"
        
        end_time = time.time()
        duration = end_time - start_time
        
        with open(summary_file, 'w') as f:
            f.write("PhenoCAI Pipeline Execution Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Station: {current_station}\n")
            f.write(f"  Instrument: {current_instrument}\n")
            f.write(f"  Training Year: {current_year}\n")
            f.write(f"  Prediction Years: {', '.join(prediction_years)}\n")
            f.write(f"  Model Type: {model_type}\n")
            f.write(f"  Training Preset: {preset}\n")
            f.write(f"  Clean Data Only: {clean_only}\n\n")
            f.write(f"Outputs:\n")
            f.write(f"  Dataset: {dataset_name}\n")
            if clean_only:
                f.write(f"  Filtered Dataset: {filtered_name}\n")
            f.write(f"  Model: {model_dir}/final_model.h5\n")
            f.write(f"  Evaluation: evaluation_results/\n")
            for pred_year in prediction_years:
                f.write(f"  Predictions {pred_year}: predictions_{pred_year}/\n")
        
        click.echo(f"✓ Summary saved: {summary_file}")
    
    except Exception as e:
        click.echo(f"⚠️ Warning: Could not generate summary: {e}")
    
    # Final Summary
    end_time = time.time()
    duration = end_time - start_time
    
    click.echo(f"\n🎉 Pipeline Completed Successfully!")
    click.echo(f"{'='*50}")
    click.echo(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    click.echo(f"Results saved to: {output_base}")
    click.echo(f"\nNext steps:")
    click.echo(f"  • Review evaluation results")
    click.echo(f"  • Check prediction outputs")
    click.echo(f"  • Use predictions for research analysis")
    
    return 0


@pipeline.command()
@click.option('--station', '-s', help='Station to check')
@click.option('--years', '-y', multiple=True, help='Years to check for data availability')
def status(station, years):
    """Check status of data, models, and predictions for a station."""
    if station:
        try:
            config.switch_station(station)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            return 1
    
    current_station = config.current_station
    current_instrument = config.current_instrument
    
    click.echo(f"\n📊 Pipeline Status for {current_station}")
    click.echo(f"{'='*50}")
    
    # Check datasets
    click.echo(f"\n📁 Datasets:")
    dataset_pattern = f"{current_station}_*_dataset_*.csv"
    datasets = list(config.experimental_data_dir.glob(dataset_pattern))
    if datasets:
        for dataset in datasets:
            click.echo(f"  ✓ {dataset.name}")
    else:
        click.echo(f"  ❌ No datasets found (pattern: {dataset_pattern})")
    
    # Check models
    click.echo(f"\n🤖 Models:")
    model_dirs = list(config.experimental_data_dir.glob("models_*"))
    if model_dirs:
        for model_dir in model_dirs:
            model_file = model_dir / "final_model.h5"
            status = "✓" if model_file.exists() else "⚠️"
            click.echo(f"  {status} {model_dir.name}")
    else:
        click.echo(f"  ❌ No models found")
    
    # Check predictions
    click.echo(f"\n🔮 Predictions:")
    if years:
        for year in years:
            pred_dirs = list(config.experimental_data_dir.glob(f"*predictions_{year}*"))
            if pred_dirs:
                for pred_dir in pred_dirs:
                    click.echo(f"  ✓ {pred_dir.name}")
            else:
                click.echo(f"  ❌ No predictions for {year}")
    else:
        pred_dirs = list(config.experimental_data_dir.glob("*predictions_*"))
        if pred_dirs:
            for pred_dir in pred_dirs:
                click.echo(f"  ✓ {pred_dir.name}")
        else:
            click.echo(f"  ❌ No predictions found")
    
    # System status
    click.echo(f"\n⚙️ System Status:")
    click.echo(f"  Station: {current_station}")
    click.echo(f"  Instrument: {current_instrument}")
    click.echo(f"  Year: {config.current_year}")
    click.echo(f"  Available instruments: {', '.join(config.list_available_instruments())}")