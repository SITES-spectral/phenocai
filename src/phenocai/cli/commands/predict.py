"""
Prediction/inference commands for applying trained models to new images.
"""
import click
from pathlib import Path
import json
from typing import Optional

from ...inference import (
    ModelPredictor,
    BatchPredictor,
    process_single_image,
    process_image_directory,
    process_date_range
)
from ...config.setup import config as app_config


@click.group()
def predict():
    """Apply models to new images."""
    pass


@predict.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--images-dir', '-i', type=click.Path(exists=True), help='Directory containing images')
@click.option('--image-path', type=click.Path(exists=True), help='Single image to process')
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Output directory for annotations')
@click.option('--batch-size', type=int, default=32, help='Batch size for processing')
@click.option('--threshold', type=float, default=0.5, help='Prediction threshold')
@click.option('--format', type=click.Choice(['yaml', 'csv', 'json']), default='yaml', help='Output format')
@click.option('--use-heuristics', is_flag=True, default=True, help='Use heuristics as fallback')
@click.option('--recursive', is_flag=True, help='Search directories recursively')
def apply(model_path, images_dir, image_path, output_dir, batch_size, threshold, format, use_heuristics, recursive):
    """Apply model to new images.
    
    Examples:
        # Process single image
        phenocai predict apply model.h5 --image-path image.jpg -o results/
        
        # Process directory
        phenocai predict apply model.h5 --images-dir /path/to/images/ -o results/
        
        # Process with custom threshold
        phenocai predict apply model.h5 -i images/ -o results/ --threshold 0.7
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if images_dir and image_path:
        click.echo("Error: Specify either --images-dir or --image-path, not both", err=True)
        return 1
    
    if not images_dir and not image_path:
        click.echo("Error: Must specify either --images-dir or --image-path", err=True)
        return 1
    
    click.echo(f"\n=== Applying Model ===")
    click.echo(f"Model: {model_path}")
    click.echo(f"Threshold: {threshold}")
    click.echo(f"Output format: {format}")
    click.echo(f"Use heuristics: {use_heuristics}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if image_path:
            # Single image processing
            image_path = Path(image_path)
            click.echo(f"\nProcessing: {image_path}")
            
            predictions = process_single_image(
                model_path=model_path,
                image_path=image_path,
                threshold=threshold,
                use_heuristics=use_heuristics
            )
            
            # Save results
            if format == 'yaml':
                import yaml
                output_file = output_dir / f"{image_path.stem}_predictions.yaml"
                with open(output_file, 'w') as f:
                    yaml.dump({
                        'filename': image_path.name,
                        'model_path': str(model_path),
                        'threshold': threshold,
                        'predictions': [pred.to_dict() for pred in predictions]
                    }, f)
            elif format == 'json':
                output_file = output_dir / f"{image_path.stem}_predictions.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'filename': image_path.name,
                        'model_path': str(model_path),
                        'threshold': threshold,
                        'predictions': [pred.to_dict() for pred in predictions]
                    }, f, indent=2)
            else:  # csv
                import pandas as pd
                rows = [pred.to_dict() for pred in predictions]
                df = pd.DataFrame(rows)
                output_file = output_dir / f"{image_path.stem}_predictions.csv"
                df.to_csv(output_file, index=False)
            
            click.echo(f"\n✓ Processed 1 image with {len(predictions)} ROIs")
            snow_count = sum(1 for p in predictions if p.snow_presence)
            click.echo(f"Snow detected in {snow_count}/{len(predictions)} ROIs")
            click.echo(f"Results saved to: {output_file}")
            
        else:
            # Directory processing
            images_dir = Path(images_dir)
            click.echo(f"\nProcessing directory: {images_dir}")
            click.echo(f"Recursive: {recursive}")
            
            stats = process_image_directory(
                model_path=model_path,
                directory=images_dir,
                output_dir=output_dir,
                threshold=threshold,
                use_heuristics=use_heuristics,
                output_format=format
            )
            
            click.echo(f"\n✓ Processing complete!")
            click.echo(f"Total images: {stats['total_images']}")
            click.echo(f"Processed: {stats['processed_images']}")
            click.echo(f"Total predictions: {stats['total_predictions']}")
            click.echo(f"Snow predictions: {stats['snow_predictions']} ({stats['snow_percentage']:.1f}%)")
            click.echo(f"\nResults saved to: {output_dir}")
            
    except Exception as e:
        click.echo(f"\n❌ Prediction failed: {str(e)}", err=True)
        raise click.Abort()


@predict.command()
@click.option('--start-day', type=int, required=True, help='Start day of year')
@click.option('--end-day', type=int, required=True, help='End day of year')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Model to use')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--station', help='Station (uses current if not specified)')
@click.option('--year', type=int, help='Year (uses current if not specified)')
@click.option('--threshold', type=float, default=0.5, help='Prediction threshold')
@click.option('--use-heuristics', is_flag=True, default=True, help='Use heuristics as fallback')
def batch(start_day, end_day, model_path, output_dir, station, year, threshold, use_heuristics):
    """Process multiple days of images.
    
    Examples:
        # Process days 100-110 for current year/station
        phenocai predict batch --start-day 100 --end-day 110 --model-path model.h5 -o results/
        
        # Process specific year
        phenocai predict batch --start-day 1 --end-day 365 --model-path model.h5 --year 2023 -o results/
    """
    model_path = Path(model_path)
    
    # Use current config if not specified
    if not station:
        station = app_config.current_station
    if not year:
        year = int(app_config.current_year)
    
    if not output_dir:
        output_dir = app_config.output_dir_for_new_annotations / f"batch_{year}_{start_day}-{end_day}"
    output_dir = Path(output_dir)
    
    click.echo(f"\n=== Batch Processing ===")
    click.echo(f"Station: {station}")
    click.echo(f"Year: {year}")
    click.echo(f"Days: {start_day} to {end_day}")
    click.echo(f"Model: {model_path}")
    click.echo(f"Output: {output_dir}")
    
    try:
        # Process date range
        stats = process_date_range(
            model_path=model_path,
            start_day=start_day,
            end_day=end_day,
            output_dir=output_dir,
            year=year,
            station=station,
            threshold=threshold,
            use_heuristics=use_heuristics
        )
        
        click.echo(f"\n✓ Batch processing complete!")
        click.echo(f"Days processed: {stats['total_days']}")
        click.echo(f"Total images: {stats['total_images']}")
        click.echo(f"Total predictions: {stats['total_predictions']}")
        click.echo(f"Snow predictions: {stats['snow_predictions']} ({stats['snow_predictions']/stats['total_predictions']*100:.1f}%)")
        
        # Show daily summary
        if stats['daily_stats']:
            click.echo("\nDaily summary:")
            for day, day_stats in sorted(stats['daily_stats'].items())[:5]:  # Show first 5 days
                click.echo(f"  Day {day}: {day_stats['images']} images, {day_stats['snow_percentage']:.1f}% snow")
            if len(stats['daily_stats']) > 5:
                click.echo(f"  ... and {len(stats['daily_stats']) - 5} more days")
        
        # Save summary report
        summary_file = output_dir / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        click.echo(f"\nResults saved to: {output_dir}")
        click.echo(f"Summary report: {summary_file}")
        
    except Exception as e:
        click.echo(f"\n❌ Batch processing failed: {str(e)}", err=True)
        raise click.Abort()


@predict.command()
@click.argument('annotations_dir', type=click.Path(exists=True))
@click.option('--output-path', '-o', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['csv', 'parquet', 'json']), default='csv', help='Output format')
@click.option('--include-proba', is_flag=True, help='Include probability scores')
def export(annotations_dir, output_path, format, include_proba):
    """Export predictions to different formats.
    
    Combines prediction YAML files into a single dataset file.
    
    Examples:
        # Export to CSV
        phenocai predict export results/ -o predictions.csv
        
        # Export to Parquet with probabilities
        phenocai predict export results/ -o predictions.parquet --format parquet --include-proba
    """
    import yaml
    import pandas as pd
    
    annotations_dir = Path(annotations_dir)
    
    click.echo(f"\n=== Exporting Predictions ===")
    click.echo(f"Source: {annotations_dir}")
    click.echo(f"Format: {format}")
    
    if not output_path:
        output_path = annotations_dir.parent / f"predictions_export.{format}"
    output_path = Path(output_path)
    
    try:
        # Find all prediction files
        yaml_files = list(annotations_dir.rglob("*_predictions.yaml"))
        
        if not yaml_files:
            click.echo("No prediction files found!", err=True)
            return 1
        
        click.echo(f"Found {len(yaml_files)} prediction files")
        
        # Load all predictions
        all_records = []
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Handle both single-image and batch formats
            if 'annotations' in data:
                # Batch format
                for ann in data['annotations']:
                    filename = data.get('filename', yaml_file.stem.replace('_predictions', ''))
                    # Construct file path based on station/year/day
                    station = data.get('station', '')
                    instrument = data.get('instrument', '')
                    year = data.get('year', '')
                    day_of_year = data.get('day_of_year', '')
                    if station and instrument and year and day_of_year:
                        # Format day_of_year as 3-digit string
                        day_dir = f"{int(day_of_year):03d}" if day_of_year else "001"
                        base_path = Path("/home/jobelund/lu2024-12-46/SITES/Spectral/data")
                        file_path = str(base_path / station / "phenocams" / "products" / instrument / "L1" / str(year) / day_dir / filename)
                    else:
                        file_path = filename  # Fallback to just filename
                    
                    record = {
                        'filename': filename,
                        'file_path': file_path,
                        'station': station,
                        'instrument': instrument,
                        'year': year,
                        'day_of_year': data.get('day_of_year', ''),
                        'roi_name': ann['roi_name'],
                        'discard': ann.get('discard', False),
                        'snow_presence': ann['snow_presence'],
                        'quality_flags': ','.join(ann.get('flags', [])),
                        'not_needed': ann.get('not_needed', False),
                        'model_predicted': ann.get('model_predicted', True)
                    }
                    if include_proba and 'snow_probability' in ann:
                        record['snow_probability'] = ann['snow_probability']
                        record['confidence'] = ann.get('confidence', 0.0)
                    all_records.append(record)
            elif 'predictions' in data:
                # Single image format
                for pred in data['predictions']:
                    record = {
                        'filename': data['filename'],
                        'roi_name': pred['roi_name'],
                        'discard': pred.get('discard', False),
                        'snow_presence': pred['snow_presence'],
                        'quality_flags': ','.join(pred.get('quality_flags', [])),
                        'model_predicted': True
                    }
                    if include_proba:
                        record['snow_probability'] = pred['snow_probability']
                        record['confidence'] = pred.get('confidence', 0.0)
                    all_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        
        # Add derived columns
        df['has_flags'] = df['quality_flags'].str.len() > 0
        df['flag_count'] = df['quality_flags'].str.split(',').apply(lambda x: len(x) if x[0] else 0)
        
        # Sort
        df = df.sort_values(['filename', 'roi_name'])
        
        # Save in requested format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        
        click.echo(f"\n✓ Export complete!")
        click.echo(f"Total records: {len(df)}")
        click.echo(f"Unique images: {df['filename'].nunique()}")
        click.echo(f"Snow predictions: {df['snow_presence'].sum()} ({df['snow_presence'].mean()*100:.1f}%)")
        
        if 'discard' in df.columns:
            click.echo(f"Discarded ROIs: {df['discard'].sum()} ({df['discard'].mean()*100:.1f}%)")
        
        if 'has_flags' in df.columns:
            click.echo(f"ROIs with quality issues: {df['has_flags'].sum()} ({df['has_flags'].mean()*100:.1f}%)")
        
        if include_proba and 'snow_probability' in df.columns:
            click.echo(f"Average confidence: {df['confidence'].mean():.3f}")
        
        click.echo(f"\nExported to: {output_path}")
        
    except Exception as e:
        click.echo(f"\n❌ Export failed: {str(e)}", err=True)
        raise click.Abort()