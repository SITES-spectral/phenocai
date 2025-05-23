#!/usr/bin/env python3
"""
Script 01: Prepare Training Data for Lönnstorp

This script prepares the training data by:
1. Creating the dataset with train/test/val splits
2. Applying heuristics to get initial labels
3. Saving the prepared dataset
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.phenocai.config.setup import config
from src.phenocai.data_management import create_master_annotation_dataframe, add_train_test_split
from src.phenocai.utils import load_image, parse_image_filename, load_roi_config_from_yaml, get_roi_points_from_config, extract_roi_sub_image
from src.phenocai.heuristics import detect_snow_hsv, should_discard_roi


@click.command()
@click.option('--station', default='lonnstorp', help='Station name')
@click.option('--output-dir', default=None, help='Output directory for prepared data')
@click.option('--apply-heuristics', is_flag=True, help='Apply heuristics to generate initial predictions')
@click.option('--sample-size', type=int, default=None, help='Sample size for testing (None for full dataset)')
def main(station, output_dir, apply_heuristics, sample_size):
    """Prepare training data for the specified station."""
    
    click.echo(f"\n=== Preparing Training Data for {station.upper()} ===\n")
    
    # Set output directory
    if output_dir is None:
        output_dir = config.experimental_data_dir
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create or load dataset
    dataset_path = output_dir / f'{station}_dataset_with_splits.csv'
    
    if dataset_path.exists():
        click.echo(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        click.echo("Creating new dataset...")
        
        # Create dataset
        df, stats = create_master_annotation_dataframe(
            config.master_annotation_pool_dir,
            include_unannotated=False
        )
        
        click.echo(f"Created dataset with {len(df)} records")
        stats.print_summary()
        
        # Add splits
        click.echo("\nAdding train/test/val splits...")
        df = add_train_test_split(df, test_size=0.2, val_size=0.1, stratify_by=None)
        
        # Save dataset
        df.to_csv(dataset_path, index=False)
        click.echo(f"Saved dataset to {dataset_path}")
    
    # Show split distribution
    click.echo("\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        click.echo(f"\nSampling {sample_size} records for testing...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Step 2: Apply heuristics if requested
    if apply_heuristics:
        click.echo("\n=== Applying Heuristics ===\n")
        
        # Load ROI configuration
        roi_config = load_roi_config_from_yaml(config.roi_config_file_path)
        
        # Add heuristic prediction columns
        df['heuristic_snow'] = False
        df['heuristic_discard'] = False
        df['heuristic_confidence'] = 0.0
        
        # Process images
        failed_images = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            try:
                # Construct image path
                image_path = Path(config.image_base_dir) / str(row['day_of_year']).zfill(3) / row['image_filename']
                
                if not image_path.exists():
                    failed_images.append(str(image_path))
                    continue
                
                # Load image
                image = load_image(image_path)
                
                # Get ROI points
                roi_points = get_roi_points_from_config(
                    roi_config,
                    row['station'],
                    row['instrument'],
                    row['roi_name']
                )
                
                # Extract ROI
                roi_image = extract_roi_sub_image(image, roi_points, row['roi_name'])
                
                # Apply heuristics
                # Snow detection
                has_snow, snow_percentage = detect_snow_hsv(roi_image)
                df.at[idx, 'heuristic_snow'] = has_snow
                df.at[idx, 'heuristic_confidence'] = snow_percentage
                
                # Quality check
                should_discard, quality_metrics = should_discard_roi(roi_image)
                df.at[idx, 'heuristic_discard'] = should_discard
                
            except Exception as e:
                failed_images.append(f"{row['image_filename']} - {str(e)}")
                continue
        
        if failed_images:
            click.echo(f"\nFailed to process {len(failed_images)} images")
            if len(failed_images) < 10:
                for img in failed_images:
                    click.echo(f"  - {img}")
        
        # Save with heuristics
        heuristic_path = output_dir / f'{station}_dataset_with_heuristics.csv'
        df.to_csv(heuristic_path, index=False)
        click.echo(f"\nSaved dataset with heuristics to {heuristic_path}")
        
        # Compare heuristics with annotations
        click.echo("\n=== Heuristic Performance ===")
        
        snow_accuracy = (df['snow_presence'] == df['heuristic_snow']).mean()
        click.echo(f"\nSnow detection accuracy: {snow_accuracy:.1%}")
        
        # Confusion matrix for snow
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df['snow_presence'], df['heuristic_snow'])
        click.echo("\nSnow detection confusion matrix:")
        click.echo("              Predicted")
        click.echo("             No Snow  Snow")
        click.echo(f"Actual No   {cm[0,0]:7d} {cm[0,1]:5d}")
        click.echo(f"       Snow {cm[1,0]:7d} {cm[1,1]:5d}")
    
    # Step 3: Create summary
    summary_path = output_dir / f'{station}_dataset_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write(f"Dataset Summary for {station.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Total images: {df['image_id'].nunique()}\n")
        f.write(f"Date range: {df['day_of_year'].min()} - {df['day_of_year'].max()}\n\n")
        
        f.write("Split distribution:\n")
        f.write(str(df['split'].value_counts()) + "\n\n")
        
        f.write("Label distribution:\n")
        f.write(f"Snow present: {df['snow_presence'].sum()} ({df['snow_presence'].mean():.1%})\n")
        f.write(f"Discarded: {df['discard'].sum()} ({df['discard'].mean():.1%})\n\n")
        
        f.write("ROI distribution:\n")
        f.write(str(df['roi_name'].value_counts()) + "\n")
    
    click.echo(f"\nSummary saved to {summary_path}")
    click.echo("\n✓ Training data preparation complete!")
    click.echo(f"\nNext steps:")
    click.echo(f"1. Review the dataset: {dataset_path}")
    click.echo(f"2. Run training script: python scripts/02_train_models.py")


if __name__ == '__main__':
    main()