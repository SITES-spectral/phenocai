#!/usr/bin/env python
"""
Predict snow presence for historical images (2022-2023) from both stations.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import defaultdict
import json

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.utils import parse_image_filename
from phenocai.config.setup import config


def get_station_images(station, years, limit_per_month=None):
    """Get image paths for a station and years."""
    image_paths = []
    
    # Determine base directory based on station
    if station == 'lonnstorp':
        base_dir = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/L1')
        instrument = 'LON_AGR_PL01_PHE01'
    elif station == 'robacksdalen':
        base_dir = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/robacksdalen/phenocams/products/RBD_AGR_PL02_PHE01/L1')
        instrument = 'RBD_AGR_PL02_PHE01'
    else:
        raise ValueError(f"Unknown station: {station}")
    
    # Collect images by month
    images_by_month = defaultdict(list)
    
    for year in years:
        year_dir = base_dir / str(year)
        if not year_dir.exists():
            print(f"Warning: Directory not found: {year_dir}")
            continue
            
        # Get all jpg files (they're in day-of-year subdirectories)
        for img_path in year_dir.glob('*/*.jpg'):
            try:
                info = parse_image_filename(img_path.name)
                if info and info.instrument == instrument:
                    month = info.full_datetime.month
                    month_key = f"{year}-{month:02d}"
                    images_by_month[month_key].append(str(img_path))
            except:
                continue
    
    # Sample images if limit specified
    for month, paths in sorted(images_by_month.items()):
        if limit_per_month and len(paths) > limit_per_month:
            # Sample evenly throughout the month
            indices = np.linspace(0, len(paths)-1, limit_per_month, dtype=int)
            sampled_paths = [paths[i] for i in indices]
            image_paths.extend(sampled_paths)
        else:
            image_paths.extend(paths)
    
    return sorted(image_paths)


def predict_batch(model, image_paths, batch_size=32, threshold=0.3):
    """Predict snow presence for a batch of images."""
    predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Load and preprocess image
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [224, 224])
                img = tf.cast(img, tf.float32)
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                
                batch_images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if batch_images:
            # Stack and predict
            batch_tensor = tf.stack(batch_images)
            batch_preds = model.predict(batch_tensor, verbose=0)
            
            # Process predictions
            for path, pred_prob in zip(valid_paths, batch_preds.flatten()):
                pred_class = int(pred_prob >= threshold)
                predictions.append({
                    'file_path': path,
                    'filename': Path(path).name,
                    'snow_probability': float(pred_prob),
                    'snow_predicted': bool(pred_class),
                    'threshold_used': threshold
                })
    
    return predictions


def analyze_predictions(predictions, station):
    """Analyze prediction results."""
    df = pd.DataFrame(predictions)
    
    # Parse dates from filenames
    dates = []
    for _, row in df.iterrows():
        try:
            info = parse_image_filename(row['filename'])
            if info:
                dates.append(pd.Timestamp(info.full_datetime))
            else:
                dates.append(pd.NaT)
        except:
            dates.append(pd.NaT)
    
    df['date'] = dates
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Calculate statistics
    stats = {
        'station': station,
        'total_images': len(df),
        'snow_detected': df['snow_predicted'].sum(),
        'snow_percentage': df['snow_predicted'].mean() * 100,
        'avg_snow_probability': df['snow_probability'].mean(),
        'yearly_stats': {},
        'monthly_stats': {}
    }
    
    # Yearly statistics
    for year in sorted(df['year'].unique()):
        if pd.notna(year):
            year_df = df[df['year'] == year]
            stats['yearly_stats'][int(year)] = {
                'total_images': len(year_df),
                'snow_detected': year_df['snow_predicted'].sum(),
                'snow_percentage': year_df['snow_predicted'].mean() * 100,
                'snow_days': year_df.groupby('day_of_year')['snow_predicted'].any().sum()
            }
    
    # Monthly statistics
    for (year, month), group in df.groupby(['year', 'month']):
        if pd.notna(year) and pd.notna(month):
            month_key = f"{int(year)}-{int(month):02d}"
            stats['monthly_stats'][month_key] = {
                'total_images': len(group),
                'snow_detected': group['snow_predicted'].sum(),
                'snow_percentage': group['snow_predicted'].mean() * 100,
                'snow_days': group.groupby('day_of_year')['snow_predicted'].any().sum()
            }
    
    return stats, df


def main():
    """Main prediction workflow."""
    # Configuration
    model_path = '/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset/final_model.keras'
    years = [2022, 2023]
    stations = ['lonnstorp', 'robacksdalen']
    threshold = 0.6  # Higher threshold for balanced model (conservative predictions)
    limit_per_month = 20  # Limit images per month for faster processing
    
    # Load model
    print("Loading model...")
    try:
        # Try loading as Keras model first
        model = tf.keras.models.load_model(model_path)
    except:
        # If that fails, load as SavedModel
        import tensorflow.keras.layers as layers
        model = layers.TFSMLayer(model_path, call_endpoint='serving_default')
        # Wrap in a simple Sequential model for predict interface
        input_shape = (224, 224, 3)
        wrapper_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            model
        ])
        model = wrapper_model
    print("Model loaded successfully")
    
    # Results storage
    all_results = {}
    
    # Create output directory
    output_dir = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/predictions')
    output_dir.mkdir(exist_ok=True)
    
    for station in stations:
        print(f"\n{'='*60}")
        print(f"Processing {station.title()}")
        print('='*60)
        
        # Get image paths
        print(f"Collecting images for years: {years}")
        image_paths = get_station_images(station, years, limit_per_month)
        print(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            print(f"No images found for {station}")
            continue
        
        # Make predictions
        print(f"Making predictions (threshold={threshold})...")
        predictions = predict_batch(model, image_paths, threshold=threshold)
        print(f"Completed {len(predictions)} predictions")
        
        # Analyze results
        stats, df = analyze_predictions(predictions, station)
        
        # Print summary
        print(f"\n{station.title()} Summary:")
        print(f"Total images analyzed: {stats['total_images']}")
        print(f"Snow detected in: {stats['snow_detected']} images ({stats['snow_percentage']:.1f}%)")
        print(f"Average snow probability: {stats['avg_snow_probability']:.3f}")
        
        print(f"\nYearly breakdown:")
        for year, year_stats in sorted(stats['yearly_stats'].items()):
            print(f"  {year}: {year_stats['snow_detected']}/{year_stats['total_images']} images "
                  f"({year_stats['snow_percentage']:.1f}%), "
                  f"{year_stats['snow_days']} snow days")
        
        print(f"\nMonthly snow presence (%):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for year in sorted(df['year'].unique()):
            if pd.notna(year):
                print(f"  {int(year)}:", end='')
                for month_idx, month_name in enumerate(months, 1):
                    month_key = f"{int(year)}-{month_idx:02d}"
                    if month_key in stats['monthly_stats']:
                        pct = stats['monthly_stats'][month_key]['snow_percentage']
                        print(f" {month_name}:{pct:4.0f}%", end='')
                    else:
                        print(f" {month_name}:   -", end='')
                print()
        
        # Save results
        all_results[station] = {
            'stats': stats,
            'predictions': predictions[:100]  # Save sample of predictions
        }
        
        # Save detailed CSV
        csv_path = output_dir / f'{station}_snow_predictions_{"-".join(map(str, years))}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed predictions saved to: {csv_path}")
    
    # Save summary JSON
    summary_path = output_dir / f'snow_prediction_summary_{"-".join(map(str, years))}.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")
    
    # Create comparison visualization
    create_comparison_plot(all_results, output_dir)


def create_comparison_plot(results, output_dir):
    """Create a comparison plot of snow patterns between stations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        for idx, (station, data) in enumerate(results.items()):
            ax = axes[idx]
            
            # Extract monthly data
            monthly_data = []
            for month_key, month_stats in sorted(data['stats']['monthly_stats'].items()):
                year, month = month_key.split('-')
                monthly_data.append({
                    'date': pd.Timestamp(f"{year}-{month}-01"),
                    'snow_percentage': month_stats['snow_percentage']
                })
            
            if monthly_data:
                df = pd.DataFrame(monthly_data)
                df = df.sort_values('date')
                
                # Plot
                ax.plot(df['date'], df['snow_percentage'], 'o-', label=station.title(), markersize=8)
                ax.fill_between(df['date'], 0, df['snow_percentage'], alpha=0.3)
                
                ax.set_ylabel('Snow Presence (%)', fontsize=12)
                ax.set_title(f'{station.title()} - Monthly Snow Detection', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
                
                # Add yearly averages as horizontal lines
                for year in [2022, 2023]:
                    if year in data['stats']['yearly_stats']:
                        avg = data['stats']['yearly_stats'][year]['snow_percentage']
                        year_start = pd.Timestamp(f"{year}-01-01")
                        year_end = pd.Timestamp(f"{year}-12-31")
                        ax.hlines(avg, year_start, year_end, 
                                 colors='red', linestyles='dashed', alpha=0.5,
                                 label=f'{year} avg: {avg:.1f}%')
        
        axes[1].set_xlabel('Date', fontsize=12)
        
        # Add legend
        for ax in axes:
            ax.legend(loc='upper right')
        
        plt.suptitle('Snow Detection Comparison: Lönnstorp vs Röbäcksdalen (2022-2023)', fontsize=16)
        plt.tight_layout()
        
        plot_path = output_dir / 'snow_detection_comparison_2022-2023.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {plot_path}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


if __name__ == "__main__":
    main()