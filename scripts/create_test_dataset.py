#!/usr/bin/env python
"""
Create a test dataset with known ground truth for evaluation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.utils import parse_image_filename


def create_test_dataset():
    """Create a test dataset with synthetic but realistic ground truth."""
    print("Creating test dataset with ground truth...")
    
    # Base directories
    base_dirs = {
        'lonnstorp': Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/L1'),
        'robacksdalen': Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/robacksdalen/phenocams/products/RBD_AGR_PL02_PHE01/L1')
    }
    
    data = []
    
    for station, base_dir in base_dirs.items():
        print(f"\nProcessing {station}...")
        
        # Sample from 2024 data
        year_dir = base_dir / '2024'
        if not year_dir.exists():
            continue
        
        # Get sample of images across different days
        day_dirs = sorted(year_dir.glob('[0-9]*'))
        
        # Sample days throughout the year
        sampled_days = day_dirs[::10][:50]  # Every 10th day, max 50 days
        
        for day_dir in sampled_days:
            # Get a few images from each day
            images = list(day_dir.glob('*.jpg'))[:5]
            
            for img_path in images:
                try:
                    info = parse_image_filename(img_path.name)
                    
                    # Create realistic ground truth based on:
                    # 1. Season (winter months more likely to have snow)
                    # 2. Station (northern more likely than southern)
                    # 3. Some randomness
                    
                    month = info.full_datetime.month
                    day_of_year = info.day_of_year
                    
                    # Base probability by month
                    if month in [12, 1, 2]:  # Winter
                        base_prob = 0.8
                    elif month in [3, 11]:  # Early spring, late fall
                        base_prob = 0.4
                    elif month in [4, 10]:  # Spring, fall
                        base_prob = 0.1
                    else:  # Summer
                        base_prob = 0.0
                    
                    # Adjust for station
                    if station == 'robacksdalen':
                        base_prob *= 1.2  # Northern station
                    
                    # Add some randomness
                    noise = np.random.normal(0, 0.1)
                    final_prob = np.clip(base_prob + noise, 0, 1)
                    
                    # Determine snow presence
                    snow_presence = np.random.random() < final_prob
                    
                    # Assign to split
                    rand = np.random.random()
                    if rand < 0.7:
                        split = 'train'
                    elif rand < 0.85:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    data.append({
                        'file_path': str(img_path),
                        'filename': img_path.name,
                        'station': station,
                        'instrument': info.instrument,
                        'year': info.year,
                        'day_of_year': day_of_year,
                        'month': month,
                        'snow_presence': snow_presence,
                        'snow_probability_true': final_prob,
                        'split': split
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Balance the dataset somewhat
    snow_df = df[df['snow_presence'] == True]
    no_snow_df = df[df['snow_presence'] == False]
    
    # Take all snow samples and 2x no-snow samples
    if len(snow_df) > 0:
        n_snow = len(snow_df)
        n_no_snow = min(len(no_snow_df), n_snow * 2)
        
        balanced_df = pd.concat([
            snow_df,
            no_snow_df.sample(n=n_no_snow, random_state=42)
        ])
        
        df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save dataset
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/test_dataset_with_truth.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset created: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Snow samples: {df['snow_presence'].sum()} ({df['snow_presence'].mean()*100:.1f}%)")
    print(f"Splits: Train={len(df[df['split']=='train'])}, Val={len(df[df['split']=='val'])}, Test={len(df[df['split']=='test'])}")
    
    return output_path


if __name__ == "__main__":
    create_test_dataset()