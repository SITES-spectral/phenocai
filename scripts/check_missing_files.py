#!/usr/bin/env python3
"""
Check for missing files in the dataset and create a cleaned version.
"""

import pandas as pd
import os
from pathlib import Path
import sys


def check_missing_files(csv_path):
    """Check which files in the dataset are missing."""
    
    # Read the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Check file existence
    df['file_exists'] = df['file_path'].apply(os.path.exists)
    
    # Count missing files
    missing_df = df[~df['file_exists']]
    print(f"\nFound {len(missing_df)} missing files out of {len(df)} total ({len(missing_df)/len(df)*100:.1f}%)")
    
    # Analyze missing files by station
    print("\nMissing files by station:")
    for station in df['station'].unique():
        station_df = df[df['station'] == station]
        station_missing = station_df[~station_df['file_exists']]
        print(f"  {station}: {len(station_missing)} missing out of {len(station_df)} ({len(station_missing)/len(station_df)*100:.1f}%)")
    
    # Analyze missing files by instrument
    print("\nMissing files by instrument:")
    for instrument in df['instrument'].unique():
        inst_df = df[df['instrument'] == instrument]
        inst_missing = inst_df[~inst_df['file_exists']]
        print(f"  {instrument}: {len(inst_missing)} missing out of {len(inst_df)} ({len(inst_missing)/len(inst_df)*100:.1f}%)")
    
    # Show sample of missing files
    if len(missing_df) > 0:
        print("\nSample of missing files:")
        for idx, row in missing_df.head(10).iterrows():
            print(f"  {row['file_path']}")
    
    # Create cleaned dataset
    clean_df = df[df['file_exists']].drop(columns=['file_exists'])
    
    # Save cleaned dataset
    output_path = csv_path.replace('.csv', '_cleaned.csv')
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset with {len(clean_df)} valid entries to:")
    print(f"  {output_path}")
    
    # Check splits distribution after cleaning
    print("\nData splits after cleaning:")
    for split in clean_df['split'].unique():
        split_df = clean_df[clean_df['split'] == split]
        snow_pct = (split_df['snow_presence'].sum() / len(split_df)) * 100
        print(f"  {split}: {len(split_df)} samples ({snow_pct:.1f}% with snow)")
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_missing_files.py <csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    cleaned_path = check_missing_files(csv_path)