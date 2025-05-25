#!/usr/bin/env python3
"""
Fix incorrect paths in the multi-station dataset CSV file.
The Röbäcksdalen images are incorrectly pointing to Lönnstorp directories.
"""

import pandas as pd
import sys
from pathlib import Path


def fix_dataset_paths(csv_path, output_path=None):
    """Fix incorrect paths in dataset CSV."""
    
    # Read the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Count issues before fixing
    robacksdalen_rows = df[df['station'] == 'robacksdalen']
    incorrect_paths = robacksdalen_rows[robacksdalen_rows['file_path'].str.contains('lonnstorp')]
    print(f"Found {len(incorrect_paths)} Röbäcksdalen rows with incorrect paths")
    
    # Create a function to fix the path
    def fix_path(row):
        if row['station'] == 'robacksdalen' and 'lonnstorp' in row['file_path']:
            # Replace the incorrect parts
            fixed_path = row['file_path'].replace(
                '/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/',
                '/data/robacksdalen/phenocams/products/RBD_AGR_PL02_PHE01/'
            )
            return fixed_path
        return row['file_path']
    
    # Apply the fix
    df['file_path'] = df.apply(fix_path, axis=1)
    
    # Verify the fix
    robacksdalen_rows_after = df[df['station'] == 'robacksdalen']
    incorrect_paths_after = robacksdalen_rows_after[robacksdalen_rows_after['file_path'].str.contains('lonnstorp')]
    print(f"After fix: {len(incorrect_paths_after)} Röbäcksdalen rows with incorrect paths")
    
    # Save the fixed dataset
    if output_path is None:
        output_path = csv_path.replace('.csv', '_fixed.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Saved fixed dataset to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset_paths.py <csv_path> [output_path]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    fixed_path = fix_dataset_paths(csv_path, output_path)
    print(f"\nFixed dataset saved to: {fixed_path}")