#!/usr/bin/env python3
"""
Analyze Quality Issues in Dataset

This script analyzes the quality flags in the dataset to help understand
common issues and their distribution.
"""
import sys
from pathlib import Path
import pandas as pd
import click
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for analysis report')
@click.option('--show-examples', '-e', is_flag=True, help='Show example images for each flag')
@click.option('--min-count', type=int, default=5, help='Minimum count to include flag in analysis')
def main(dataset_path, output, show_examples, min_count):
    """Analyze quality issues in phenocam dataset."""
    
    click.echo(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Add has_flags column if not present
    if 'has_flags' not in df.columns:
        df['has_flags'] = df['flag_count'] > 0
    
    click.echo(f"Total records: {len(df)}")
    click.echo(f"Records with quality flags: {df['has_flags'].sum()} ({df['has_flags'].mean():.1%})")
    
    # Analyze flag distribution
    click.echo("\n=== Quality Flag Analysis ===\n")
    
    # Parse all flags
    all_flags = []
    for flags_str in df[df['has_flags']]['flags']:
        if pd.notna(flags_str) and flags_str:
            all_flags.extend(flags_str.split(','))
    
    # Count flags
    flag_counts = Counter(all_flags)
    
    # Display flag statistics
    click.echo("Flag frequency (sorted by count):")
    click.echo("-" * 50)
    
    total_flags = sum(flag_counts.values())
    for flag, count in flag_counts.most_common():
        if count >= min_count:
            percentage = count / total_flags * 100
            click.echo(f"{flag:30s} {count:6d} ({percentage:5.1f}%)")
    
    # Analyze co-occurrence
    click.echo("\n\nCommon flag combinations:")
    click.echo("-" * 50)
    
    # Get flag combinations
    flag_combinations = Counter()
    for flags_str in df[df['has_flags']]['flags']:
        if pd.notna(flags_str) and flags_str:
            flags = tuple(sorted(flags_str.split(',')))
            if len(flags) > 1:  # Only combinations
                flag_combinations[flags] += 1
    
    # Show top combinations
    for combo, count in flag_combinations.most_common(10):
        if count >= min_count:
            combo_str = " + ".join(combo)
            click.echo(f"{combo_str:50s} {count:6d}")
    
    # Analyze by ROI
    click.echo("\n\nQuality issues by ROI:")
    click.echo("-" * 50)
    
    roi_flag_stats = df.groupby('roi_name').agg({
        'has_flags': ['sum', 'mean'],
        'image_id': 'count'
    }).round(3)
    
    roi_flag_stats.columns = ['flags_count', 'flags_rate', 'total_count']
    roi_flag_stats['flags_rate_pct'] = (roi_flag_stats['flags_rate'] * 100).round(1)
    
    for roi in roi_flag_stats.index:
        stats = roi_flag_stats.loc[roi]
        click.echo(f"{roi}: {stats['flags_count']:4.0f}/{stats['total_count']:4.0f} ({stats['flags_rate_pct']:5.1f}%) have flags")
    
    # Analyze temporal patterns
    if 'day_of_year' in df.columns:
        click.echo("\n\nTemporal analysis:")
        click.echo("-" * 50)
        
        # Group by month (approximate)
        df['month'] = pd.cut(df['day_of_year'], 
                            bins=[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
                            labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        month_stats = df.groupby('month')['has_flags'].agg(['sum', 'mean', 'count'])
        month_stats['percentage'] = (month_stats['mean'] * 100).round(1)
        
        click.echo("Quality issues by month:")
        for month in month_stats.index:
            if month_stats.loc[month, 'count'] > 0:
                click.echo(f"{month}: {month_stats.loc[month, 'percentage']:5.1f}% have flags "
                          f"({month_stats.loc[month, 'sum']}/{month_stats.loc[month, 'count']})")
    
    # Show examples if requested
    if show_examples:
        click.echo("\n\nExample images for each flag:")
        click.echo("-" * 50)
        
        for flag in list(flag_counts.keys())[:10]:  # Top 10 flags
            # Find images with this flag
            mask = df['flags'].str.contains(flag, na=False)
            examples = df[mask].head(3)
            
            if len(examples) > 0:
                click.echo(f"\n{flag}:")
                for _, row in examples.iterrows():
                    click.echo(f"  - {row['image_filename']} (ROI: {row['roi_name']})")
    
    # Create report if output specified
    if output:
        with open(output, 'w') as f:
            f.write("PhenoCAI Quality Issues Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Records with quality flags: {df['has_flags'].sum()} ({df['has_flags'].mean():.1%})\n\n")
            
            f.write("Flag frequency:\n")
            for flag, count in flag_counts.most_common():
                if count >= min_count:
                    percentage = count / total_flags * 100
                    f.write(f"  {flag:30s} {count:6d} ({percentage:5.1f}%)\n")
        
        click.echo(f"\nReport saved to: {output}")
    
    # Summary recommendations
    click.echo("\n\n=== Recommendations ===")
    click.echo("-" * 50)
    
    # Find most problematic flags
    top_flags = [flag for flag, _ in flag_counts.most_common(5)]
    click.echo(f"Most common quality issues: {', '.join(top_flags)}")
    
    # Check if certain ROIs have more issues
    worst_roi = roi_flag_stats['flags_rate'].idxmax()
    click.echo(f"ROI with most quality issues: {worst_roi} ({roi_flag_stats.loc[worst_roi, 'flags_rate_pct']:.1f}%)")
    
    # Suggest filtering
    clean_count = len(df[~df['has_flags']])
    click.echo(f"\nClean samples available: {clean_count} ({clean_count/len(df):.1%})")
    
    if df['has_flags'].mean() > 0.5:
        click.echo("\n⚠️  Warning: More than 50% of samples have quality flags!")
        click.echo("Consider:")
        click.echo("  1. Training separate models for different weather conditions")
        click.echo("  2. Using data augmentation to handle quality variations")
        click.echo("  3. Filtering out the most problematic flags for initial training")


if __name__ == '__main__':
    main()