#!/usr/bin/env python
"""
Analyze predictions at different thresholds to understand model behavior.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_thresholds():
    """Analyze predictions across different thresholds."""
    # Load prediction CSVs
    pred_dir = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/predictions')
    
    stations = ['lonnstorp', 'robacksdalen']
    thresholds = np.arange(0.1, 1.0, 0.1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, station in enumerate(stations):
        csv_path = pred_dir / f'{station}_snow_predictions_2022-2023.csv'
        if not csv_path.exists():
            print(f"No predictions found for {station}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Analyze by threshold
        snow_pcts = []
        for threshold in thresholds:
            snow_detected = (df['snow_probability'] >= threshold).sum()
            snow_pct = (snow_detected / len(df)) * 100
            snow_pcts.append(snow_pct)
        
        # Plot 1: Snow detection percentage by threshold
        ax1 = axes[idx, 0]
        ax1.plot(thresholds, snow_pcts, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Images with Snow Detected (%)', fontsize=12)
        ax1.set_title(f'{station.title()} - Snow Detection by Threshold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 105)
        
        # Add vertical lines for key thresholds
        ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax1.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='Low (0.3)')
        ax1.axvline(0.7, color='orange', linestyle='--', alpha=0.5, label='High (0.7)')
        ax1.legend()
        
        # Plot 2: Probability distribution
        ax2 = axes[idx, 1]
        ax2.hist(df['snow_probability'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Snow Probability', fontsize=12)
        ax2.set_ylabel('Number of Images', fontsize=12)
        ax2.set_title(f'{station.title()} - Probability Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_prob = df['snow_probability'].mean()
        median_prob = df['snow_probability'].median()
        ax2.axvline(mean_prob, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_prob:.3f}')
        ax2.axvline(median_prob, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_prob:.3f}')
        ax2.legend()
        
        # Print monthly analysis at optimal threshold
        print(f"\n{station.title()} Analysis:")
        print(f"Total images: {len(df)}")
        print(f"Mean probability: {mean_prob:.3f}")
        print(f"Median probability: {median_prob:.3f}")
        
        # Find optimal threshold (where derivative is smallest)
        if len(snow_pcts) > 1:
            derivatives = np.diff(snow_pcts)
            optimal_idx = np.argmin(np.abs(derivatives)) + 1
            optimal_threshold = thresholds[optimal_idx]
            print(f"Suggested threshold (most stable): {optimal_threshold:.1f}")
        
        # Monthly analysis at different thresholds
        if 'month' in df.columns:
            print("\nMonthly snow detection (%) at different thresholds:")
            print("Month  ", end='')
            for t in [0.3, 0.5, 0.7]:
                print(f"  T={t}", end='')
            print()
            
            for month in range(1, 13):
                month_df = df[df['month'] == month]
                if len(month_df) > 0:
                    print(f"{month:2d}    ", end='')
                    for t in [0.3, 0.5, 0.7]:
                        pct = ((month_df['snow_probability'] >= t).sum() / len(month_df)) * 100
                        print(f"  {pct:3.0f}%", end='')
                    print()
    
    plt.suptitle('Snow Detection Model Analysis - 2022-2023', fontsize=16)
    plt.tight_layout()
    
    output_path = pred_dir / 'threshold_analysis_2022-2023.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {output_path}")
    
    # Create seasonal analysis
    create_seasonal_analysis(pred_dir)


def create_seasonal_analysis(pred_dir):
    """Create seasonal snow detection analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    stations = ['lonnstorp', 'robacksdalen']
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    for idx, station in enumerate(stations):
        csv_path = pred_dir / f'{station}_snow_predictions_2022-2023.csv'
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        
        # Calculate seasonal averages
        seasonal_probs = {}
        for season, months in seasons.items():
            season_df = df[df['month'].isin(months)]
            if len(season_df) > 0:
                seasonal_probs[season] = season_df['snow_probability'].mean()
            else:
                seasonal_probs[season] = 0
        
        ax = axes[idx]
        bars = ax.bar(seasonal_probs.keys(), seasonal_probs.values(), 
                      color=['lightblue', 'lightgreen', 'yellow', 'orange'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Average Snow Probability', fontsize=12)
        ax.set_title(f'{station.title()} - Seasonal Snow Probability', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold 0.5')
        ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Threshold 0.3')
        ax.legend()
    
    plt.suptitle('Seasonal Snow Probability Analysis - 2022-2023', fontsize=16)
    plt.tight_layout()
    
    output_path = pred_dir / 'seasonal_analysis_2022-2023.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSeasonal analysis saved to: {output_path}")


if __name__ == "__main__":
    analyze_thresholds()