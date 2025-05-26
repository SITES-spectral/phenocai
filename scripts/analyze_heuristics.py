#!/usr/bin/env python
"""
Analyze and improve heuristic performance for snow detection.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.heuristics.snow_detection import detect_snow_hsv, detect_snow_with_refinement
from phenocai.heuristics.image_quality import detect_blur, detect_low_brightness, calculate_image_statistics
from phenocai.utils import load_image
from phenocai.config.setup import config


def analyze_snow_heuristics(dataset_csv, sample_size=500):
    """Analyze current snow detection heuristics performance."""
    print("=== Analyzing Snow Detection Heuristics ===\n")
    
    # Load dataset
    df = pd.read_csv(dataset_csv)
    
    # Sample data
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"Analyzing {len(df_sample)} images...")
    
    results = []
    hsv_distributions = {'snow': [], 'no_snow': []}
    
    for idx, row in df_sample.iterrows():
        try:
            # Load image
            image = load_image(row['file_path'])
            if image is None:
                continue
            
            # Apply current heuristics
            has_snow_hsv, snow_percentage = detect_snow_hsv(image)
            has_snow_refined, confidence, metadata = detect_snow_with_refinement(image)
            
            # Calculate additional metrics
            stats = calculate_image_statistics(image)
            
            # Convert to HSV and analyze
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Store results
            result = {
                'file_path': row['file_path'],
                'true_snow': row['snow_presence'],
                'hsv_detected': has_snow_hsv,
                'refined_detected': has_snow_refined,
                'snow_percentage': snow_percentage,
                'confidence': confidence,
                'brightness': stats['mean'],
                'contrast': stats['contrast'],
                'blur_metric': stats['blur_metric'],
                'edge_density': stats['edge_density']
            }
            results.append(result)
            
            # Collect HSV distributions
            if row['snow_presence']:
                hsv_distributions['snow'].append(hsv)
            else:
                hsv_distributions['no_snow'].append(hsv)
                
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
            continue
    
    # Analyze results
    results_df = pd.DataFrame(results)
    analyze_heuristic_performance(results_df)
    analyze_hsv_distributions(hsv_distributions)
    propose_improved_thresholds(results_df, hsv_distributions)
    
    return results_df


def analyze_heuristic_performance(results_df):
    """Analyze performance of current heuristics."""
    print("\n=== Heuristic Performance ===")
    
    # Calculate metrics for HSV method
    hsv_tp = ((results_df['true_snow'] == True) & (results_df['hsv_detected'] == True)).sum()
    hsv_fp = ((results_df['true_snow'] == False) & (results_df['hsv_detected'] == True)).sum()
    hsv_fn = ((results_df['true_snow'] == True) & (results_df['hsv_detected'] == False)).sum()
    hsv_tn = ((results_df['true_snow'] == False) & (results_df['hsv_detected'] == False)).sum()
    
    hsv_precision = hsv_tp / (hsv_tp + hsv_fp) if (hsv_tp + hsv_fp) > 0 else 0
    hsv_recall = hsv_tp / (hsv_tp + hsv_fn) if (hsv_tp + hsv_fn) > 0 else 0
    hsv_f1 = 2 * (hsv_precision * hsv_recall) / (hsv_precision + hsv_recall) if (hsv_precision + hsv_recall) > 0 else 0
    
    print(f"\nHSV Method Performance:")
    print(f"  Precision: {hsv_precision:.3f}")
    print(f"  Recall: {hsv_recall:.3f}")
    print(f"  F1 Score: {hsv_f1:.3f}")
    
    # Calculate metrics for refined method
    ref_tp = ((results_df['true_snow'] == True) & (results_df['refined_detected'] == True)).sum()
    ref_fp = ((results_df['true_snow'] == False) & (results_df['refined_detected'] == True)).sum()
    ref_fn = ((results_df['true_snow'] == True) & (results_df['refined_detected'] == False)).sum()
    ref_tn = ((results_df['true_snow'] == False) & (results_df['refined_detected'] == False)).sum()
    
    ref_precision = ref_tp / (ref_tp + ref_fp) if (ref_tp + ref_fp) > 0 else 0
    ref_recall = ref_tp / (ref_tp + ref_fn) if (ref_tp + ref_fn) > 0 else 0
    ref_f1 = 2 * (ref_precision * ref_recall) / (ref_precision + ref_recall) if (ref_precision + ref_recall) > 0 else 0
    
    print(f"\nRefined Method Performance:")
    print(f"  Precision: {ref_precision:.3f}")
    print(f"  Recall: {ref_recall:.3f}")
    print(f"  F1 Score: {ref_f1:.3f}")
    
    # Analyze failure cases
    print("\n=== Failure Analysis ===")
    
    # False negatives (missed snow)
    fn_df = results_df[(results_df['true_snow'] == True) & (results_df['hsv_detected'] == False)]
    if len(fn_df) > 0:
        print(f"\nFalse Negatives ({len(fn_df)} cases):")
        print(f"  Average brightness: {fn_df['brightness'].mean():.1f}")
        print(f"  Average snow percentage: {fn_df['snow_percentage'].mean():.3f}")
        print(f"  Average contrast: {fn_df['contrast'].mean():.1f}")
    
    # False positives (detected snow when none)
    fp_df = results_df[(results_df['true_snow'] == False) & (results_df['hsv_detected'] == True)]
    if len(fp_df) > 0:
        print(f"\nFalse Positives ({len(fp_df)} cases):")
        print(f"  Average brightness: {fp_df['brightness'].mean():.1f}")
        print(f"  Average snow percentage: {fp_df['snow_percentage'].mean():.3f}")
        print(f"  Average contrast: {fp_df['contrast'].mean():.1f}")


def analyze_hsv_distributions(hsv_distributions):
    """Analyze HSV value distributions for snow vs no-snow images."""
    print("\n=== HSV Distribution Analysis ===")
    
    # Calculate mean HSV values for each class
    if hsv_distributions['snow'] and hsv_distributions['no_snow']:
        snow_hsvs = np.vstack([img.reshape(-1, 3) for img in hsv_distributions['snow'][:50]])
        no_snow_hsvs = np.vstack([img.reshape(-1, 3) for img in hsv_distributions['no_snow'][:50]])
        
        # Sample to avoid memory issues
        snow_sample = snow_hsvs[::100]
        no_snow_sample = no_snow_hsvs[::100]
        
        print(f"\nSnow pixels HSV statistics:")
        print(f"  H: mean={np.mean(snow_sample[:, 0]):.1f}, std={np.std(snow_sample[:, 0]):.1f}")
        print(f"  S: mean={np.mean(snow_sample[:, 1]):.1f}, std={np.std(snow_sample[:, 1]):.1f}")
        print(f"  V: mean={np.mean(snow_sample[:, 2]):.1f}, std={np.std(snow_sample[:, 2]):.1f}")
        
        print(f"\nNo-snow pixels HSV statistics:")
        print(f"  H: mean={np.mean(no_snow_sample[:, 0]):.1f}, std={np.std(no_snow_sample[:, 0]):.1f}")
        print(f"  S: mean={np.mean(no_snow_sample[:, 1]):.1f}, std={np.std(no_snow_sample[:, 1]):.1f}")
        print(f"  V: mean={np.mean(no_snow_sample[:, 2]):.1f}, std={np.std(no_snow_sample[:, 2]):.1f}")


def propose_improved_thresholds(results_df, hsv_distributions):
    """Propose improved thresholds based on analysis."""
    print("\n=== Proposed Improvements ===")
    
    # Current thresholds
    print(f"\nCurrent HSV thresholds:")
    print(f"  Lower: H={config.snow_lower_hsv[0]}, S={config.snow_lower_hsv[1]}, V={config.snow_lower_hsv[2]}")
    print(f"  Upper: H={config.snow_upper_hsv[0]}, S={config.snow_upper_hsv[1]}, V={config.snow_upper_hsv[2]}")
    print(f"  Min pixel percentage: {config.snow_min_pixel_percentage:.1%}")
    
    # Analyze optimal pixel percentage threshold
    best_f1 = 0
    best_threshold = 0
    
    for threshold in np.arange(0.1, 0.8, 0.05):
        tp = ((results_df['true_snow'] == True) & (results_df['snow_percentage'] >= threshold)).sum()
        fp = ((results_df['true_snow'] == False) & (results_df['snow_percentage'] >= threshold)).sum()
        fn = ((results_df['true_snow'] == True) & (results_df['snow_percentage'] < threshold)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal pixel percentage threshold: {best_threshold:.1%} (F1={best_f1:.3f})")
    
    # Suggest multi-stage approach
    print("\n=== Suggested Multi-Stage Approach ===")
    print("""
1. Primary HSV Detection:
   - Lower HSV: [0, 0, 180]  (slightly higher V threshold)
   - Upper HSV: [180, 50, 255]  (slightly lower S threshold)
   - Min pixels: 30% (more lenient)

2. Secondary Validation:
   - Brightness check: mean > 160
   - Texture analysis: low texture variance (smooth snow)
   - Edge density: < 0.1 (snow areas have few edges)

3. Context-Aware Adjustments:
   - Season-based thresholds (winter vs summer)
   - Time-of-day adjustments (morning/evening light)
   - Weather data integration if available

4. Machine Learning Hybrid:
   - Use heuristics for initial filtering
   - Apply ML model only on uncertain cases
   - Combine both predictions with weighted voting
""")


def create_improved_snow_detector():
    """Create an improved snow detection function."""
    print("\n=== Improved Snow Detector Code ===")
    
    code = '''
def detect_snow_improved(image, season='unknown', time_of_day='unknown'):
    """Improved snow detection with multiple stages."""
    
    # Stage 1: Relaxed HSV detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 0, 180])    # Higher V threshold
    upper_hsv = np.array([180, 50, 255])  # Lower S threshold
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    snow_percentage = np.count_nonzero(mask) / mask.size
    
    # Stage 2: Brightness validation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Stage 3: Texture analysis
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = laplacian.var()
    
    # Stage 4: Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Decision logic
    confidence = 0.0
    
    # Base detection
    if snow_percentage >= 0.3:
        confidence += 0.4
    
    # Brightness bonus
    if mean_brightness > 160:
        confidence += 0.2
    elif mean_brightness < 100:
        confidence -= 0.2
    
    # Texture bonus (snow is smooth)
    if texture_variance < 100:
        confidence += 0.2
    
    # Edge penalty (snow has few edges)
    if edge_density < 0.05:
        confidence += 0.2
    elif edge_density > 0.15:
        confidence -= 0.2
    
    # Seasonal adjustments
    if season == 'winter':
        confidence += 0.1
    elif season == 'summer':
        confidence -= 0.1
    
    # Time adjustments
    if time_of_day in ['dawn', 'dusk']:
        # More lenient during difficult lighting
        confidence += 0.1
    
    has_snow = confidence >= 0.5
    
    return has_snow, confidence, {
        'snow_percentage': snow_percentage,
        'brightness': mean_brightness,
        'texture_variance': texture_variance,
        'edge_density': edge_density
    }
'''
    print(code)


def visualize_heuristic_analysis(results_df):
    """Create visualizations of heuristic performance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Snow percentage distribution
    ax1 = axes[0, 0]
    snow_df = results_df[results_df['true_snow'] == True]
    no_snow_df = results_df[results_df['true_snow'] == False]
    
    ax1.hist(snow_df['snow_percentage'], bins=30, alpha=0.5, label='Snow', color='blue')
    ax1.hist(no_snow_df['snow_percentage'], bins=30, alpha=0.5, label='No Snow', color='red')
    ax1.axvline(config.snow_min_pixel_percentage, color='black', linestyle='--', label='Current Threshold')
    ax1.set_xlabel('Snow Pixel Percentage')
    ax1.set_ylabel('Count')
    ax1.set_title('Snow Pixel Distribution')
    ax1.legend()
    
    # Plot 2: Brightness vs Snow Percentage
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['brightness'], results_df['snow_percentage'], 
                         c=results_df['true_snow'], cmap='coolwarm', alpha=0.6)
    ax2.set_xlabel('Mean Brightness')
    ax2.set_ylabel('Snow Pixel Percentage')
    ax2.set_title('Brightness vs Snow Detection')
    plt.colorbar(scatter, ax=ax2, label='Has Snow')
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['true_snow'], results_df['hsv_detected'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('HSV Method Confusion Matrix')
    
    # Plot 4: Feature importance
    ax4 = axes[1, 1]
    features = ['snow_percentage', 'brightness', 'contrast', 'blur_metric']
    importances = []
    
    for feature in features:
        # Simple correlation as proxy for importance
        corr = results_df[feature].corr(results_df['true_snow'])
        importances.append(abs(corr))
    
    ax4.bar(features, importances)
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Correlation with Snow')
    ax4.set_title('Feature Importance for Snow Detection')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/analysis')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'heuristic_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path / 'heuristic_analysis.png'}")


def main():
    """Main analysis workflow."""
    # Use the balanced dataset for analysis
    dataset_path = '/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_balanced.csv'
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Using original dataset instead...")
        dataset_path = '/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_fixed.csv'
    
    # Analyze heuristics
    results_df = analyze_snow_heuristics(dataset_path, sample_size=500)
    
    # Create visualizations
    visualize_heuristic_analysis(results_df)
    
    # Generate improved detector code
    create_improved_snow_detector()
    
    # Save analysis results
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/analysis')
    results_df.to_csv(output_path / 'heuristic_analysis_results.csv', index=False)
    print(f"\nResults saved to: {output_path / 'heuristic_analysis_results.csv'}")


if __name__ == "__main__":
    main()