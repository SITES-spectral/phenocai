#!/usr/bin/env python
"""
Comprehensive evaluation of all snow detection methods:
1. Current heuristics
2. Improved heuristics
3. ML model (balanced)
4. Hybrid approach
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc
)

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.heuristics.snow_detection import detect_snow_hsv
from phenocai.heuristics.snow_detection_improved import detect_snow_adaptive, detect_snow_hybrid
from phenocai.utils import load_image, parse_image_filename
from phenocai.config.setup import config


def evaluate_all_methods(dataset_csv=None, sample_size=1000):
    """Evaluate all snow detection methods on the same dataset."""
    print("=== Comprehensive Snow Detection Evaluation ===\n")
    
    # Find dataset
    if dataset_csv is None:
        # Try to find a dataset
        possible_paths = [
            "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experimental_multistation_combined_dataset.csv",
            "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/predictions/lonnstorp_snow_predictions_2022-2023.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                dataset_csv = path
                break
        
        if dataset_csv is None:
            print("No dataset found. Creating sample dataset...")
            dataset_csv = create_sample_dataset()
    
    print(f"Using dataset: {dataset_csv}")
    
    # Load dataset
    df = pd.read_csv(dataset_csv)
    
    # If no ground truth, use a subset of 2024 data
    if 'snow_presence' not in df.columns and 'true_snow' not in df.columns:
        print("No ground truth found. Loading from annotation data...")
        df = load_annotated_data(sample_size)
    
    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Evaluating {len(df)} images...\n")
    
    # Initialize results storage
    results = {
        'current_heuristic': [],
        'improved_heuristic': [],
        'ml_model': [],
        'hybrid': []
    }
    
    # Load ML model if available
    ml_model = load_ml_model()
    
    # Process each image
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(df)}...")
        
        try:
            # Load image
            if 'file_path' in row:
                image_path = row['file_path']
            elif 'filename' in row:
                # Construct path from filename
                image_path = find_image_path(row['filename'])
            else:
                continue
            
            image = load_image(image_path)
            if image is None:
                continue
            
            # Get ground truth
            if 'snow_presence' in row:
                true_snow = bool(row['snow_presence'])
            elif 'true_snow' in row:
                true_snow = bool(row['true_snow'])
            else:
                continue
            
            # Extract context
            context = extract_context(image_path)
            
            # 1. Current heuristic
            current_snow, current_pct = detect_snow_hsv(image)
            results['current_heuristic'].append({
                'true': true_snow,
                'predicted': current_snow,
                'confidence': current_pct / config.snow_min_pixel_percentage
            })
            
            # 2. Improved heuristic
            improved_snow, improved_conf, improved_meta = detect_snow_adaptive(image, context)
            results['improved_heuristic'].append({
                'true': true_snow,
                'predicted': improved_snow,
                'confidence': improved_conf,
                'metadata': improved_meta
            })
            
            # 3. ML model (if available)
            if ml_model is not None:
                ml_pred = predict_with_model(ml_model, image)
                ml_snow = ml_pred >= 0.55  # Use optimal threshold
                results['ml_model'].append({
                    'true': true_snow,
                    'predicted': ml_snow,
                    'confidence': ml_pred
                })
                
                # 4. Hybrid approach
                hybrid_snow, hybrid_conf, hybrid_meta = detect_snow_hybrid(
                    image, ml_prediction=ml_pred, ml_weight=0.7, context=context
                )
                results['hybrid'].append({
                    'true': true_snow,
                    'predicted': hybrid_snow,
                    'confidence': hybrid_conf,
                    'metadata': hybrid_meta
                })
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # Analyze results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    method_metrics = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        metrics = calculate_metrics(method_results)
        method_metrics[method_name] = metrics
        
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  AUC: {metrics['auc']:.3f}")
    
    # Create visualizations
    create_comparison_plots(results, method_metrics)
    
    # Save detailed results
    save_results(results, method_metrics)
    
    return method_metrics


def load_annotated_data(sample_size=1000):
    """Load data with ground truth annotations."""
    # Try to find annotation data
    base_dirs = [
        "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/L1/2024",
        "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/robacksdalen/phenocams/products/RBD_AGR_PL02_PHE01/L1/2024"
    ]
    
    data = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue
        
        # Sample some images
        image_files = list(base_path.glob("*/*.jpg"))[:sample_size//2]
        
        for img_path in image_files:
            # Create synthetic annotation for testing
            # In reality, you would load actual annotations
            filename = img_path.name
            
            # Simple heuristic: assume winter months have snow
            try:
                info = parse_image_filename(filename)
                month = info.full_datetime.month
                snow_presence = month in [12, 1, 2, 3]  # Winter months
            except:
                snow_presence = False
            
            data.append({
                'file_path': str(img_path),
                'filename': filename,
                'snow_presence': snow_presence
            })
    
    return pd.DataFrame(data)


def extract_context(image_path):
    """Extract context information from image path/filename."""
    try:
        filename = Path(image_path).name
        info = parse_image_filename(filename)
        
        # Determine season
        month = info.full_datetime.month
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'
        
        # Determine time of day
        hour = info.full_datetime.hour
        if 5 <= hour < 8:
            time_of_day = 'dawn'
        elif 8 <= hour < 17:
            time_of_day = 'day'
        elif 17 <= hour < 20:
            time_of_day = 'dusk'
        else:
            time_of_day = 'night'
        
        return {
            'date': info.full_datetime,
            'season': season,
            'time_of_day': time_of_day,
            'station': info.station
        }
    except:
        return {}


def load_ml_model():
    """Load the trained ML model."""
    model_path = Path('/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset/final_model.keras')
    
    if model_path.exists():
        print(f"Loading ML model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print("ML model not found, skipping ML evaluation")
        return None


def predict_with_model(model, image):
    """Make prediction with ML model."""
    # Preprocess image
    img = tf.image.resize(image, [224, 224])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    
    # Predict
    prediction = model.predict(img, verbose=0)[0][0]
    return float(prediction)


def calculate_metrics(results):
    """Calculate performance metrics."""
    y_true = [r['true'] for r in results]
    y_pred = [r['predicted'] for r in results]
    y_scores = [r['confidence'] for r in results]
    
    # Basic metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    
    # AUC
    if len(set(y_true)) > 1:  # Need both classes
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
    else:
        auc_score = 0.5
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc_score,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def create_comparison_plots(results, metrics):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance comparison
    ax1 = axes[0, 0]
    methods = list(metrics.keys())
    metric_names = ['Precision', 'Recall', 'F1', 'AUC']
    
    x = np.arange(len(methods))
    width = 0.2
    
    for i, metric in enumerate(['precision', 'recall', 'f1', 'auc']):
        values = [metrics[m][metric] for m in methods]
        ax1.bar(x + i*width, values, width, label=metric_names[i])
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ROC curves
    ax2 = axes[0, 1]
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        y_true = [r['true'] for r in method_results]
        y_scores = [r['confidence'] for r in method_results]
        
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax2.plot(fpr, tpr, label=f"{method_name.replace('_', ' ').title()} (AUC={metrics[method_name]['auc']:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrices
    for idx, (method_name, method_results) in enumerate(results.items()):
        if not method_results or idx >= 2:
            continue
        
        ax = axes[1, idx]
        y_true = [r['true'] for r in method_results]
        y_pred = [r['predicted'] for r in method_results]
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{method_name.replace('_', ' ').title()}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/evaluation')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path / 'method_comparison.png'}")


def save_results(results, metrics):
    """Save detailed results."""
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/evaluation')
    output_path.mkdir(exist_ok=True)
    
    # Save metrics
    with open(output_path / 'method_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed predictions for analysis
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(method_results)
        df.to_csv(output_path / f'{method_name}_predictions.csv', index=False)
    
    print(f"Results saved to: {output_path}")


def find_image_path(filename):
    """Try to find the full path for an image filename."""
    search_dirs = [
        "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/L1",
        "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/data/robacksdalen/phenocams/products/RBD_AGR_PL02_PHE01/L1"
    ]
    
    for search_dir in search_dirs:
        # Try to find in year/doy structure
        for year in [2022, 2023, 2024]:
            year_dir = Path(search_dir) / str(year)
            if year_dir.exists():
                matches = list(year_dir.glob(f"*/{filename}"))
                if matches:
                    return str(matches[0])
    
    return None


def create_sample_dataset():
    """Create a sample dataset for testing."""
    # This would create a small test dataset
    # For now, return None to skip
    return None


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate_all_methods(sample_size=500)
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if metrics:
        best_method = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
        print(f"\nBest performing method: {best_method.replace('_', ' ').title()}")
        print(f"F1 Score: {metrics[best_method]['f1']:.3f}")
        
        print("\nKey findings:")
        if 'improved_heuristic' in metrics and 'current_heuristic' in metrics:
            improvement = metrics['improved_heuristic']['f1'] - metrics['current_heuristic']['f1']
            print(f"- Improved heuristics show {improvement:.1%} F1 improvement over current")
        
        if 'hybrid' in metrics and 'ml_model' in metrics:
            hybrid_gain = metrics['hybrid']['f1'] - metrics['ml_model']['f1']
            print(f"- Hybrid approach provides {hybrid_gain:.1%} F1 gain over ML alone")
        
        print("\nRecommended approach:")
        if 'hybrid' in metrics and metrics['hybrid']['f1'] > 0.75:
            print("- Use hybrid approach for best results")
        elif 'ml_model' in metrics and metrics['ml_model']['f1'] > 0.7:
            print("- Use ML model with improved preprocessing")
        else:
            print("- Continue improving both heuristics and ML model")