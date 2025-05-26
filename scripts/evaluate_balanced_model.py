#!/usr/bin/env python
"""
Evaluate models trained on balanced vs unbalanced datasets.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    classification_report, confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.utils import load_image, parse_filename_info


def evaluate_model(model_path, dataset_csv, model_name="Model"):
    """Evaluate a model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_csv}")
    print('='*60)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    df = pd.read_csv(dataset_csv)
    test_df = df[df['split'] == 'test'].copy()
    print(f"\nTest set size: {len(test_df)} samples")
    print(f"Class distribution:")
    print(f"  Snow: {test_df['snow_presence'].sum()} ({test_df['snow_presence'].mean()*100:.1f}%)")
    print(f"  No snow: {(~test_df['snow_presence']).sum()} ({(~test_df['snow_presence']).mean()*100:.1f}%)")
    
    # Create test dataset directly from file paths
    import tensorflow as tf
    
    # Create a simple dataset from the test dataframe
    def load_and_preprocess_image(row):
        """Load and preprocess a single image."""
        image = tf.io.read_file(row['file_path'])
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        # Apply MobileNetV2 preprocessing
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    
    # Create dataset from file paths
    file_paths = test_df['file_path'].values
    labels = test_df['snow_presence'].astype(int).values
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def load_image_and_label(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label
    
    test_dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_dataset:
        y_true.extend(labels.numpy())
        y_pred_probs.extend(model.predict(images, verbose=0).flatten())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate metrics at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {
        'model_name': model_name,
        'test_samples': len(test_df),
        'snow_samples': test_df['snow_presence'].sum(),
        'no_snow_samples': (~test_df['snow_presence']).sum(),
        'thresholds': {}
    }
    
    print("\n" + "-"*60)
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10} | {'Accuracy':>10}")
    print("-"*60)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        # Calculate metrics with zero_division handling
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true)
        
        results['thresholds'][threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
        
        print(f"{threshold:>10.1f} | {precision:>10.3f} | {recall:>10.3f} | {f1:>10.3f} | {accuracy:>10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("-"*60)
    print(f"Best F1 Score: {best_f1:.3f} at threshold {best_threshold}")
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    results['auc'] = roc_auc
    results['best_threshold'] = best_threshold
    results['best_f1'] = best_f1
    
    print(f"\nAUC-ROC: {roc_auc:.3f}")
    
    # Detailed classification report at best threshold
    y_pred_best = (y_pred_probs >= best_threshold).astype(int)
    print(f"\nDetailed Classification Report (threshold={best_threshold}):")
    print(classification_report(y_true, y_pred_best, 
                              target_names=['No Snow', 'Snow'],
                              digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_best)
    print("\nConfusion Matrix:")
    print(f"{'':>15} | {'Predicted No Snow':>18} | {'Predicted Snow':>15}")
    print("-"*50)
    print(f"{'Actual No Snow':>15} | {cm[0,0]:>18} | {cm[0,1]:>15}")
    print(f"{'Actual Snow':>15} | {cm[1,0]:>18} | {cm[1,1]:>15}")
    
    return results


def compare_models():
    """Compare balanced and unbalanced models."""
    # Model paths
    unbalanced_model = "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_snow_binary_12epoch"
    balanced_model = "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset"
    
    # Dataset paths
    unbalanced_dataset = "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_fixed.csv"
    balanced_dataset = "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_balanced.csv"
    
    # Evaluate both models on their respective test sets
    results_unbalanced = evaluate_model(unbalanced_model, unbalanced_dataset, "Unbalanced Model")
    results_balanced = evaluate_model(balanced_model, balanced_dataset, "Balanced Model")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':>20} | {'Unbalanced':>15} | {'Balanced':>15}")
    print("-"*55)
    print(f"{'Test Set Size':>20} | {results_unbalanced['test_samples']:>15} | {results_balanced['test_samples']:>15}")
    print(f"{'Snow Samples':>20} | {results_unbalanced['snow_samples']:>15} | {results_balanced['snow_samples']:>15}")
    print(f"{'No Snow Samples':>20} | {results_unbalanced['no_snow_samples']:>15} | {results_balanced['no_snow_samples']:>15}")
    print(f"{'AUC-ROC':>20} | {results_unbalanced['auc']:>15.3f} | {results_balanced['auc']:>15.3f}")
    print(f"{'Best F1 Score':>20} | {results_unbalanced['best_f1']:>15.3f} | {results_balanced['best_f1']:>15.3f}")
    print(f"{'Best Threshold':>20} | {results_unbalanced['best_threshold']:>15.1f} | {results_balanced['best_threshold']:>15.1f}")
    
    # Performance at standard threshold (0.5)
    print(f"\n{'At Threshold 0.5':>20} | {'Unbalanced':>15} | {'Balanced':>15}")
    print("-"*55)
    metrics_05_unbal = results_unbalanced['thresholds'][0.5]
    metrics_05_bal = results_balanced['thresholds'][0.5]
    print(f"{'Precision':>20} | {metrics_05_unbal['precision']:>15.3f} | {metrics_05_bal['precision']:>15.3f}")
    print(f"{'Recall':>20} | {metrics_05_unbal['recall']:>15.3f} | {metrics_05_bal['recall']:>15.3f}")
    print(f"{'F1 Score':>20} | {metrics_05_unbal['f1']:>15.3f} | {metrics_05_bal['f1']:>15.3f}")
    print(f"{'Accuracy':>20} | {metrics_05_unbal['accuracy']:>15.3f} | {metrics_05_bal['accuracy']:>15.3f}")
    
    # Save results
    output_path = Path("/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/evaluation_results")
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results
    import json
    with open(output_path / "model_comparison_results.json", 'w') as f:
        json.dump({
            'unbalanced': results_unbalanced,
            'balanced': results_balanced
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path / 'model_comparison_results.json'}")
    
    # Create visualization
    create_comparison_plots(results_unbalanced, results_balanced, output_path)


def create_comparison_plots(results_unbal, results_bal, output_path):
    """Create comparison plots for the two models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: F1 scores across thresholds
    thresholds = list(results_unbal['thresholds'].keys())
    f1_unbal = [results_unbal['thresholds'][t]['f1'] for t in thresholds]
    f1_bal = [results_bal['thresholds'][t]['f1'] for t in thresholds]
    
    axes[0].plot(thresholds, f1_unbal, 'b-o', label='Unbalanced')
    axes[0].plot(thresholds, f1_bal, 'r-o', label='Balanced')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall trade-off
    prec_unbal = [results_unbal['thresholds'][t]['precision'] for t in thresholds]
    rec_unbal = [results_unbal['thresholds'][t]['recall'] for t in thresholds]
    prec_bal = [results_bal['thresholds'][t]['precision'] for t in thresholds]
    rec_bal = [results_bal['thresholds'][t]['recall'] for t in thresholds]
    
    axes[1].plot(rec_unbal, prec_unbal, 'b-o', label='Unbalanced')
    axes[1].plot(rec_bal, prec_bal, 'r-o', label='Balanced')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of best performance
    metrics = ['AUC', 'Best F1', 'Precision@0.5', 'Recall@0.5']
    unbal_values = [
        results_unbal['auc'],
        results_unbal['best_f1'],
        results_unbal['thresholds'][0.5]['precision'],
        results_unbal['thresholds'][0.5]['recall']
    ]
    bal_values = [
        results_bal['auc'],
        results_bal['best_f1'],
        results_bal['thresholds'][0.5]['precision'],
        results_bal['thresholds'][0.5]['recall']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[2].bar(x - width/2, unbal_values, width, label='Unbalanced', color='blue', alpha=0.7)
    axes[2].bar(x + width/2, bal_values, width, label='Balanced', color='red', alpha=0.7)
    axes[2].set_xlabel('Metrics')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Model Performance Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_path / 'model_comparison_plots.png'}")


if __name__ == "__main__":
    compare_models()