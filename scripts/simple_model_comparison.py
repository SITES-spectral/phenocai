#!/usr/bin/env python
"""
Simple model evaluation comparison.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model_simple(model_path, dataset_csv, model_name):
    """Simple evaluation of a model."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print('='*60)
    
    # Load model (handle SavedModel format)
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        # For SavedModel format
        model = tf.saved_model.load(model_path)
        # Get the inference function
        model = model.signatures['serving_default']
    
    # Load test data
    df = pd.read_csv(dataset_csv)
    test_df = df[df['split'] == 'test'].sample(n=1000, random_state=42)  # Sample for speed
    
    print(f"Test samples: {len(test_df)}")
    print(f"Snow: {test_df['snow_presence'].sum()} ({test_df['snow_presence'].mean()*100:.1f}%)")
    
    # Prepare data
    images = []
    labels = []
    
    for _, row in test_df.iterrows():
        try:
            # Load and preprocess image
            img = tf.io.read_file(row['file_path'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            
            images.append(img)
            labels.append(int(row['snow_presence']))
        except:
            continue
    
    images = tf.stack(images)
    labels = np.array(labels)
    
    # Predict
    if hasattr(model, 'predict'):
        predictions = model.predict(images, batch_size=32, verbose=0)
        pred_probs = predictions.flatten()
    else:
        # For SavedModel format
        predictions = model(tf.constant(images))
        # Get the output tensor (might be named differently)
        output_key = list(predictions.keys())[0]
        pred_probs = predictions[output_key].numpy().flatten()
    
    # Evaluate at different thresholds
    print("\nThreshold Analysis:")
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-"*50)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_binary = (pred_probs >= threshold).astype(int)
        
        tp = np.sum((labels == 1) & (pred_binary == 1))
        fp = np.sum((labels == 0) & (pred_binary == 1))
        fn = np.sum((labels == 1) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>10.1f} | {precision:>10.3f} | {recall:>10.3f} | {f1:>10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest F1: {best_f1:.3f} at threshold {best_threshold}")
    
    # Confusion matrix at 0.5
    pred_05 = (pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, pred_05)
    print(f"\nConfusion Matrix (threshold=0.5):")
    print(f"TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    return {
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'cm': cm
    }


def main():
    # Paths
    models = {
        'Unbalanced': {
            'model': '/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_snow_binary_12epoch',
            'dataset': '/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_fixed.csv'
        },
        'Balanced': {
            'model': '/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset',
            'dataset': '/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/data/lonnstorp/training_datasets/multistation_snow_dataset_balanced.csv'
        }
    }
    
    results = {}
    for name, paths in models.items():
        results[name] = evaluate_model_simple(paths['model'], paths['dataset'], name)
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':>15} | {'Best F1':>10} | {'Threshold':>10}")
    print("-"*40)
    for name, res in results.items():
        print(f"{name:>15} | {res['best_f1']:>10.3f} | {res['best_threshold']:>10.1f}")


if __name__ == "__main__":
    main()