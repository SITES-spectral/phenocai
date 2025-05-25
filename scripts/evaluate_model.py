#!/usr/bin/env python3
"""
Evaluate a trained model and calculate proper metrics.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_path, dataset_csv, split='test'):
    """Evaluate model on specified dataset split."""
    
    # Load model
    print(f"Loading model from {model_path}")
    if model_path.endswith('.keras'):
        model = tf.keras.models.load_model(model_path)
    else:
        # Load architecture and weights separately if needed
        raise ValueError("Please provide a .keras model file")
    
    # Load dataset
    df = pd.read_csv(dataset_csv)
    if split:
        df = df[df['split'] == split]
    print(f"Evaluating on {len(df)} {split} samples")
    
    # Prepare data
    image_paths = df['file_path'].values
    labels = df['snow_presence'].astype(int).values
    
    # Make predictions
    predictions = []
    prediction_probs = []
    batch_size = 32
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            batch_images.append(img_array)
        
        batch_images = np.array(batch_images)
        batch_probs = model.predict(batch_images, verbose=0)
        batch_preds = (batch_probs > 0.5).astype(int).flatten()
        
        prediction_probs.extend(batch_probs.flatten())
        predictions.extend(batch_preds)
        
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images", end='\r')
    
    predictions = np.array(predictions)
    prediction_probs = np.array(prediction_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Calculate AUC only if we have both classes
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, prediction_probs)
    else:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    print(f"\n\n{'='*50}")
    print(f"Evaluation Results on {split} set:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nPrediction Distribution:")
    print(f"Predicted 0 (no snow): {np.sum(predictions == 0)} ({np.mean(predictions == 0)*100:.1f}%)")
    print(f"Predicted 1 (snow):    {np.sum(predictions == 1)} ({np.mean(predictions == 1)*100:.1f}%)")
    print(f"\nTrue Distribution:")
    print(f"True 0 (no snow): {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    print(f"True 1 (snow):    {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(labels, predictions, 
                              target_names=['No Snow', 'Snow'],
                              digits=4))
    
    # Find optimal threshold
    print(f"\nFinding optimal threshold...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (prediction_probs > threshold).astype(int)
        f1_thresh = f1_score(labels, preds, zero_division=0)
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Save results
    results = {
        'split': split,
        'num_samples': len(df),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'prediction_distribution': {
            'no_snow': int(np.sum(predictions == 0)),
            'snow': int(np.sum(predictions == 1))
        },
        'true_distribution': {
            'no_snow': int(np.sum(labels == 0)),
            'snow': int(np.sum(labels == 1))
        }
    }
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluate_model.py <model_path> <dataset_csv> [split]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dataset_csv = sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else 'test'
    
    results = evaluate_model(model_path, dataset_csv, split)
    
    # Save results
    output_path = Path(model_path).parent / f'evaluation_{split}_detailed.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")