"""Custom validation metrics callback to handle edge cases."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict, Any


class ValidationMetricsCallback(keras.callbacks.Callback):
    """Custom callback to calculate validation metrics properly."""
    
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        """Calculate metrics at end of each epoch."""
        if logs is None:
            logs = {}
            
        # Get validation data
        val_x, val_y = self.validation_data
        
        # Get predictions
        y_pred_proba = self.model.predict(val_x, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = val_y.flatten()
        
        # Calculate metrics with zero_division parameter to handle edge cases
        val_precision = precision_score(y_true, y_pred, zero_division=0)
        val_recall = recall_score(y_true, y_pred, zero_division=0)
        val_f1 = f1_score(y_true, y_pred, zero_division=0)
        val_accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate class distribution
        val_positive_rate = np.mean(y_pred)
        val_true_positive_rate = np.mean(y_true)
        
        # Update logs
        logs['val_precision_custom'] = val_precision
        logs['val_recall_custom'] = val_recall
        logs['val_f1_score'] = val_f1
        logs['val_accuracy_custom'] = val_accuracy
        logs['val_predicted_positive_rate'] = val_positive_rate
        logs['val_true_positive_rate'] = val_true_positive_rate
        
        # Print metrics
        print(f"\nValidation Metrics (Epoch {epoch + 1}):")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Predicted positive rate: {val_positive_rate:.4f}")
        print(f"  True positive rate: {val_true_positive_rate:.4f}")
        
        # Check for potential issues
        if val_positive_rate == 0:
            print("  ⚠️  Warning: Model predicting all negatives on validation set")
        elif val_positive_rate == 1:
            print("  ⚠️  Warning: Model predicting all positives on validation set")
            
        return logs


class ThresholdTuningCallback(keras.callbacks.Callback):
    """Callback to find optimal classification threshold."""
    
    def __init__(self, validation_data, thresholds=None):
        super().__init__()
        self.validation_data = validation_data
        self.thresholds = thresholds or np.arange(0.1, 0.9, 0.05)
        self.best_threshold = 0.5
        self.best_f1 = 0
        
    def on_epoch_end(self, epoch, logs=None):
        """Find best threshold at end of each epoch."""
        if epoch % 5 == 0:  # Only check every 5 epochs to save time
            val_x, val_y = self.validation_data
            y_pred_proba = self.model.predict(val_x, verbose=0)
            y_true = val_y.flatten()
            
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in self.thresholds:
                y_pred = (y_pred_proba > threshold).astype(int).flatten()
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.best_threshold = best_threshold
            self.best_f1 = best_f1
            
            if logs is not None:
                logs['best_threshold'] = best_threshold
                logs['best_f1_score'] = best_f1
            
            print(f"\n  Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")