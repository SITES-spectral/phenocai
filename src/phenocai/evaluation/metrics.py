"""Evaluation metrics and visualization for PhenoCAI models."""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    save_predictions: bool = True
) -> Dict[str, Any]:
    """Comprehensive model evaluation on test dataset.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        output_dir: Directory to save results
        class_names: List of class names
        save_predictions: Whether to save raw predictions
        
    Returns:
        Dictionary of evaluation results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions and true labels
    y_true = []
    y_pred_proba = []
    
    print("Generating predictions...")
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Handle binary vs multi-class
    if y_pred_proba.shape[1] == 1:  # Binary classification
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()
    else:  # Multi-class
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Generate visualizations if output directory provided
    if output_dir:
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, 
            class_names=class_names,
            save_path=output_dir / 'confusion_matrix.png'
        )
        
        # ROC curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            plot_roc_curve(
                y_true, y_pred_proba,
                save_path=output_dir / 'roc_curve.png'
            )
        
        # Classification report
        report = generate_classification_report(
            y_true, y_pred,
            class_names=class_names,
            save_path=output_dir / 'classification_report.txt'
        )
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions if requested
        if save_predictions:
            predictions_data = {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
            with open(output_dir / 'predictions.json', 'w') as f:
                json.dump(predictions_data, f)
    
    return metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate comprehensive metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    # Binary classification specific metrics
    if len(np.unique(y_true)) == 2:
        metrics.update({
            'precision_binary': float(precision_score(y_true, y_pred)),
            'recall_binary': float(recall_score(y_true, y_pred)),
            'f1_score_binary': float(f1_score(y_true, y_pred))
        })
        
        # AUC if probabilities provided
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['auc'] = float(auc(fpr, tpr))
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2:  # Binary
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        })
    
    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> str:
    """Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
        
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            
            # Add confusion matrix
            f.write("\n\nConfusion Matrix:\n")
            cm = confusion_matrix(y_true, y_pred)
            if class_names:
                f.write("Predicted:  " + "  ".join(f"{name:>10}" for name in class_names) + "\n")
                for i, row in enumerate(cm):
                    f.write(f"{class_names[i]:>10}: " + "  ".join(f"{val:>10}" for val in row) + "\n")
            else:
                f.write(str(cm))
    
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize values
        save_path: Path to save plot
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        save_path: Path to save plot
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_samples(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_samples: int = 16,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """Plot sample predictions with images.
    
    Args:
        model: Trained model
        dataset: Dataset with images
        num_samples: Number of samples to plot
        class_names: List of class names
        save_path: Path to save plot
        figsize: Figure size
    """
    # Get a batch of data
    for images, labels in dataset.take(1):
        batch_size = min(len(images), num_samples)
        images = images[:batch_size]
        labels = labels[:batch_size]
        
        # Get predictions
        predictions = model.predict(images)
        
        # Setup plot
        rows = int(np.sqrt(batch_size))
        cols = int(np.ceil(batch_size / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if batch_size > 1 else [axes]
        
        for i in range(batch_size):
            ax = axes[i]
            
            # Display image
            img = images[i].numpy()
            if img.max() <= 1.0:
                img = img * 255
            ax.imshow(img.astype(np.uint8))
            
            # Get prediction
            if predictions.shape[1] == 1:  # Binary
                pred_class = int(predictions[i] > 0.5)
                pred_conf = predictions[i][0] if pred_class == 1 else 1 - predictions[i][0]
            else:  # Multi-class
                pred_class = np.argmax(predictions[i])
                pred_conf = predictions[i][pred_class]
            
            true_class = int(labels[i])
            
            # Format title
            if class_names:
                true_name = class_names[true_class]
                pred_name = class_names[pred_class]
                title = f"True: {true_name}\nPred: {pred_name} ({pred_conf:.2f})"
            else:
                title = f"True: {true_class}\nPred: {pred_class} ({pred_conf:.2f})"
            
            # Color based on correctness
            color = 'green' if pred_class == true_class else 'red'
            ax.set_title(title, color=color, fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        break  # Only process first batch


def analyze_errors(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
    num_worst: int = 20
):
    """Analyze model errors and save worst predictions.
    
    Args:
        model: Trained model
        dataset: Test dataset
        output_dir: Directory for outputs
        class_names: List of class names
        num_worst: Number of worst predictions to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect predictions and errors
    errors = []
    
    for batch_idx, (images, labels) in enumerate(dataset):
        predictions = model.predict(images, verbose=0)
        
        for i in range(len(images)):
            true_label = int(labels[i])
            
            if predictions.shape[1] == 1:  # Binary
                pred_prob = predictions[i][0]
                pred_label = int(pred_prob > 0.5)
                confidence = pred_prob if pred_label == 1 else 1 - pred_prob
                error = abs(pred_prob - true_label)
            else:  # Multi-class
                pred_label = np.argmax(predictions[i])
                confidence = predictions[i][pred_label]
                error = 1.0 - predictions[i][true_label]
            
            if pred_label != true_label:
                errors.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': float(confidence),
                    'error': float(error)
                })
    
    # Sort by error (worst first)
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    # Save error analysis
    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump({
            'total_errors': len(errors),
            'error_rate': len(errors) / sum(1 for _ in dataset.unbatch()),
            'worst_errors': errors[:num_worst]
        }, f, indent=2)
    
    print(f"Error analysis saved. Total errors: {len(errors)}")
    
    return errors