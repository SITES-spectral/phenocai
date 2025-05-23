"""Model evaluation utilities for PhenoCAI."""

from .metrics import (
    evaluate_model,
    calculate_metrics,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_samples,
    analyze_errors
)

__all__ = [
    'evaluate_model',
    'calculate_metrics',
    'generate_classification_report',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_prediction_samples',
    'analyze_errors'
]