"""
Inference module for applying trained models to new images.
"""

from .predictor import (
    ModelPredictor,
    BatchPredictor,
    PredictionResult,
    process_single_image,
    process_image_directory,
    process_date_range
)

__all__ = [
    'ModelPredictor',
    'BatchPredictor',
    'PredictionResult',
    'process_single_image',
    'process_image_directory',
    'process_date_range'
]