"""
PhenoCAI Heuristics Module

Rule-based methods for initial classification of phenocam images.
"""

from .snow_detection import detect_snow_hsv
from .image_quality import (
    detect_blur,
    detect_low_brightness,
    detect_high_brightness,
    detect_low_contrast,
    should_discard_roi
)

__all__ = [
    'detect_snow_hsv',
    'detect_blur',
    'detect_low_brightness',
    'detect_high_brightness',
    'detect_low_contrast',
    'should_discard_roi'
]