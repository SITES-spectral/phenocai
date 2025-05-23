"""
Image Quality Heuristics

Methods for detecting image quality issues like blur, poor illumination, etc.
"""
import numpy as np
import cv2
import logging
from typing import Tuple, Dict, Any

from ..config.setup import config


logger = logging.getLogger(__name__)


def detect_blur(
    image: np.ndarray,
    threshold: float = None
) -> Tuple[bool, float]:
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        image: Input image
        threshold: Blur threshold (lower values = more blur)
        
    Returns:
        Tuple of (is_blurry, blur_metric)
    """
    if threshold is None:
        threshold = config.discard_blur_threshold_roi
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_metric = laplacian.var()
    
    is_blurry = blur_metric < threshold
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Blur detection: metric={blur_metric:.1f}, threshold={threshold}")
    
    return is_blurry, blur_metric


def detect_low_brightness(
    image: np.ndarray,
    threshold: int = None
) -> Tuple[bool, float]:
    """
    Detect if image has low brightness.
    
    Args:
        image: Input image
        threshold: Minimum brightness threshold
        
    Returns:
        Tuple of (is_dark, mean_brightness)
    """
    if threshold is None:
        threshold = config.discard_illumination_dark_threshold_roi
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    mean_brightness = np.mean(gray)
    is_dark = mean_brightness < threshold
    
    return is_dark, mean_brightness


def detect_high_brightness(
    image: np.ndarray,
    threshold: int = None
) -> Tuple[bool, float]:
    """
    Detect if image has high brightness (overexposed).
    
    Args:
        image: Input image
        threshold: Maximum brightness threshold
        
    Returns:
        Tuple of (is_bright, mean_brightness)
    """
    if threshold is None:
        threshold = config.discard_illumination_bright_threshold_roi
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    mean_brightness = np.mean(gray)
    is_bright = mean_brightness > threshold
    
    return is_bright, mean_brightness


def detect_low_contrast(
    image: np.ndarray,
    threshold: int = None
) -> Tuple[bool, float]:
    """
    Detect if image has low contrast.
    
    Args:
        image: Input image
        threshold: Minimum contrast threshold
        
    Returns:
        Tuple of (is_low_contrast, contrast_value)
    """
    if threshold is None:
        threshold = config.discard_low_contrast_threshold_roi
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate contrast as difference between max and min
    contrast = int(np.max(gray)) - int(np.min(gray))
    is_low_contrast = contrast < threshold
    
    return is_low_contrast, contrast


def should_discard_roi(
    image: np.ndarray,
    check_blur: bool = True,
    check_brightness: bool = True,
    check_contrast: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if ROI should be discarded based on quality checks.
    
    Args:
        image: Input ROI image
        check_blur: Whether to check for blur
        check_brightness: Whether to check brightness
        check_contrast: Whether to check contrast
        
    Returns:
        Tuple of (should_discard, quality_metrics)
    """
    should_discard = False
    metrics = {}
    reasons = []
    
    if check_blur:
        is_blurry, blur_metric = detect_blur(image)
        metrics['blur_metric'] = blur_metric
        metrics['is_blurry'] = is_blurry
        if is_blurry:
            should_discard = True
            reasons.append('blur')
    
    if check_brightness:
        is_dark, brightness = detect_low_brightness(image)
        is_bright, _ = detect_high_brightness(image)
        
        metrics['brightness'] = brightness
        metrics['is_dark'] = is_dark
        metrics['is_bright'] = is_bright
        
        if is_dark:
            should_discard = True
            reasons.append('low_brightness')
        elif is_bright:
            should_discard = True
            reasons.append('high_brightness')
    
    if check_contrast:
        is_low_contrast, contrast = detect_low_contrast(image)
        metrics['contrast'] = contrast
        metrics['is_low_contrast'] = is_low_contrast
        
        if is_low_contrast:
            should_discard = True
            reasons.append('low_contrast')
    
    metrics['discard_reasons'] = reasons
    
    return should_discard, metrics


def calculate_image_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive image statistics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of statistics
    """
    # Convert to grayscale for statistics
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    stats = {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'median': float(np.median(gray)),
        'contrast': float(np.max(gray) - np.min(gray))
    }
    
    # Calculate blur metric
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    stats['blur_metric'] = float(laplacian.var())
    
    # Calculate edge density (useful for quality assessment)
    edges = cv2.Canny(gray, 50, 150)
    stats['edge_density'] = float(np.count_nonzero(edges)) / (edges.shape[0] * edges.shape[1])
    
    return stats


def detect_image_quality_issues(image: np.ndarray) -> list:
    """
    Detect various quality issues in an image.
    
    Args:
        image: Input image
        
    Returns:
        List of quality flag strings
    """
    flags = []
    
    # Check blur
    is_blurry, _ = detect_blur(image)
    if is_blurry:
        flags.append('blur')
    
    # Check illumination
    is_too_dark, _ = detect_low_brightness(image)
    if is_too_dark:
        flags.append('low_brightness')
    
    is_too_bright, _ = detect_high_brightness(image)
    if is_too_bright:
        flags.append('high_brightness')
    
    # Check contrast
    has_low_contrast, _ = detect_low_contrast(image)
    if has_low_contrast:
        flags.append('low_contrast')
    
    # Could add more sophisticated checks here:
    # - fog detection (based on histogram analysis)
    # - lens obstruction (based on circular patterns)
    # - etc.
    
    return flags