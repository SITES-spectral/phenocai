"""
Snow Detection Heuristics

HSV-based snow detection for phenocam images.
"""
import numpy as np
import cv2
import logging
from typing import Tuple, Optional

from ..config.setup import config


logger = logging.getLogger(__name__)


def detect_snow_hsv(
    image: np.ndarray,
    lower_hsv: Optional[np.ndarray] = None,
    upper_hsv: Optional[np.ndarray] = None,
    min_pixel_percentage: Optional[float] = None
) -> Tuple[bool, float]:
    """
    Detect snow in image using HSV color thresholding.
    
    Args:
        image: Input image (BGR or RGB format)
        lower_hsv: Lower HSV threshold (default from config)
        upper_hsv: Upper HSV threshold (default from config)
        min_pixel_percentage: Minimum percentage of pixels to classify as snow
        
    Returns:
        Tuple of (has_snow, snow_pixel_percentage)
    """
    if lower_hsv is None:
        lower_hsv = config.snow_lower_hsv
    if upper_hsv is None:
        upper_hsv = config.snow_upper_hsv
    if min_pixel_percentage is None:
        min_pixel_percentage = config.snow_min_pixel_percentage
    
    # Convert to HSV
    if len(image.shape) == 2:
        # Grayscale image
        logger.warning("Snow detection on grayscale image may be less accurate")
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    else:
        # Assume BGR format (OpenCV default)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for snow pixels
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Calculate percentage of snow pixels
    total_pixels = mask.shape[0] * mask.shape[1]
    snow_pixels = np.count_nonzero(mask)
    snow_percentage = snow_pixels / total_pixels
    
    # Determine if snow is present
    has_snow = snow_percentage >= min_pixel_percentage
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Snow detection: {snow_percentage:.1%} pixels, threshold: {min_pixel_percentage:.1%}")
    
    return has_snow, snow_percentage


def detect_snow_with_refinement(
    image: np.ndarray,
    roi_name: str = "ROI",
    use_brightness_check: bool = True
) -> Tuple[bool, float, dict]:
    """
    Enhanced snow detection with additional refinements.
    
    Args:
        image: Input image
        roi_name: Name of ROI for logging
        use_brightness_check: Whether to check brightness consistency
        
    Returns:
        Tuple of (has_snow, confidence, metadata)
    """
    # Basic HSV detection
    has_snow_hsv, snow_percentage = detect_snow_hsv(image)
    
    metadata = {
        'hsv_snow_percentage': snow_percentage,
        'method': 'hsv'
    }
    
    if not has_snow_hsv:
        return False, 1.0 - snow_percentage, metadata
    
    # Additional brightness check for snow
    if use_brightness_check:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)
        
        # Snow should be bright
        if mean_brightness < 150:  # Threshold for snow brightness
            logger.debug(f"Snow detection rejected for {roi_name}: low brightness ({mean_brightness:.1f})")
            metadata['brightness_check_failed'] = True
            metadata['mean_brightness'] = mean_brightness
            return False, 0.5, metadata
    
    # Calculate confidence based on percentage
    confidence = min(snow_percentage / (config.snow_min_pixel_percentage * 2), 1.0)
    metadata['confidence'] = confidence
    
    return True, confidence, metadata


def create_snow_mask(
    image: np.ndarray,
    lower_hsv: Optional[np.ndarray] = None,
    upper_hsv: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create binary mask of snow pixels.
    
    Args:
        image: Input image
        lower_hsv: Lower HSV threshold
        upper_hsv: Upper HSV threshold
        
    Returns:
        Binary mask (255 for snow, 0 for non-snow)
    """
    if lower_hsv is None:
        lower_hsv = config.snow_lower_hsv
    if upper_hsv is None:
        upper_hsv = config.snow_upper_hsv
    
    # Convert to HSV
    if len(image.shape) == 2:
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    return mask