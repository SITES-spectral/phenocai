"""
Improved Snow Detection Heuristics

Multi-stage adaptive snow detection with validation and context awareness.
"""
import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def detect_snow_adaptive(
    image: np.ndarray,
    context: Optional[Dict[str, Any]] = None,
    return_visualization: bool = False
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Adaptive snow detection with multiple validation stages.
    
    Args:
        image: Input image (BGR format)
        context: Optional dict with 'season', 'time_of_day', 'station', 'date'
        return_visualization: Whether to return mask visualization
    
    Returns:
        tuple: (has_snow, confidence, metadata)
    """
    if context is None:
        context = {}
    
    # Auto-detect context if date provided
    if 'date' in context and 'season' not in context:
        context['season'] = _get_season_from_date(context['date'])
    
    # Stage 1: Multiple HSV range detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define multiple HSV ranges for different snow conditions
    hsv_ranges = [
        # Bright snow (direct sunlight)
        {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255]), 'weight': 1.0},
        # Shadowed snow
        {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 50, 200]), 'weight': 0.7},
        # Blue-tinted snow (morning/evening)
        {'lower': np.array([100, 0, 150]), 'upper': np.array([130, 60, 255]), 'weight': 0.5},
        # Overcast snow (grayish)
        {'lower': np.array([0, 0, 160]), 'upper': np.array([180, 40, 220]), 'weight': 0.6}
    ]
    
    # Calculate weighted snow percentage
    total_weight = 0
    weighted_percentage = 0
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for hsv_range in hsv_ranges:
        mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
        percentage = np.count_nonzero(mask) / mask.size
        weighted_percentage += percentage * hsv_range['weight']
        total_weight += hsv_range['weight']
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    snow_percentage = weighted_percentage / total_weight
    
    # Stage 2: Validation checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Brightness check
    mean_brightness = np.mean(gray)
    brightness_score = _sigmoid(mean_brightness, 150, 30)  # Centered at 150
    
    # Texture uniformity (snow is relatively uniform)
    # Use local standard deviation
    kernel_size = 5
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    texture_uniformity = np.mean(local_std)
    uniformity_score = 1.0 - _sigmoid(texture_uniformity, 20, 10)
    
    # Edge density (snow has few edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    edge_score = 1.0 - _sigmoid(edge_density, 0.1, 20)
    
    # Color saturation (snow has low saturation)
    mean_saturation = np.mean(hsv[:, :, 1])
    saturation_score = 1.0 - _sigmoid(mean_saturation, 40, 20)
    
    # Stage 3: Additional snow-specific features
    
    # White balance check (snow should be neutral)
    b, g, r = cv2.split(image)
    color_variance = np.std([np.mean(b), np.mean(g), np.mean(r)])
    color_balance_score = 1.0 - _sigmoid(color_variance, 20, 10)
    
    # High value pixels concentration
    high_value_mask = hsv[:, :, 2] > 200
    high_value_percentage = np.count_nonzero(high_value_mask) / high_value_mask.size
    high_value_score = _sigmoid(high_value_percentage, 0.3, 10)
    
    # Stage 4: Context-aware adjustments
    context_multiplier = 1.0
    
    # Seasonal adjustments
    season = context.get('season', 'unknown')
    if season == 'winter':
        context_multiplier *= 1.3
    elif season == 'spring':
        context_multiplier *= 1.1
    elif season == 'fall':
        context_multiplier *= 0.9
    elif season == 'summer':
        context_multiplier *= 0.7
    
    # Time of day adjustments
    time_of_day = context.get('time_of_day', 'unknown')
    if time_of_day in ['dawn', 'dusk']:
        context_multiplier *= 1.1  # More lenient during difficult lighting
    elif time_of_day == 'night':
        context_multiplier *= 0.8  # Less likely to have visible snow at night
    
    # Station-specific adjustments
    station = context.get('station', 'unknown')
    if station == 'robacksdalen':  # Northern station
        context_multiplier *= 1.1
    elif station == 'lonnstorp':  # Southern station
        context_multiplier *= 0.95
    
    # Stage 5: Calculate final confidence
    # Weight the different scores
    validation_weights = {
        'brightness': 0.15,
        'uniformity': 0.20,
        'edge': 0.15,
        'saturation': 0.20,
        'color_balance': 0.15,
        'high_value': 0.15
    }
    
    validation_score = (
        brightness_score * validation_weights['brightness'] +
        uniformity_score * validation_weights['uniformity'] +
        edge_score * validation_weights['edge'] +
        saturation_score * validation_weights['saturation'] +
        color_balance_score * validation_weights['color_balance'] +
        high_value_score * validation_weights['high_value']
    )
    
    # Base confidence from snow percentage
    # Use a more nuanced scaling
    if snow_percentage < 0.1:
        base_confidence = snow_percentage * 3.0
    elif snow_percentage < 0.3:
        base_confidence = 0.3 + (snow_percentage - 0.1) * 2.0
    else:
        base_confidence = 0.7 + (snow_percentage - 0.3) * 1.0
    
    confidence = min(base_confidence * validation_score * context_multiplier, 1.0)
    
    # Adaptive threshold based on validation quality
    if validation_score > 0.75:
        threshold = 0.35  # High quality indicators, lower threshold
    elif validation_score > 0.6:
        threshold = 0.45  # Medium quality
    elif validation_score > 0.5:
        threshold = 0.5   # Standard threshold
    else:
        threshold = 0.55  # Low quality, be conservative
    
    has_snow = confidence >= threshold
    
    # Compile metadata
    metadata = {
        'snow_percentage': snow_percentage,
        'brightness': mean_brightness,
        'texture_uniformity': texture_uniformity,
        'edge_density': edge_density,
        'mean_saturation': mean_saturation,
        'color_variance': color_variance,
        'high_value_percentage': high_value_percentage,
        'validation_score': validation_score,
        'context_multiplier': context_multiplier,
        'threshold_used': threshold,
        'individual_scores': {
            'brightness': brightness_score,
            'uniformity': uniformity_score,
            'edge': edge_score,
            'saturation': saturation_score,
            'color_balance': color_balance_score,
            'high_value': high_value_score
        }
    }
    
    if return_visualization:
        metadata['snow_mask'] = combined_mask
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Snow detection: confidence={confidence:.3f}, "
            f"threshold={threshold:.3f}, has_snow={has_snow}, "
            f"validation={validation_score:.3f}"
        )
    
    return has_snow, confidence, metadata


def detect_snow_hybrid(
    image: np.ndarray,
    ml_prediction: Optional[float] = None,
    ml_weight: float = 0.7,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Hybrid snow detection combining heuristics and ML predictions.
    
    Args:
        image: Input image
        ml_prediction: Optional ML model prediction (0-1)
        ml_weight: Weight for ML prediction (0-1)
        context: Optional context information
    
    Returns:
        tuple: (has_snow, confidence, metadata)
    """
    # Get heuristic prediction
    heur_has_snow, heur_confidence, heur_metadata = detect_snow_adaptive(image, context)
    
    if ml_prediction is None:
        # No ML prediction, use heuristics only
        return heur_has_snow, heur_confidence, heur_metadata
    
    # Combine predictions
    heur_weight = 1.0 - ml_weight
    combined_confidence = (ml_prediction * ml_weight + heur_confidence * heur_weight)
    
    # Use adaptive threshold based on agreement
    agreement = 1.0 - abs(ml_prediction - heur_confidence)
    if agreement > 0.8:
        # High agreement, use standard threshold
        threshold = 0.5
    elif agreement > 0.6:
        # Medium agreement, slightly conservative
        threshold = 0.55
    else:
        # Low agreement, examine which source to trust more
        if heur_metadata['validation_score'] > 0.7:
            # Trust heuristics more
            combined_confidence = heur_confidence * 0.8 + ml_prediction * 0.2
            threshold = 0.5
        else:
            # Trust ML more but be conservative
            threshold = 0.6
    
    has_snow = combined_confidence >= threshold
    
    metadata = heur_metadata.copy()
    metadata.update({
        'ml_prediction': ml_prediction,
        'heuristic_confidence': heur_confidence,
        'combined_confidence': combined_confidence,
        'agreement': agreement,
        'hybrid_threshold': threshold,
        'ml_weight': ml_weight
    })
    
    return has_snow, combined_confidence, metadata


def _sigmoid(x: float, center: float, scale: float) -> float:
    """Sigmoid function for smooth transitions."""
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


def _get_season_from_date(date) -> str:
    """Determine season from date."""
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def create_snow_confidence_map(
    image: np.ndarray,
    block_size: int = 32
) -> np.ndarray:
    """
    Create a confidence map showing snow likelihood across the image.
    
    Args:
        image: Input image
        block_size: Size of blocks to analyze
    
    Returns:
        Confidence map (same size as input, values 0-1)
    """
    h, w = image.shape[:2]
    confidence_map = np.zeros((h, w), dtype=np.float32)
    
    # Process image in blocks
    for y in range(0, h - block_size + 1, block_size // 2):
        for x in range(0, w - block_size + 1, block_size // 2):
            # Extract block
            block = image[y:y+block_size, x:x+block_size]
            
            # Analyze block
            _, confidence, _ = detect_snow_adaptive(block)
            
            # Fill confidence map (with overlap handling)
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            confidence_map[y:y_end, x:x_end] = np.maximum(
                confidence_map[y:y_end, x:x_end],
                confidence
            )
    
    # Smooth the confidence map
    confidence_map = cv2.GaussianBlur(confidence_map, (15, 15), 5)
    
    return confidence_map