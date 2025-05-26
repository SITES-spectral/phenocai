#!/usr/bin/env python
"""
Simple test of heuristics on sample images.
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.heuristics.snow_detection import detect_snow_hsv, detect_snow_with_refinement
from phenocai.heuristics.image_quality import calculate_image_statistics
from phenocai.config.setup import config


def test_heuristics_on_samples():
    """Test heuristics on a few sample images."""
    print("=== Testing Snow Detection Heuristics ===\n")
    
    # Current HSV thresholds
    print(f"Current HSV thresholds:")
    print(f"  Lower: H={config.snow_lower_hsv[0]}, S={config.snow_lower_hsv[1]}, V={config.snow_lower_hsv[2]}")
    print(f"  Upper: H={config.snow_upper_hsv[0]}, S={config.snow_upper_hsv[1]}, V={config.snow_upper_hsv[2]}")
    print(f"  Min pixel percentage: {config.snow_min_pixel_percentage:.1%}\n")
    
    # Test different threshold combinations
    test_configs = [
        {
            'name': 'Current',
            'lower': np.array([0, 0, 170]),
            'upper': np.array([180, 60, 255]),
            'min_pct': 0.40
        },
        {
            'name': 'Higher V threshold',
            'lower': np.array([0, 0, 180]),
            'upper': np.array([180, 60, 255]),
            'min_pct': 0.40
        },
        {
            'name': 'Lower S threshold',
            'lower': np.array([0, 0, 170]),
            'upper': np.array([180, 50, 255]),
            'min_pct': 0.40
        },
        {
            'name': 'Lower percentage',
            'lower': np.array([0, 0, 170]),
            'upper': np.array([180, 60, 255]),
            'min_pct': 0.30
        },
        {
            'name': 'Combined relaxed',
            'lower': np.array([0, 0, 180]),
            'upper': np.array([180, 50, 255]),
            'min_pct': 0.30
        }
    ]
    
    # Create synthetic test images
    test_images = create_test_images()
    
    print("\n=== Testing on Synthetic Images ===")
    for img_name, image in test_images.items():
        print(f"\n{img_name}:")
        
        for config_test in test_configs:
            has_snow, snow_pct = detect_snow_hsv(
                image, 
                config_test['lower'], 
                config_test['upper'], 
                config_test['min_pct']
            )
            print(f"  {config_test['name']:20s}: Snow={has_snow}, Percentage={snow_pct:.1%}")
    
    # Propose improved detection function
    print("\n=== Improved Multi-Stage Snow Detector ===")
    print_improved_detector()


def create_test_images():
    """Create synthetic test images."""
    images = {}
    
    # Pure white (should be snow)
    images['Pure White'] = np.full((100, 100, 3), 255, dtype=np.uint8)
    
    # Light gray (borderline)
    images['Light Gray'] = np.full((100, 100, 3), 200, dtype=np.uint8)
    
    # Medium gray (not snow)
    images['Medium Gray'] = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    # Blue sky (should not be snow)
    images['Blue Sky'] = np.zeros((100, 100, 3), dtype=np.uint8)
    images['Blue Sky'][:, :] = [255, 128, 0]  # BGR format - blue
    
    # Mixed (50% white, 50% dark)
    images['Half Snow'] = np.zeros((100, 100, 3), dtype=np.uint8)
    images['Half Snow'][:50, :] = 255  # Top half white
    images['Half Snow'][50:, :] = 50   # Bottom half dark
    
    # Textured snow (white with slight variations)
    images['Textured Snow'] = np.random.normal(240, 10, (100, 100, 3))
    images['Textured Snow'] = np.clip(images['Textured Snow'], 0, 255).astype(np.uint8)
    
    return images


def print_improved_detector():
    """Print improved snow detection code."""
    code = '''
def detect_snow_adaptive(image, context=None):
    """
    Adaptive snow detection with multiple validation stages.
    
    Args:
        image: Input image (BGR format)
        context: Optional dict with 'season', 'time_of_day', 'station'
    
    Returns:
        tuple: (has_snow, confidence, metadata)
    """
    if context is None:
        context = {}
    
    # Stage 1: Multiple HSV range detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define multiple HSV ranges for different snow conditions
    hsv_ranges = [
        # Bright snow
        {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255]), 'weight': 1.0},
        # Shadowed snow
        {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 50, 200]), 'weight': 0.7},
        # Blue-tinted snow (morning/evening)
        {'lower': np.array([100, 0, 150]), 'upper': np.array([130, 60, 255]), 'weight': 0.5}
    ]
    
    # Calculate weighted snow percentage
    total_weight = 0
    weighted_percentage = 0
    
    for hsv_range in hsv_ranges:
        mask = cv2.inRange(hsv, hsv_range['lower'], hsv_range['upper'])
        percentage = np.count_nonzero(mask) / mask.size
        weighted_percentage += percentage * hsv_range['weight']
        total_weight += hsv_range['weight']
    
    snow_percentage = weighted_percentage / total_weight
    
    # Stage 2: Validation checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Brightness check
    mean_brightness = np.mean(gray)
    brightness_score = min(mean_brightness / 255, 1.0)
    
    # Texture uniformity (snow is relatively uniform)
    texture_std = np.std(gray)
    uniformity_score = 1.0 - min(texture_std / 128, 1.0)
    
    # Edge density (snow has few edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    edge_score = 1.0 - min(edge_density / 0.2, 1.0)
    
    # Color saturation (snow has low saturation)
    mean_saturation = np.mean(hsv[:, :, 1])
    saturation_score = 1.0 - min(mean_saturation / 128, 1.0)
    
    # Stage 3: Context-aware adjustments
    context_multiplier = 1.0
    
    if context.get('season') == 'winter':
        context_multiplier *= 1.2
    elif context.get('season') == 'summer':
        context_multiplier *= 0.8
    
    if context.get('time_of_day') in ['dawn', 'dusk']:
        # More lenient during difficult lighting
        context_multiplier *= 1.1
    
    # Stage 4: Calculate final confidence
    base_confidence = snow_percentage * 2.0  # Scale up since typical coverage is lower
    
    validation_score = (
        brightness_score * 0.25 +
        uniformity_score * 0.25 +
        edge_score * 0.25 +
        saturation_score * 0.25
    )
    
    confidence = min(base_confidence * validation_score * context_multiplier, 1.0)
    
    # Adaptive threshold based on validation
    if validation_score > 0.7:
        threshold = 0.3  # High quality indicators, lower threshold
    elif validation_score > 0.5:
        threshold = 0.4  # Medium quality
    else:
        threshold = 0.5  # Low quality, be conservative
    
    has_snow = confidence >= threshold
    
    metadata = {
        'snow_percentage': snow_percentage,
        'brightness': mean_brightness,
        'texture_std': texture_std,
        'edge_density': edge_density,
        'mean_saturation': mean_saturation,
        'validation_score': validation_score,
        'threshold_used': threshold
    }
    
    return has_snow, confidence, metadata


# Example usage with seasonal context
context = {
    'season': 'winter',
    'time_of_day': 'noon',
    'station': 'robacksdalen'
}

has_snow, confidence, metadata = detect_snow_adaptive(image, context)
'''
    print(code)


def main():
    """Run heuristic tests."""
    test_heuristics_on_samples()
    
    print("\n=== Key Recommendations ===")
    print("""
1. Use multiple HSV ranges with weights for different snow conditions
2. Add validation checks (brightness, texture, edges, saturation)
3. Implement adaptive thresholds based on image quality
4. Include context information (season, time, location)
5. Consider a hybrid approach: heuristics for clear cases, ML for uncertain ones

The current heuristics are too rigid. The improved approach:
- Handles varying lighting conditions better
- Reduces false positives from bright non-snow objects
- Adapts to seasonal and temporal variations
- Provides confidence scores for uncertainty quantification
""")


if __name__ == "__main__":
    main()