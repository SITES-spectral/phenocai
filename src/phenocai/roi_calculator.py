"""
ROI_00 Calculator for PhenoCAI

Calculates the full image ROI (excluding sky) for phenocam images.
Based on methods from phenotag/phenocams packages.
"""

import numpy as np
import cv2
import gc
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def serialize_polygons(phenocam_rois: Dict) -> Dict:
    """
    Converts a dictionary of polygons to be YAML-friendly by converting tuples to lists.
    
    Based on phenotag implementation.
    
    Args:
        phenocam_rois: Dictionary of ROI definitions
        
    Returns:
        YAML-friendly dictionary with lists instead of tuples
    """
    yaml_friendly_rois = {}
    for roi, polygon in phenocam_rois.items():
        yaml_friendly_polygon = {
            'points': [list(point) for point in polygon['points']],
            'color': list(polygon['color']),
            'thickness': polygon['thickness']
        }
        if 'description' in polygon:
            yaml_friendly_polygon['description'] = polygon['description']
        if 'auto_generated' in polygon:
            yaml_friendly_polygon['auto_generated'] = polygon['auto_generated']
        yaml_friendly_rois[roi] = yaml_friendly_polygon
    return yaml_friendly_rois


def deserialize_polygons(yaml_friendly_rois: Dict) -> Dict:
    """
    Converts YAML-friendly polygons back to their original format with tuples.
    
    Based on phenotag implementation.
    
    Args:
        yaml_friendly_rois: Dictionary with list-based points
        
    Returns:
        Dictionary with tuple-based points
    """
    original_rois = {}
    for roi_name, roi_data in yaml_friendly_rois.items():
        points = []
        for point in roi_data['points']:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((int(point[0]), int(point[1])))
        
        original_roi = {
            'points': points,
            'color': tuple(int(c) for c in roi_data['color'][:3]),
            'thickness': int(roi_data.get('thickness', 2)),
            'alpha': 0.0  # Disable filling
        }
        
        # Preserve additional fields
        if 'description' in roi_data:
            original_roi['description'] = roi_data['description']
        if 'auto_generated' in roi_data:
            original_roi['auto_generated'] = roi_data['auto_generated']
            
        original_rois[roi_name] = original_roi
    return original_rois


def calculate_roi_00(
    image: np.ndarray,
    horizon_method: str = 'gradient',
    horizon_fraction: float = 0.15,
    padding: int = 10
) -> List[Tuple[int, int]]:
    """
    Calculate ROI_00 (full image excluding sky) polygon points.
    
    Args:
        image: Input image as numpy array
        horizon_method: Method to detect horizon ('gradient', 'color', 'fixed')
        horizon_fraction: For 'fixed' method, fraction from top as horizon
        padding: Pixels to add as padding from edges
        
    Returns:
        List of (x, y) points defining ROI_00 polygon
    """
    height, width = image.shape[:2]
    
    if horizon_method == 'gradient':
        horizon_y = detect_horizon_gradient(image)
    elif horizon_method == 'color':
        horizon_y = detect_horizon_color(image)
    else:  # fixed
        horizon_y = int(height * horizon_fraction)
    
    # Add some padding below horizon to ensure no sky
    horizon_y = min(height - padding, horizon_y + padding)
    
    # Create polygon points (clockwise from top-left)
    points = [
        (int(padding), int(horizon_y)),                    # Top-left
        (int(width - padding), int(horizon_y)),            # Top-right
        (int(width - padding), int(height - padding)),     # Bottom-right
        (int(padding), int(height - padding))              # Bottom-left
    ]
    
    return points


def detect_horizon_gradient(image: np.ndarray) -> int:
    """
    Detect horizon using advanced sky detection algorithm from phenotag/phenocams.
    
    Uses HSV color space to detect blue and white/cloudy sky regions.
    
    Args:
        image: Input image (RGB format)
        
    Returns:
        Y-coordinate of detected horizon (bottom edge of sky)
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width = image.shape[:2]
    
    # Focus on top third of image where sky typically appears
    top_third_height = height // 3
    sky_line = 0  # Default if no sky found
    
    try:
        # Process in chunks for memory efficiency
        chunk_size = 500
        sky_threshold = 0.3  # 30% of row must be sky
        row_sampling = 5  # Check every 5th row for speed
        
        for start_y in range(0, top_third_height, chunk_size):
            end_y = min(start_y + chunk_size, top_third_height)
            hsv_chunk = hsv[start_y:end_y, :]
            
            # Blue sky: hue 90-140, saturation 50-255, value 150-255
            sky_mask_blue = cv2.inRange(
                hsv_chunk, 
                np.array([90, 50, 150]), 
                np.array([140, 255, 255])
            )
            
            # White/cloudy sky: any hue, saturation 0-50, value 180-255
            sky_mask_white = cv2.inRange(
                hsv_chunk, 
                np.array([0, 0, 180]), 
                np.array([180, 50, 255])
            )
            
            # Combine masks
            sky_mask_chunk = cv2.bitwise_or(sky_mask_blue, sky_mask_white)
            
            # Find the last sky boundary (bottom edge of sky)
            for y_offset in range(0, end_y - start_y, row_sampling):
                if y_offset < sky_mask_chunk.shape[0]:
                    row_sum = np.sum(sky_mask_chunk[y_offset, :]) / 255
                    if row_sum > width * sky_threshold:
                        sky_line = start_y + y_offset
                        # Continue to find the bottom edge of sky
            
            # Clean up memory
            del hsv_chunk, sky_mask_blue, sky_mask_white, sky_mask_chunk
    
    except Exception as e:
        logger.warning(f"Sky detection failed: {e}. Using fallback.")
        sky_line = int(height * 0.15)
    
    return sky_line


def detect_horizon_color(image: np.ndarray) -> int:
    """
    Detect horizon using color-based method (same as gradient method).
    
    This is kept for backward compatibility. Uses the same advanced
    sky detection algorithm from phenotag/phenocams.
    
    Args:
        image: Input image (RGB format)
        
    Returns:
        Y-coordinate of detected horizon
    """
    # Use the same algorithm as gradient method
    return detect_horizon_gradient(image)


def calculate_sky_mask(image: np.ndarray, roi_00_points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Calculate binary mask where sky pixels are 0 and ground pixels are 1.
    
    Args:
        image: Input image
        roi_00_points: ROI_00 polygon points
        
    Returns:
        Binary mask (same size as image)
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert points to numpy array
    points = np.array(roi_00_points, dtype=np.int32)
    
    # Fill the polygon
    cv2.fillPoly(mask, [points], 255)
    
    return mask


def add_roi_00_to_station_config_in_place(
    config: Dict,
    station_name: str,
    instrument_id: str,
    sample_image_path: Optional[Path] = None,
    force: bool = False
) -> Dict:
    """
    Calculate and add ROI_00 to station configuration (in-place).
    
    Args:
        config: Existing configuration dictionary
        station_name: Station name (e.g., 'lonnstorp')
        instrument_id: Instrument ID (e.g., 'LON_AGR_PL01_PHE01')
        sample_image_path: Optional sample image to calculate ROI_00
        force: Force recalculation even if ROI_00 exists
        
    Returns:
        Updated configuration dictionary
    """
    # Navigate to instrument
    try:
        stations = config['stations']
        station = stations[station_name]
        platforms = station['phenocams']['platforms']
        
        # Find the instrument
        instrument = None
        for platform_name, platform_data in platforms.items():
            if instrument_id in platform_data['instruments']:
                instrument = platform_data['instruments'][instrument_id]
                break
        
        if not instrument:
            raise ValueError(f"Instrument {instrument_id} not found for station {station_name}")
        
        # Check if ROI_00 already exists
        if 'rois' not in instrument:
            instrument['rois'] = {}
        
        if 'ROI_00' in instrument['rois'] and instrument['rois']['ROI_00'] is not None and not force:
            logger.info(f"ROI_00 already exists for {station_name} - {instrument_id}. Use --force to recalculate.")
            return config
        
        # Calculate ROI_00
        if sample_image_path and sample_image_path.exists():
            # Load sample image
            image = cv2.imread(str(sample_image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate ROI_00 points
            roi_00_points = calculate_roi_00(image, 'gradient')
        else:
            # Use default based on typical phenocam setup
            # Assuming 4608x3456 resolution (common for phenocams)
            # Sky typically in top 15-20% of image
            logger.warning(f"No sample image provided for {station_name}, using default ROI_00")
            roi_00_points = [
                (10, 520),      # ~15% from top
                (4598, 520),
                (4598, 3446),
                (10, 3446)
            ]
        
        # Add ROI_00 to instrument
        # Ensure points are lists, not tuples
        roi_00_points_list = [[x, y] for x, y in roi_00_points]
        
        instrument['rois']['ROI_00'] = {
            'points': roi_00_points_list,
            'color': [255, 255, 255],  # White
            'thickness': 7,
            'description': 'Full image excluding sky (auto-calculated)',
            'auto_generated': True
        }
        
        logger.info(f"Added ROI_00 to {station_name} - {instrument_id}")
        
        return config
        
    except KeyError as e:
        logger.error(f"Error navigating station configuration: {e}")
        raise


def add_roi_00_to_station_config(
    station_yaml_path: Path,
    station_name: str,
    instrument_id: str,
    sample_image_path: Optional[Path] = None,
    horizon_method: str = 'gradient',
    force: bool = False
) -> Dict:
    """
    Calculate and add ROI_00 to station configuration.
    
    Args:
        station_yaml_path: Path to stations.yaml
        station_name: Station name (e.g., 'lonnstorp')
        instrument_id: Instrument ID (e.g., 'LON_AGR_PL01_PHE01')
        sample_image_path: Optional sample image to calculate ROI_00
        horizon_method: Method to detect horizon
        
    Returns:
        Updated configuration dictionary
    """
    # Load existing configuration
    with open(station_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Navigate to instrument
    try:
        stations = config['stations']
        station = stations[station_name]
        platforms = station['phenocams']['platforms']
        
        # Find the instrument
        instrument = None
        for platform_name, platform_data in platforms.items():
            if instrument_id in platform_data['instruments']:
                instrument = platform_data['instruments'][instrument_id]
                break
        
        if not instrument:
            raise ValueError(f"Instrument {instrument_id} not found for station {station_name}")
        
        # Check if ROI_00 already exists
        if 'rois' not in instrument:
            instrument['rois'] = {}
        
        if 'ROI_00' in instrument['rois'] and instrument['rois']['ROI_00'] is not None and not force:
            logger.info(f"ROI_00 already exists for {station_name} - {instrument_id}. Use --force to recalculate.")
            return config
        
        # Calculate ROI_00
        if sample_image_path and sample_image_path.exists():
            # Load sample image
            image = cv2.imread(str(sample_image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate ROI_00 points
            roi_00_points = calculate_roi_00(image, horizon_method)
        else:
            # Use default based on typical phenocam setup
            # Assuming 4608x3456 resolution (common for phenocams)
            # Sky typically in top 15-20% of image
            logger.warning(f"No sample image provided for {station_name}, using default ROI_00")
            roi_00_points = [
                (10, 520),      # ~15% from top
                (4598, 520),
                (4598, 3446),
                (10, 3446)
            ]
        
        # Add ROI_00 to configuration
        instrument['rois']['ROI_00'] = {
            'points': [[int(x), int(y)] for x, y in roi_00_points],
            'color': [255, 255, 255],  # White
            'thickness': 7,
            'description': 'Full image excluding sky (auto-calculated)',
            'auto_generated': True
        }
        
        logger.info(f"Added ROI_00 to {station_name} - {instrument_id}")
        
        return config
        
    except KeyError as e:
        logger.error(f"Error navigating station configuration: {e}")
        raise


def update_all_stations_roi_00(
    station_yaml_path: Path,
    sample_images_dir: Optional[Path] = None,
    force: bool = False
) -> None:
    """
    Update all stations with ROI_00 calculations.
    
    Args:
        station_yaml_path: Path to stations.yaml
        sample_images_dir: Optional directory containing sample images
    """
    import os
    
    # Primary stations and instruments to update
    updates = [
        ('lonnstorp', 'LON_AGR_PL01_PHE01'),
        ('lonnstorp', 'LON_AGR_PL01_PHE02'),
        ('lonnstorp', 'LON_AGR_PL01_PHE03'),
        ('robacksdalen', 'RBD_AGR_PL01_PHE01'),
        ('robacksdalen', 'RBD_AGR_PL02_PHE01')
    ]
    
    # Load configuration ONCE at the beginning
    with open(station_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    updated = False
    
    # Get PHENOCAMS_DATA_DIR if no sample directory provided
    phenocams_data_dir = os.environ.get('PHENOCAMS_DATA_DIR')
    
    for station_name, instrument_id in updates:
        try:
            # Look for sample image
            sample_image = None
            
            # First try provided directory
            if sample_images_dir:
                # Try to find a sample image for this instrument
                pattern = f"{station_name}*{instrument_id}*.jpg"
                matches = list(sample_images_dir.glob(pattern))
                if matches:
                    sample_image = matches[0]
                    logger.info(f"Found sample image for {instrument_id}: {sample_image}")
            
            # If not found and we have PHENOCAMS_DATA_DIR, search there
            if not sample_image and phenocams_data_dir:
                data_dir = Path(phenocams_data_dir)
                # Search in backup directory structure
                search_paths = [
                    data_dir / station_name / 'phenocams' / 'backup' / instrument_id / 'sftp' / 'mirror',
                    data_dir / station_name / 'phenocams' / 'backup' / instrument_id,
                    data_dir / station_name / 'phenocams',
                ]
                
                for search_path in search_paths:
                    if search_path.exists():
                        # Find any jpg file, preferably from 2024
                        jpg_files = list(search_path.rglob("*2024*.jpg"))
                        if not jpg_files:
                            jpg_files = list(search_path.rglob("*.jpg"))
                        
                        if jpg_files:
                            # Pick a midday image if possible
                            for jpg_file in jpg_files:
                                if any(time in str(jpg_file) for time in ['12-00', '1200', '13-00', '1300', '14-00', '1400']):
                                    sample_image = jpg_file
                                    break
                            
                            if not sample_image:
                                sample_image = jpg_files[0]
                                
                            logger.info(f"Found sample image for {instrument_id}: {sample_image}")
                            break
            
            # Update configuration IN PLACE (pass existing config)
            config = add_roi_00_to_station_config_in_place(
                config,
                station_name,
                instrument_id,
                sample_image,
                force=force
            )
            updated = True
            
        except Exception as e:
            logger.error(f"Error updating {station_name} - {instrument_id}: {e}")
            continue
    
    if updated:
        # Save updated configuration
        with open(station_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated stations.yaml with ROI_00 definitions")


if __name__ == "__main__":
    # Test ROI_00 calculation
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided image
        image_path = Path(sys.argv[1])
        if image_path.exists():
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Test different methods
            for method in ['gradient', 'color', 'fixed']:
                print(f"\nTesting {method} method:")
                points = calculate_roi_00(image, method)
                print(f"ROI_00 points: {points}")
                
                # Visualize
                vis_image = image.copy()
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 3)
                
                output_path = image_path.parent / f"roi_00_{method}_{image_path.name}"
                cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                print(f"Saved visualization to {output_path}")
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Usage: python roi_calculator.py <image_path>")