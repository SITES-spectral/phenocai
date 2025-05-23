"""
PhenoCAI Utility Functions

This module provides core utility functions for image processing, ROI extraction,
and filename parsing for the PhenoCAI system.
"""
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2
import yaml
from PIL import Image

from .config.setup import config


logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata extracted from phenocam image filename."""
    station: str
    instrument: str
    year: int
    day_of_year: int
    date_str: str
    time_str: str
    full_datetime: datetime
    filename: str
    
    @property
    def image_id(self) -> str:
        """Generate unique image ID."""
        return f"{self.station}_{self.instrument}_{self.year}_{self.day_of_year:03d}_{self.time_str}"


def load_roi_config_from_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load ROI configuration from stations.yaml file.
    
    Args:
        yaml_path: Path to the stations.yaml configuration file
        
    Returns:
        Dictionary containing station configurations with ROI definitions
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"ROI configuration file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            roi_config = yaml.safe_load(f)
        
        logger.info(f"Loaded ROI configuration from {yaml_path}")
        
        # Validate basic structure
        if 'stations' not in roi_config:
            raise ValueError("ROI configuration missing 'stations' key")
        
        return roi_config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading ROI configuration: {e}")
        raise


def parse_image_filename(filename: str) -> ImageMetadata:
    """
    Parse phenocam image filename to extract metadata.
    
    Expected format: station_instrument_year_doy_date_time.jpg
    Example: lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg
    
    Args:
        filename: Image filename to parse
        
    Returns:
        ImageMetadata object with extracted information
        
    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Remove path and extension if present
    filename_only = Path(filename).stem
    
    # Pattern for phenocam filenames
    pattern = r'^([^_]+)_([^_]+_[^_]+_[^_]+_[^_]+)_(\d{4})_(\d{3})_(\d{8})_(\d{6})$'
    match = re.match(pattern, filename_only)
    
    if not match:
        raise ValueError(f"Filename '{filename}' doesn't match expected pattern")
    
    station, instrument, year_str, doy_str, date_str, time_str = match.groups()
    
    # Parse datetime
    year = int(year_str)
    day_of_year = int(doy_str)
    
    # Parse date and time
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    time_obj = datetime.strptime(time_str, '%H%M%S')
    
    # Combine date and time
    full_datetime = datetime.combine(date_obj.date(), time_obj.time())
    
    return ImageMetadata(
        station=station.lower(),
        instrument=instrument,
        year=year,
        day_of_year=day_of_year,
        date_str=date_str,
        time_str=time_str,
        full_datetime=full_datetime,
        filename=filename
    )


def get_roi_points_from_config(
    roi_config: Dict[str, Any],
    station: str,
    instrument: str,
    roi_name: str
) -> Optional[List[Tuple[int, int]]]:
    """
    Extract ROI points for a specific station, instrument, and ROI name.
    
    Args:
        roi_config: ROI configuration dictionary from YAML
        station: Station name (e.g., 'lonnstorp')
        instrument: Instrument ID (e.g., 'LON_AGR_PL01_PHE01')
        roi_name: ROI name (e.g., 'ROI_01')
        
    Returns:
        List of (x, y) tuples defining the ROI polygon, or None if not found
    """
    station = station.lower()
    
    try:
        # Navigate through the configuration structure
        stations = roi_config.get('stations', {})
        
        if station not in stations:
            logger.warning(f"Station '{station}' not found in ROI configuration")
            return None
        
        station_config = stations[station]
        instruments = station_config.get('instruments', {})
        
        if instrument not in instruments:
            logger.warning(f"Instrument '{instrument}' not found for station '{station}'")
            return None
        
        instrument_config = instruments[instrument]
        rois = instrument_config.get('ROIs', {})
        
        if roi_name not in rois:
            if config.debug_roi_lookup:
                logger.debug(f"ROI '{roi_name}' not found for {station}/{instrument}")
            return None
        
        roi_data = rois[roi_name]
        
        # Handle different ROI formats
        if roi_name == 'ROI_00':
            # ROI_00 is typically the full image
            return None
        
        # Extract points based on format in YAML
        if 'x' in roi_data and 'y' in roi_data:
            # Format: separate x and y arrays
            x_coords = roi_data['x']
            y_coords = roi_data['y']
            
            if len(x_coords) != len(y_coords):
                raise ValueError(f"Mismatched x/y coordinates for {roi_name}")
            
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
            
        elif 'points' in roi_data:
            # Format: list of [x, y] pairs
            points = [(int(p[0]), int(p[1])) for p in roi_data['points']]
            
        else:
            raise ValueError(f"Unknown ROI format for {roi_name}")
        
        return points
        
    except Exception as e:
        logger.error(f"Error extracting ROI points: {e}")
        return None


def extract_roi_sub_image(
    image: np.ndarray,
    roi_points: Optional[List[Tuple[int, int]]],
    roi_name: str = "ROI"
) -> np.ndarray:
    """
    Extract ROI sub-image from full image using polygon points.
    
    Implements memory-efficient extraction following best practices:
    - Pre-allocates arrays
    - Uses in-place operations where possible
    - Cleans up intermediate arrays
    
    Args:
        image: Input image as numpy array (H, W, C)
        roi_points: List of (x, y) points defining ROI polygon, or None for full image
        roi_name: Name of the ROI for logging
        
    Returns:
        Cropped ROI image as numpy array
    """
    if roi_points is None or roi_name == "ROI_00":
        # Return full image for ROI_00
        return image.copy()
    
    try:
        # Convert points to numpy array
        points = np.array(roi_points, dtype=np.int32)
        
        # Get bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add small padding to avoid edge effects
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        # Create mask for polygon (memory efficient)
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        
        # Adjust points to cropped coordinates
        adjusted_points = points - np.array([x_min, y_min])
        
        # Fill polygon
        cv2.fillPoly(mask, [adjusted_points], 255)
        
        # Extract cropped region
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Apply mask efficiently
        if len(cropped.shape) == 3:
            # Color image - expand mask to 3 channels
            mask_3d = np.expand_dims(mask, axis=2)
            result = np.where(mask_3d > 0, cropped, 0)
        else:
            # Grayscale image
            result = np.where(mask > 0, cropped, 0)
        
        # Clean up intermediate arrays
        del mask
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting ROI {roi_name}: {e}")
        # Return full image as fallback
        return image.copy()


def load_image(
    image_path: Union[str, Path],
    as_rgb: bool = True,
    max_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load image from file with proper error handling and optional resizing.
    
    Args:
        image_path: Path to image file
        as_rgb: If True, convert to RGB; if False, keep original format
        max_size: Optional (width, height) tuple to resize image
        
    Returns:
        Image as numpy array
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Use PIL for initial loading (handles more formats)
        with Image.open(image_path) as pil_image:
            
            # Convert to RGB if requested
            if as_rgb and pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize if requested
            if max_size is not None:
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image = np.array(pil_image)
            
        # Ensure we have a copy (not a view)
        image = image.copy()
        
        logger.debug(f"Loaded image {image_path.name}: shape={image.shape}, dtype={image.dtype}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise IOError(f"Failed to load image: {e}")


def resize_image_preserve_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image to target size while preserving aspect ratio.
    
    Args:
        image: Input image array
        target_size: Target (width, height)
        
    Returns:
        Resized image with padding if needed
    """
    target_width, target_height = target_size
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_height, target_width), dtype=image.dtype)
    
    # Calculate padding
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return padded


def calculate_image_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for an image.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with statistics (mean, std, min, max, etc.)
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate statistics
    stats = {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'median': float(np.median(gray)),
    }
    
    # Calculate additional metrics
    stats['contrast'] = stats['max'] - stats['min']
    
    # Calculate blur metric (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    stats['blur_metric'] = float(laplacian.var())
    
    return stats


def get_image_paths_for_day(
    base_dir: Union[str, Path],
    day_of_year: int,
    pattern: str = "*.jpg"
) -> List[Path]:
    """
    Get all image paths for a specific day of year.
    
    Args:
        base_dir: Base directory to search
        day_of_year: Day of year (1-365)
        pattern: File pattern to match
        
    Returns:
        List of image paths
    """
    base_dir = Path(base_dir)
    day_dir = base_dir / f"{day_of_year:03d}"
    
    if not day_dir.exists():
        logger.warning(f"Day directory not found: {day_dir}")
        return []
    
    image_paths = sorted(day_dir.glob(pattern))
    
    logger.info(f"Found {len(image_paths)} images for day {day_of_year}")
    
    return image_paths


if __name__ == "__main__":
    # Test utilities
    print("Testing PhenoCAI utilities...")
    
    # Test filename parsing
    test_filename = "lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg"
    try:
        metadata = parse_image_filename(test_filename)
        print(f"\nParsed metadata: {metadata}")
        print(f"Image ID: {metadata.image_id}")
    except Exception as e:
        print(f"Error parsing filename: {e}")
    
    # Test ROI config loading
    try:
        roi_config = load_roi_config_from_yaml(config.roi_config_file_path)
        print(f"\nLoaded ROI config with {len(roi_config.get('stations', {}))} stations")
    except Exception as e:
        print(f"Error loading ROI config: {e}")