"""
Core prediction/inference functionality for PhenoCAI.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
import yaml
import json
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

from ..config.setup import config
from ..utils import (
    load_image,
    parse_image_filename,
    extract_roi_sub_image,
    load_roi_config_from_yaml,
    get_roi_points_from_config
)
from ..heuristics import (
    detect_snow_hsv,
    should_discard_roi
)
from ..heuristics.image_quality import detect_image_quality_issues
from ..data_management.annotation_loader import AnnotationData, ROIAnnotation


logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a single prediction."""
    filename: str
    roi_name: str
    snow_probability: float
    snow_presence: bool
    confidence: float
    quality_flags: List[str] = field(default_factory=list)
    discard: bool = False
    has_flags: bool = False
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filename': self.filename,
            'roi_name': self.roi_name,
            'snow_probability': float(self.snow_probability),
            'snow_presence': self.snow_presence,
            'confidence': float(self.confidence),
            'quality_flags': self.quality_flags,
            'discard': self.discard,
            'has_flags': self.has_flags,
            'flag_count': len(self.quality_flags),
            'processing_time': self.processing_time
        }


class ModelPredictor:
    """Handles predictions using trained models."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        threshold: float = 0.5,
        batch_size: int = 32,
        use_heuristics: bool = True,
        roi_config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            threshold: Probability threshold for binary classification
            batch_size: Batch size for processing
            use_heuristics: Whether to use heuristics as fallback
            roi_config_path: Path to ROI configuration file
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_heuristics = use_heuristics
        
        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))
        
        # Get input shape from model
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        logger.info(f"Model expects input shape: {self.input_shape}")
        
        # Load ROI configuration
        if roi_config_path is None:
            roi_config_path = config.roi_config_file_path
        self.roi_config = load_roi_config_from_yaml(roi_config_path)
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Resize to model input shape
        img_pil = Image.fromarray(image)
        img_pil = img_pil.resize(self.input_shape, Image.Resampling.LANCZOS)
        img_array = np.array(img_pil)
        
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        # Apply same preprocessing as training
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def predict_single_roi(
        self,
        roi_image: np.ndarray,
        apply_heuristics: bool = True
    ) -> Tuple[float, List[str], bool]:
        """
        Predict snow presence for a single ROI.
        
        Args:
            roi_image: ROI image as numpy array
            apply_heuristics: Whether to check quality first
            
        Returns:
            Tuple of (snow_probability, quality_flags, should_discard)
        """
        quality_flags = []
        should_discard = False
        
        # Check image quality if requested
        if apply_heuristics:
            quality_flags = detect_image_quality_issues(roi_image)
            
            # Check if ROI should be discarded based on quality
            discard_reasons = {'unusable', 'too_dark', 'too_bright', 'blur', 'lens_obstruction'}
            if any(flag in discard_reasons for flag in quality_flags):
                should_discard = True
            
            # If image has severe quality issues, use heuristics
            if any(flag in ['unusable', 'too_dark', 'too_bright'] for flag in quality_flags):
                if 'snow' in quality_flags or 'lens_snow' in quality_flags:
                    return 1.0, quality_flags, should_discard
                else:
                    return 0.0, quality_flags, should_discard
        
        # Preprocess and predict
        img_preprocessed = self.preprocess_image(roi_image)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Get prediction
        snow_probability = float(self.model.predict(img_batch, verbose=0)[0, 0])
        
        # Apply heuristic override if confident
        if self.use_heuristics and apply_heuristics:
            heuristic_snow = detect_snow_hsv(roi_image)
            if heuristic_snow is not None:
                # If heuristics are very confident, override model
                if heuristic_snow and snow_probability < 0.3:
                    snow_probability = 0.8  # Boost probability
                elif not heuristic_snow and snow_probability > 0.7:
                    snow_probability = 0.2  # Reduce probability
        
        return snow_probability, quality_flags, should_discard
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        station: Optional[str] = None,
        instrument: Optional[str] = None
    ) -> List[PredictionResult]:
        """
        Generate predictions for all ROIs in an image.
        
        Args:
            image_path: Path to image file
            station: Station name (auto-detected if None)
            instrument: Instrument name (auto-detected if None)
            
        Returns:
            List of predictions for each ROI
        """
        image_path = Path(image_path)
        start_time = datetime.now()
        
        # Parse filename
        try:
            metadata = parse_image_filename(image_path.name)
            if station is None:
                station = metadata.station
            if instrument is None:
                instrument = metadata.instrument
        except Exception as e:
            logger.warning(f"Failed to parse filename {image_path.name}: {e}")
            if station is None or instrument is None:
                raise ValueError("Could not determine station/instrument from filename")
        
        # Load image
        image = load_image(str(image_path), as_rgb=True)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []
        
        # Get ROIs for this station/instrument
        roi_names = self.roi_config.get(station, {}).get(instrument, {}).keys()
        
        results = []
        for roi_name in roi_names:
            try:
                # Get ROI points
                roi_points = get_roi_points_from_config(
                    self.roi_config, station, instrument, roi_name
                )
                
                if roi_points is None:
                    logger.warning(f"No ROI points found for {roi_name}")
                    continue
                
                # Extract ROI
                roi_image = extract_roi_sub_image(image, roi_points)
                
                # Predict
                snow_prob, quality_flags, should_discard = self.predict_single_roi(roi_image)
                
                # Create result
                result = PredictionResult(
                    filename=image_path.name,
                    roi_name=roi_name,
                    snow_probability=snow_prob,
                    snow_presence=snow_prob >= self.threshold,
                    confidence=abs(snow_prob - 0.5) * 2,  # Convert to confidence
                    quality_flags=quality_flags,
                    discard=should_discard,
                    has_flags=len(quality_flags) > 0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing ROI {roi_name}: {e}")
                continue
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> Dict[str, List[PredictionResult]]:
        """
        Generate predictions for multiple images.
        
        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping image paths to prediction results
        """
        results = {}
        
        iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
        
        for image_path in iterator:
            try:
                predictions = self.predict_image(image_path)
                results[str(image_path)] = predictions
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[str(image_path)] = []
        
        return results


class BatchPredictor:
    """Handles batch prediction for large datasets."""
    
    def __init__(self, predictor: ModelPredictor):
        """
        Initialize batch predictor.
        
        Args:
            predictor: ModelPredictor instance
        """
        self.predictor = predictor
    
    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        recursive: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        output_format: str = 'yaml'
    ) -> Dict[str, Any]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            pattern: File pattern to match
            recursive: Whether to search recursively
            output_dir: Directory to save results
            output_format: Output format (yaml, json, csv)
            
        Returns:
            Summary statistics
        """
        directory = Path(directory)
        
        # Find all images
        if recursive:
            image_files = sorted(directory.rglob(pattern))
        else:
            image_files = sorted(directory.glob(pattern))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process in batches
        all_results = {}
        for i in range(0, len(image_files), self.predictor.batch_size):
            batch = image_files[i:i + self.predictor.batch_size]
            batch_results = self.predictor.predict_batch(batch)
            all_results.update(batch_results)
        
        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if output_format == 'yaml':
                self._save_as_yaml(all_results, output_dir)
            elif output_format == 'json':
                self._save_as_json(all_results, output_dir)
            elif output_format == 'csv':
                self._save_as_csv(all_results, output_dir)
        
        # Calculate summary statistics
        total_predictions = sum(len(preds) for preds in all_results.values())
        snow_count = sum(
            1 for preds in all_results.values()
            for pred in preds if pred.snow_presence
        )
        
        return {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'total_predictions': total_predictions,
            'snow_predictions': snow_count,
            'snow_percentage': snow_count / total_predictions * 100 if total_predictions > 0 else 0
        }
    
    def process_date_range(
        self,
        start_day: int,
        end_day: int,
        year: Optional[int] = None,
        station: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process images for a range of days.
        
        Args:
            start_day: Start day of year (1-365)
            end_day: End day of year (1-365)
            year: Year (uses config default if None)
            station: Station (uses config default if None)
            output_dir: Output directory for results
            
        Returns:
            Summary statistics
        """
        if year is None:
            year = int(config.current_year)
        if station is None:
            station = config.current_station
        
        # Get base image directory
        image_base = config.image_base_dir
        
        all_stats = {
            'total_days': 0,
            'total_images': 0,
            'total_predictions': 0,
            'snow_predictions': 0,
            'daily_stats': {}
        }
        
        for day in range(start_day, end_day + 1):
            day_str = f"{day:03d}"
            logger.info(f"Processing day {day_str}")
            
            # Find images for this day
            day_pattern = f"*_{year}_{day_str}_*.jpg"
            day_images = sorted(image_base.glob(day_pattern))
            
            if not day_images:
                logger.warning(f"No images found for day {day_str}")
                continue
            
            # Process this day's images
            day_results = self.predictor.predict_batch(day_images)
            
            # Save daily results
            if output_dir:
                output_dir = Path(output_dir)
                day_output_dir = output_dir / f"day_{day_str}"
                day_output_dir.mkdir(parents=True, exist_ok=True)
                self._save_as_yaml(day_results, day_output_dir)
            
            # Update statistics
            day_predictions = sum(len(preds) for preds in day_results.values())
            day_snow = sum(
                1 for preds in day_results.values()
                for pred in preds if pred.snow_presence
            )
            
            all_stats['total_days'] += 1
            all_stats['total_images'] += len(day_images)
            all_stats['total_predictions'] += day_predictions
            all_stats['snow_predictions'] += day_snow
            all_stats['daily_stats'][day_str] = {
                'images': len(day_images),
                'predictions': day_predictions,
                'snow_count': day_snow,
                'snow_percentage': day_snow / day_predictions * 100 if day_predictions > 0 else 0
            }
        
        return all_stats
    
    def _save_as_yaml(self, results: Dict[str, List[PredictionResult]], output_dir: Path):
        """Save results as YAML files matching manual annotation format."""
        for image_path, predictions in results.items():
            image_name = Path(image_path).stem
            
            # Parse metadata from filename
            try:
                from ..utils import parse_image_filename
                metadata = parse_image_filename(Path(image_path).name)
                station = metadata.station
                instrument = metadata.instrument
                year = metadata.year
                day_of_year = metadata.day_of_year
            except:
                # Fallback values
                station = self.predictor.station if hasattr(self.predictor, 'station') else ''
                instrument = self.predictor.instrument if hasattr(self.predictor, 'instrument') else ''
                year = ''
                day_of_year = ''
            
            # Create annotation data structure matching manual format
            annotation_data = {
                'created': datetime.now().isoformat(),
                'model_path': str(self.predictor.model_path),
                'threshold': self.predictor.threshold,
                'filename': Path(image_path).name,
                'station': station,
                'instrument': instrument,
                'year': str(year),
                'day_of_year': str(day_of_year),
                'status': 'completed',
                'annotation_time_minutes': 0.0,  # Model predictions are instant
                'annotations': [
                    {
                        'roi_name': pred.roi_name,
                        'discard': pred.discard,
                        'snow_presence': pred.snow_presence,
                        'flags': pred.quality_flags,
                        'not_needed': False,
                        'snow_probability': float(pred.snow_probability),
                        'confidence': float(pred.confidence),
                        'model_predicted': True
                    }
                    for pred in predictions
                ]
            }
            
            # Save to file
            output_file = output_dir / f"{image_name}_predictions.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(annotation_data, f, default_flow_style=False)
    
    def _save_as_json(self, results: Dict[str, List[PredictionResult]], output_dir: Path):
        """Save results as JSON."""
        # Convert to serializable format
        json_data = {
            image_path: [pred.to_dict() for pred in predictions]
            for image_path, predictions in results.items()
        }
        
        output_file = output_dir / 'predictions.json'
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_as_csv(self, results: Dict[str, List[PredictionResult]], output_dir: Path):
        """Save results as CSV matching dataset format."""
        rows = []
        for image_path, predictions in results.items():
            # Parse metadata
            try:
                from ..utils import parse_image_filename
                metadata = parse_image_filename(Path(image_path).name)
            except:
                metadata = None
            
            for pred in predictions:
                row = {
                    'image_filename': pred.filename,
                    'file_path': image_path,
                    'image_id': metadata.image_id if metadata else Path(image_path).stem,
                    'station': metadata.station if metadata else '',
                    'instrument': metadata.instrument if metadata else '',
                    'year': metadata.year if metadata else '',
                    'day_of_year': metadata.day_of_year if metadata else '',
                    'roi_name': pred.roi_name,
                    'discard': pred.discard,
                    'snow_presence': pred.snow_presence,
                    'flags': ','.join(pred.quality_flags),
                    'flag_count': len(pred.quality_flags),
                    'has_flags': pred.has_flags,
                    'not_needed': False,
                    'snow_probability': pred.snow_probability,
                    'confidence': pred.confidence,
                    'model_predicted': True,
                    'processing_time': pred.processing_time
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        output_file = output_dir / 'predictions.csv'
        df.to_csv(output_file, index=False)


# Convenience functions
def process_single_image(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    threshold: float = 0.5,
    use_heuristics: bool = True
) -> List[PredictionResult]:
    """
    Quick function to process a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        threshold: Probability threshold
        use_heuristics: Whether to use heuristics
        
    Returns:
        List of predictions for each ROI
    """
    predictor = ModelPredictor(model_path, threshold=threshold, use_heuristics=use_heuristics)
    return predictor.predict_image(image_path)


def process_image_directory(
    model_path: Union[str, Path],
    directory: Union[str, Path],
    output_dir: Union[str, Path],
    threshold: float = 0.5,
    use_heuristics: bool = True,
    output_format: str = 'yaml'
) -> Dict[str, Any]:
    """
    Process all images in a directory.
    
    Args:
        model_path: Path to trained model
        directory: Directory containing images
        output_dir: Output directory
        threshold: Probability threshold
        use_heuristics: Whether to use heuristics
        output_format: Output format (yaml, json, csv)
        
    Returns:
        Summary statistics
    """
    predictor = ModelPredictor(model_path, threshold=threshold, use_heuristics=use_heuristics)
    batch_predictor = BatchPredictor(predictor)
    return batch_predictor.process_directory(directory, output_dir=output_dir, output_format=output_format)


def process_date_range(
    model_path: Union[str, Path],
    start_day: int,
    end_day: int,
    output_dir: Union[str, Path],
    year: Optional[int] = None,
    station: Optional[str] = None,
    threshold: float = 0.5,
    use_heuristics: bool = True
) -> Dict[str, Any]:
    """
    Process images for a date range.
    
    Args:
        model_path: Path to trained model
        start_day: Start day of year
        end_day: End day of year  
        output_dir: Output directory
        year: Year (uses config default if None)
        station: Station (uses config default if None)
        threshold: Probability threshold
        use_heuristics: Whether to use heuristics
        
    Returns:
        Summary statistics
    """
    predictor = ModelPredictor(model_path, threshold=threshold, use_heuristics=use_heuristics)
    batch_predictor = BatchPredictor(predictor)
    return batch_predictor.process_date_range(
        start_day, end_day, year=year, station=station, output_dir=output_dir
    )