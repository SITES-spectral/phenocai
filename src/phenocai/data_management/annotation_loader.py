"""
Annotation Loading Module

Handles loading of both daily and individual annotation YAML files.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

import yaml
import numpy as np

from ..config.setup import config
from ..utils import parse_image_filename


logger = logging.getLogger(__name__)


@dataclass
class ROIAnnotation:
    """Single ROI annotation data."""
    roi_name: str
    discard: bool = False
    snow_presence: bool = False
    flags: List[str] = field(default_factory=list)
    not_needed: bool = False
    
    @property
    def is_annotated(self) -> bool:
        """Check if ROI has any annotation."""
        return (
            self.discard or 
            self.snow_presence or 
            len(self.flags) > 0 or 
            self.not_needed
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'roi_name': self.roi_name,
            'discard': self.discard,
            'snow_presence': self.snow_presence,
            'flags': self.flags.copy(),
            'not_needed': self.not_needed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROIAnnotation':
        """Create from dictionary."""
        # Handle legacy format where not_needed might be in flags
        flags = data.get('flags', [])
        not_needed = data.get('not_needed', False)
        
        # Check for legacy not_needed flag
        if 'not_needed' in flags and not not_needed:
            flags = [f for f in flags if f != 'not_needed']
            not_needed = True
        
        return cls(
            roi_name=data['roi_name'],
            discard=data.get('discard', False),
            snow_presence=data.get('snow_presence', False),
            flags=flags,
            not_needed=not_needed
        )


@dataclass
class AnnotationData:
    """Complete annotation data for an image."""
    filename: str
    station: str
    instrument: str
    year: int
    day_of_year: int
    annotations: List[ROIAnnotation]
    created: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    annotation_time_minutes: float = 0.0
    status: str = 'pending'
    
    @property
    def is_complete(self) -> bool:
        """Check if all ROIs are annotated."""
        return all(roi.is_annotated for roi in self.annotations)
    
    @property
    def annotation_count(self) -> int:
        """Count of annotated ROIs."""
        return sum(1 for roi in self.annotations if roi.is_annotated)
    
    def get_roi_annotation(self, roi_name: str) -> Optional[ROIAnnotation]:
        """Get annotation for specific ROI."""
        for roi in self.annotations:
            if roi.roi_name == roi_name:
                return roi
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for YAML saving."""
        return {
            'filename': self.filename,
            'station': self.station,
            'instrument': self.instrument,
            'year': str(self.year),
            'day_of_year': f"{self.day_of_year:03d}",
            'annotations': [roi.to_dict() for roi in self.annotations],
            'created': self.created.isoformat() if self.created else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'annotation_time_minutes': self.annotation_time_minutes,
            'status': self.status
        }


def load_individual_annotation(file_path: Union[str, Path]) -> Optional[AnnotationData]:
    """
    Load individual annotation file (new format).
    
    Format: station_instrument_year_doy_date_time_annotations.yaml
    
    Args:
        file_path: Path to annotation YAML file
        
    Returns:
        AnnotationData object or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Annotation file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract metadata
        filename = data.get('filename', file_path.stem.replace('_annotations', '.jpg'))
        
        # Parse dates
        created = None
        if 'created' in data:
            try:
                created = datetime.fromisoformat(data['created'])
            except:
                pass
        
        last_modified = None
        if 'last_modified' in data:
            try:
                last_modified = datetime.fromisoformat(data['last_modified'])
            except:
                pass
        
        # Load annotations
        annotations = []
        for ann_data in data.get('annotations', []):
            annotations.append(ROIAnnotation.from_dict(ann_data))
        
        # Create AnnotationData object
        annotation = AnnotationData(
            filename=filename,
            station=data.get('station', ''),
            instrument=data.get('instrument', ''),
            year=int(data.get('year', 0)),
            day_of_year=int(data.get('day_of_year', 0)),
            annotations=annotations,
            created=created,
            last_modified=last_modified,
            annotation_time_minutes=float(data.get('annotation_time_minutes', 0)),
            status=data.get('status', 'pending')
        )
        
        return annotation
        
    except Exception as e:
        logger.error(f"Error loading annotation file {file_path}: {e}")
        return None


def load_daily_annotations(file_path: Union[str, Path]) -> Dict[str, List[ROIAnnotation]]:
    """
    Load daily annotation file (legacy format).
    
    Format: annotations_doy.yaml containing multiple images
    
    Args:
        file_path: Path to daily annotation YAML file
        
    Returns:
        Dictionary mapping image filenames to lists of ROI annotations
    """
    file_path = Path(file_path)
    annotations_by_image = {}
    
    if not file_path.exists():
        logger.warning(f"Daily annotation file not found: {file_path}")
        return annotations_by_image
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract annotations section
        annotations_data = data.get('annotations', {})
        
        for image_filename, roi_list in annotations_data.items():
            if not isinstance(roi_list, list):
                continue
            
            roi_annotations = []
            for roi_data in roi_list:
                roi_annotations.append(ROIAnnotation.from_dict(roi_data))
            
            annotations_by_image[image_filename] = roi_annotations
        
        logger.info(f"Loaded annotations for {len(annotations_by_image)} images from {file_path.name}")
        
    except Exception as e:
        logger.error(f"Error loading daily annotation file {file_path}: {e}")
    
    return annotations_by_image


def scan_annotation_directory(
    directory: Union[str, Path],
    pattern: str = "*_annotations.yaml"
) -> List[Path]:
    """
    Scan directory for annotation files.
    
    Args:
        directory: Directory to scan
        pattern: File pattern to match
        
    Returns:
        List of annotation file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Annotation directory not found: {directory}")
        return []
    
    annotation_files = sorted(directory.glob(pattern))
    
    # Also check for daily annotation files
    daily_files = sorted(directory.glob("annotations_???.yaml"))
    
    all_files = annotation_files + daily_files
    
    logger.info(f"Found {len(all_files)} annotation files in {directory}")
    
    return all_files


def merge_annotations(
    annotations_list: List[AnnotationData],
    by_day: bool = True
) -> Dict[str, List[AnnotationData]]:
    """
    Merge annotations by day or other criteria.
    
    Args:
        annotations_list: List of AnnotationData objects
        by_day: If True, group by day_of_year
        
    Returns:
        Dictionary grouping annotations
    """
    grouped = {}
    
    for annotation in annotations_list:
        if by_day:
            key = f"{annotation.year}_{annotation.day_of_year:03d}"
        else:
            key = annotation.station
        
        if key not in grouped:
            grouped[key] = []
        
        grouped[key].append(annotation)
    
    return grouped


def validate_annotations(
    annotation: AnnotationData,
    expected_rois: List[str]
) -> List[str]:
    """
    Validate annotation data against expected ROIs.
    
    Args:
        annotation: AnnotationData to validate
        expected_rois: List of expected ROI names
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check for missing ROIs
    annotated_rois = {roi.roi_name for roi in annotation.annotations}
    missing_rois = set(expected_rois) - annotated_rois
    
    if missing_rois:
        warnings.append(f"Missing ROIs: {', '.join(sorted(missing_rois))}")
    
    # Check for unexpected ROIs
    unexpected_rois = annotated_rois - set(expected_rois)
    if unexpected_rois:
        warnings.append(f"Unexpected ROIs: {', '.join(sorted(unexpected_rois))}")
    
    # Check for unannotated ROIs
    unannotated = [roi.roi_name for roi in annotation.annotations if not roi.is_annotated]
    if unannotated:
        warnings.append(f"Unannotated ROIs: {', '.join(unannotated)}")
    
    return warnings


if __name__ == "__main__":
    # Test annotation loading
    print("Testing annotation loading...")
    
    # Test with sample annotation directory
    test_dir = config.master_annotation_pool_dir
    
    if test_dir.exists():
        files = scan_annotation_directory(test_dir)
        print(f"\nFound {len(files)} annotation files")
        
        # Try loading first individual annotation
        individual_files = [f for f in files if '_annotations.yaml' in f.name]
        if individual_files:
            print(f"\nLoading individual annotation: {individual_files[0].name}")
            annotation = load_individual_annotation(individual_files[0])
            if annotation:
                print(f"  Station: {annotation.station}")
                print(f"  Day: {annotation.day_of_year}")
                print(f"  ROIs: {len(annotation.annotations)}")
                print(f"  Complete: {annotation.is_complete}")
        
        # Try loading daily annotation
        daily_files = [f for f in files if f.name.startswith('annotations_') and not f.name.endswith('_annotations.yaml')]
        if daily_files:
            print(f"\nLoading daily annotation: {daily_files[0].name}")
            daily_anns = load_daily_annotations(daily_files[0])
            print(f"  Images: {len(daily_anns)}")
            if daily_anns:
                first_image = list(daily_anns.keys())[0]
                print(f"  First image: {first_image}")
                print(f"  ROIs: {len(daily_anns[first_image])}")