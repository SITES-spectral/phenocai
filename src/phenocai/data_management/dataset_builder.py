"""
Dataset Builder Module

Creates master annotation dataframes and handles train/test splitting.
"""
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ..config.setup import config
from ..utils import parse_image_filename
from .annotation_loader import (
    load_individual_annotation,
    load_daily_annotations,
    scan_annotation_directory,
    AnnotationData,
    ROIAnnotation
)


logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about the created dataset."""
    total_images: int
    total_rois: int
    annotated_rois: int
    snow_present_count: int
    discard_count: int
    flag_counts: Dict[str, int]
    roi_distribution: Dict[str, int]
    train_size: int = 0
    test_size: int = 0
    val_size: int = 0
    
    def print_summary(self):
        """Print dataset statistics summary."""
        print("\n=== Dataset Statistics ===")
        print(f"Total images: {self.total_images}")
        print(f"Total ROIs: {self.total_rois}")
        print(f"Annotated ROIs: {self.annotated_rois} ({self.annotated_rois/self.total_rois*100:.1f}%)")
        print(f"Snow present: {self.snow_present_count}")
        print(f"Discarded: {self.discard_count}")
        
        if self.flag_counts:
            print("\nFlag distribution:")
            for flag, count in sorted(self.flag_counts.items()):
                print(f"  {flag}: {count}")
        
        print("\nROI distribution:")
        for roi, count in sorted(self.roi_distribution.items()):
            print(f"  {roi}: {count}")
        
        if self.train_size > 0:
            print(f"\nSplit sizes:")
            print(f"  Train: {self.train_size}")
            print(f"  Test: {self.test_size}")
            if self.val_size > 0:
                print(f"  Val: {self.val_size}")


def create_master_annotation_dataframe(
    annotation_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    include_unannotated: bool = False,
    batch_size: int = 100
) -> Tuple[pd.DataFrame, DatasetStats]:
    """
    Create master dataframe from annotation files.
    
    Processes annotation files in batches for memory efficiency.
    
    Args:
        annotation_dir: Directory containing annotation files
        output_path: Optional path to save the dataframe
        include_unannotated: Whether to include unannotated ROIs
        batch_size: Number of files to process at once
        
    Returns:
        Tuple of (dataframe, statistics)
    """
    annotation_dir = Path(annotation_dir)
    
    # Scan for annotation files
    annotation_files = scan_annotation_directory(annotation_dir)
    
    if not annotation_files:
        raise ValueError(f"No annotation files found in {annotation_dir}")
    
    logger.info(f"Processing {len(annotation_files)} annotation files...")
    
    # Process files in batches
    all_records = []
    stats = DatasetStats(
        total_images=0,
        total_rois=0,
        annotated_rois=0,
        snow_present_count=0,
        discard_count=0,
        flag_counts={},
        roi_distribution={}
    )
    
    for batch_start in range(0, len(annotation_files), batch_size):
        batch_end = min(batch_start + batch_size, len(annotation_files))
        batch_files = annotation_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(annotation_files) + batch_size - 1)//batch_size}")
        
        batch_records = process_annotation_batch(
            batch_files,
            stats,
            include_unannotated
        )
        
        all_records.extend(batch_records)
        
        # Force garbage collection after each batch
        if config.enable_garbage_collection:
            gc.collect()
    
    # Create dataframe
    df = pd.DataFrame(all_records)
    
    # Add additional columns
    df['annotation_id'] = df.apply(
        lambda row: f"{row['image_id']}_{row['roi_name']}", 
        axis=1
    )
    
    # Sort by image and ROI
    df = df.sort_values(['station', 'year', 'day_of_year', 'time_str', 'roi_name'])
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved master dataframe to {output_path}")
    
    # Update stats
    stats.total_images = df['image_id'].nunique()
    
    return df, stats


def process_annotation_batch(
    annotation_files: List[Path],
    stats: DatasetStats,
    include_unannotated: bool
) -> List[Dict[str, Any]]:
    """
    Process a batch of annotation files.
    
    Args:
        annotation_files: List of annotation file paths
        stats: DatasetStats object to update
        include_unannotated: Whether to include unannotated ROIs
        
    Returns:
        List of annotation records
    """
    records = []
    
    for file_path in annotation_files:
        try:
            if file_path.name.endswith('_annotations.yaml'):
                # Individual annotation file
                annotation = load_individual_annotation(file_path)
                if annotation:
                    records.extend(
                        process_single_annotation(annotation, stats, include_unannotated)
                    )
            
            elif file_path.name.startswith('annotations_'):
                # Daily annotation file
                daily_anns = load_daily_annotations(file_path)
                
                for image_filename, roi_list in daily_anns.items():
                    # Create AnnotationData from daily format
                    try:
                        metadata = parse_image_filename(image_filename)
                        annotation = AnnotationData(
                            filename=image_filename,
                            station=metadata.station,
                            instrument=metadata.instrument,
                            year=metadata.year,
                            day_of_year=metadata.day_of_year,
                            annotations=roi_list
                        )
                        
                        records.extend(
                            process_single_annotation(annotation, stats, include_unannotated)
                        )
                    except Exception as e:
                        logger.warning(f"Error processing {image_filename}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return records


def process_single_annotation(
    annotation: AnnotationData,
    stats: DatasetStats,
    include_unannotated: bool
) -> List[Dict[str, Any]]:
    """
    Process single annotation into records.
    
    Args:
        annotation: AnnotationData object
        stats: DatasetStats to update
        include_unannotated: Whether to include unannotated ROIs
        
    Returns:
        List of record dictionaries
    """
    records = []
    
    # Parse image metadata
    try:
        metadata = parse_image_filename(annotation.filename)
    except:
        # Use annotation metadata if parsing fails
        class MockMetadata:
            pass
        
        metadata = MockMetadata()
        metadata.station = annotation.station
        metadata.instrument = annotation.instrument
        metadata.year = annotation.year
        metadata.day_of_year = annotation.day_of_year
        metadata.date_str = ""
        metadata.time_str = ""
        metadata.image_id = f"{annotation.station}_{annotation.instrument}_{annotation.year}_{annotation.day_of_year:03d}"
    
    # Process each ROI
    for roi in annotation.annotations:
        if not include_unannotated and not roi.is_annotated:
            continue
        
        # Construct file path with day_of_year subdirectory
        day_dir = f"{metadata.day_of_year:03d}"
        file_path = config.image_base_dir / day_dir / annotation.filename
        
        record = {
            'image_filename': annotation.filename,
            'file_path': str(file_path),
            'image_id': metadata.image_id,
            'station': metadata.station,
            'instrument': metadata.instrument,
            'year': metadata.year,
            'day_of_year': metadata.day_of_year,
            'date_str': metadata.date_str,
            'time_str': metadata.time_str,
            'roi_name': roi.roi_name,
            'discard': roi.discard,
            'snow_presence': roi.snow_presence,
            'flags': ','.join(roi.flags) if roi.flags else '',
            'flag_count': len(roi.flags),
            'has_flags': len(roi.flags) > 0,  # New field to easily filter quality issues
            'not_needed': roi.not_needed,
            'is_annotated': roi.is_annotated,
            'annotation_status': annotation.status,
            'annotation_time_minutes': annotation.annotation_time_minutes
        }
        
        records.append(record)
        
        # Update statistics
        stats.total_rois += 1
        
        if roi.is_annotated:
            stats.annotated_rois += 1
        
        if roi.snow_presence:
            stats.snow_present_count += 1
        
        if roi.discard:
            stats.discard_count += 1
        
        # Update flag counts
        for flag in roi.flags:
            stats.flag_counts[flag] = stats.flag_counts.get(flag, 0) + 1
        
        # Update ROI distribution
        stats.roi_distribution[roi.roi_name] = stats.roi_distribution.get(roi.roi_name, 0) + 1
    
    return records


def add_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_by: Optional[str] = 'station',
    random_state: int = 42,
    group_by_day: bool = True
) -> pd.DataFrame:
    """
    Add train/test/val split to dataframe.
    
    Ensures all images from the same day stay in the same split.
    
    Args:
        df: Input dataframe
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training set)
        stratify_by: Column to stratify by
        random_state: Random seed
        group_by_day: Keep all images from same day together
        
    Returns:
        DataFrame with added 'split' column
    """
    df = df.copy()
    
    if group_by_day:
        # Create day groups
        df['day_group'] = df['station'] + '_' + df['year'].astype(str) + '_' + df['day_of_year'].astype(str)
        
        # Get unique days
        unique_days = df['day_group'].unique()
        
        # Create stratification data if requested
        if stratify_by and stratify_by in df.columns:
            # Get most common value for each day group
            day_stratify = df.groupby('day_group')[stratify_by].agg(lambda x: x.mode()[0])
            stratify_values = day_stratify[unique_days].values
        else:
            stratify_values = None
        
        # Split days into train/test
        if stratify_values is not None:
            train_days, test_days = train_test_split(
                unique_days,
                test_size=test_size,
                stratify=stratify_values,
                random_state=random_state
            )
        else:
            train_days, test_days = train_test_split(
                unique_days,
                test_size=test_size,
                random_state=random_state
            )
        
        # Further split train into train/val if requested
        if val_size > 0:
            # Adjust val_size relative to training set
            val_size_adjusted = val_size / (1 - test_size)
            
            if stratify_values is not None:
                train_stratify = day_stratify[train_days].values
                train_days, val_days = train_test_split(
                    train_days,
                    test_size=val_size_adjusted,
                    stratify=train_stratify,
                    random_state=random_state
                )
            else:
                train_days, val_days = train_test_split(
                    train_days,
                    test_size=val_size_adjusted,
                    random_state=random_state
                )
        
        # Assign splits
        df['split'] = 'train'
        df.loc[df['day_group'].isin(test_days), 'split'] = 'test'
        if val_size > 0:
            df.loc[df['day_group'].isin(val_days), 'split'] = 'val'
        
        # Clean up temporary column
        df = df.drop('day_group', axis=1)
        
    else:
        # Simple random split
        if stratify_by and stratify_by in df.columns:
            stratify_values = df[stratify_by].values
        else:
            stratify_values = None
        
        # Create splits
        indices = np.arange(len(df))
        
        if stratify_values is not None:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=stratify_values,
                random_state=random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state
            )
        
        # Further split if validation requested
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            
            if stratify_values is not None:
                train_stratify = stratify_values[train_idx]
                train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=val_size_adjusted,
                    stratify=train_stratify,
                    random_state=random_state
                )
            else:
                train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=val_size_adjusted,
                    random_state=random_state
                )
        
        # Assign splits
        df['split'] = 'train'
        df.iloc[test_idx] = 'test'
        if val_size > 0:
            df.iloc[val_idx] = 'val'
    
    # Add split hash for reproducibility
    split_string = f"{test_size}_{val_size}_{stratify_by}_{random_state}_{group_by_day}"
    df['split_hash'] = hashlib.md5(split_string.encode()).hexdigest()[:8]
    
    logger.info(f"Split sizes - Train: {len(df[df['split']=='train'])}, "
                f"Test: {len(df[df['split']=='test'])}, "
                f"Val: {len(df[df['split']=='val']) if val_size > 0 else 0}")
    
    return df


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset builder...")
    
    # Create master dataframe
    try:
        df, stats = create_master_annotation_dataframe(
            config.master_annotation_pool_dir,
            include_unannotated=False
        )
        
        print(f"\nCreated dataframe with {len(df)} records")
        stats.print_summary()
        
        # Add splits
        if len(df) > 10:
            df_split = add_train_test_split(df)
            print("\nSplit distribution:")
            print(df_split['split'].value_counts())
            
            # Save with splits
            output_path = config.master_df_with_splits_path
            df_split.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()