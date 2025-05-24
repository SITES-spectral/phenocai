"""
Multi-Station Dataset Builder

Handles creating datasets that combine data from multiple stations,
particularly Lönnstorp and Röbäcksdalen.
"""
import logging
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd

from ..config.setup import config
from ..config.station_configs import get_primary_stations, get_station_config
from .dataset_builder import create_master_annotation_dataframe, add_train_test_split, DatasetStats


logger = logging.getLogger(__name__)


def create_multi_station_dataset(
    stations: List[str] = None,
    base_data_dir: Union[str, Path] = None,
    output_path: Optional[Union[str, Path]] = None,
    include_unannotated: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.1,
    balance_stations: bool = True,
    roi_filter: Optional[List[str]] = None,
    years: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, DatasetStats]]:
    """
    Create a combined dataset from multiple stations.
    
    Args:
        stations: List of station names (defaults to primary stations)
        base_data_dir: Base directory containing station data
        output_path: Optional path to save combined dataset
        include_unannotated: Whether to include unannotated ROIs
        test_size: Fraction for test set
        val_size: Fraction for validation set
        balance_stations: Whether to balance samples across stations
        roi_filter: List of ROI names to include (e.g., ['ROI_00'])
        years: List of years to include (defaults to current year)
        
    Returns:
        Tuple of (combined dataframe, dict of stats per station)
    """
    if stations is None:
        stations = get_primary_stations()
    
    if base_data_dir is None:
        # Use parent of current data directory
        base_data_dir = config.data_dir.parent
    else:
        base_data_dir = Path(base_data_dir)
    
    logger.info(f"Creating multi-station dataset for: {', '.join(stations)}")
    
    # Collect dataframes and stats for each station
    station_dfs = []
    station_stats = {}
    
    for station in stations:
        logger.info(f"\nProcessing {station}...")
        
        # Get station configuration
        station_config = get_station_config(station)
        
        # Determine annotation directory
        annotation_dir = base_data_dir / station / 'master_annotation_pool'
        
        if not annotation_dir.exists():
            logger.warning(f"Annotation directory not found for {station}: {annotation_dir}")
            continue
        
        try:
            # Create dataset for this station
            df, stats = create_master_annotation_dataframe(
                annotation_dir,
                include_unannotated=include_unannotated
            )
            
            # Add station column
            df['station_full_name'] = station_config['full_name']
            df['station_code'] = station_config['station_code']
            
            # Apply ROI filter if specified
            if roi_filter:
                initial_count = len(df)
                df = df[df['roi_name'].isin(roi_filter)]
                logger.info(f"Filtered to ROIs {roi_filter}: {initial_count} → {len(df)} records")
            
            # Apply year filter if specified
            if years:
                initial_count = len(df)
                df = df[df['year'].astype(str).isin([str(y) for y in years])]
                logger.info(f"Filtered to years {years}: {initial_count} → {len(df)} records")
            
            if len(df) > 0:
                station_dfs.append(df)
                station_stats[station] = stats
                logger.info(f"Loaded {len(df)} records from {station}")
            else:
                logger.warning(f"No records remaining after filters for {station}")
            
        except Exception as e:
            logger.error(f"Error processing {station}: {e}")
            continue
    
    if not station_dfs:
        raise ValueError("No data loaded from any station")
    
    # Combine dataframes
    combined_df = pd.concat(station_dfs, ignore_index=True)
    
    logger.info(f"\nCombined dataset: {len(combined_df)} total records")
    
    # Balance stations if requested
    if balance_stations and len(stations) > 1:
        combined_df = balance_station_samples(combined_df)
        logger.info(f"After balancing: {len(combined_df)} records")
    
    # Add train/test/val splits
    # Stratify by station to ensure each split has data from all stations
    combined_df = add_train_test_split(
        combined_df,
        test_size=test_size,
        val_size=val_size,
        stratify_by='station',
        group_by_day=True
    )
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            combined_df.to_parquet(output_path, index=False)
        else:
            combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved combined dataset to {output_path}")
    
    # Print summary statistics
    print_multi_station_summary(combined_df, station_stats)
    
    return combined_df, station_stats


def balance_station_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance samples across stations by undersampling.
    
    Args:
        df: Input dataframe with 'station' column
        
    Returns:
        Balanced dataframe
    """
    # Count samples per station
    station_counts = df['station'].value_counts()
    min_count = station_counts.min()
    
    logger.info(f"Balancing stations to {min_count} samples each")
    
    # Sample from each station
    balanced_dfs = []
    
    for station in station_counts.index:
        station_df = df[df['station'] == station]
        
        if len(station_df) > min_count:
            # Sample by day groups to keep days together
            day_groups = station_df.groupby(['year', 'day_of_year']).size()
            
            # Randomly sample days until we reach target count
            sampled_days = []
            current_count = 0
            
            for day, count in day_groups.sample(frac=1).items():
                if current_count + count <= min_count:
                    sampled_days.append(day)
                    current_count += count
                
                if current_count >= min_count:
                    break
            
            # Filter to sampled days
            mask = station_df.set_index(['year', 'day_of_year']).index.isin(sampled_days)
            station_df = station_df[mask.values]
        
        balanced_dfs.append(station_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)


def print_multi_station_summary(df: pd.DataFrame, station_stats: Dict[str, DatasetStats]):
    """Print summary of multi-station dataset."""
    print("\n" + "="*60)
    print("MULTI-STATION DATASET SUMMARY")
    print("="*60)
    
    # Overall statistics
    print(f"\nTotal records: {len(df)}")
    print(f"Total images: {df['image_id'].nunique()}")
    print(f"Stations: {', '.join(df['station'].unique())}")
    
    # Per-station breakdown
    print("\nPer-station breakdown:")
    for station in df['station'].unique():
        station_df = df[df['station'] == station]
        print(f"\n{station}:")
        print(f"  Records: {len(station_df)}")
        print(f"  Images: {station_df['image_id'].nunique()}")
        print(f"  Days: {station_df['day_of_year'].nunique()}")
        
        if station in station_stats:
            stats = station_stats[station]
            print(f"  Snow present: {stats.snow_present_count}")
            print(f"  Discarded: {stats.discard_count}")
    
    # Split distribution
    print("\nSplit distribution:")
    split_counts = df.groupby(['split', 'station']).size().unstack(fill_value=0)
    print(split_counts)
    
    # Label distribution
    print("\nLabel distribution by station:")
    label_dist = df.groupby(['station', 'snow_presence']).size().unstack(fill_value=0)
    print(label_dist)
    
    # ROI distribution
    print("\nROI distribution:")
    roi_counts = df['roi_name'].value_counts()
    for roi, count in roi_counts.items():
        print(f"  {roi}: {count}")


def load_multi_station_dataset(
    dataset_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Load a previously created multi-station dataset.
    
    Args:
        dataset_path: Path to dataset file (CSV or Parquet)
        
    Returns:
        Loaded dataframe
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    logger.info(f"Loaded dataset with {len(df)} records from {df['station'].nunique()} stations")
    
    return df


def filter_dataset_by_criteria(
    df: pd.DataFrame,
    stations: List[str] = None,
    rois: List[str] = None,
    min_year: int = None,
    max_year: int = None,
    exclude_flags: List[str] = None,
    annotated_only: bool = True
) -> pd.DataFrame:
    """
    Filter dataset by various criteria.
    
    Args:
        df: Input dataframe
        stations: List of stations to include
        rois: List of ROI names to include
        min_year: Minimum year
        max_year: Maximum year
        exclude_flags: Flags to exclude
        annotated_only: Include only annotated samples
        
    Returns:
        Filtered dataframe
    """
    filtered = df.copy()
    
    if stations:
        filtered = filtered[filtered['station'].isin(stations)]
    
    if rois:
        filtered = filtered[filtered['roi_name'].isin(rois)]
    
    if min_year:
        filtered = filtered[filtered['year'] >= min_year]
    
    if max_year:
        filtered = filtered[filtered['year'] <= max_year]
    
    if annotated_only:
        filtered = filtered[filtered['is_annotated'] == True]
    
    if exclude_flags:
        # Exclude rows containing any of the specified flags
        for flag in exclude_flags:
            mask = ~filtered['flags'].str.contains(flag, na=False)
            filtered = filtered[mask]
    
    logger.info(f"Filtered dataset: {len(filtered)} records (from {len(df)})")
    
    return filtered


if __name__ == "__main__":
    # Test multi-station dataset creation
    print("Creating multi-station dataset for Lönnstorp and Röbäcksdalen...")
    
    try:
        # Create combined dataset
        output_path = config.experimental_data_dir / 'multi_station_dataset.csv'
        
        df, stats = create_multi_station_dataset(
            stations=['lonnstorp', 'robacksdalen'],
            output_path=output_path,
            include_unannotated=False,
            balance_stations=False  # Keep all data for now
        )
        
        print(f"\nDataset saved to: {output_path}")
        
        # Test filtering
        print("\n\nTesting dataset filtering...")
        filtered = filter_dataset_by_criteria(
            df,
            rois=['ROI_00', 'ROI_01'],
            min_year=2024
        )
        print(f"Filtered to ROI_00/01 and year 2024: {len(filtered)} records")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()