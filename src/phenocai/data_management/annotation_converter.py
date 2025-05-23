"""Utility to convert daily annotation YAML files to individual annotation format."""

import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def parse_filename_components(filename: str) -> Optional[Dict[str, str]]:
    """Parse phenocam filename to extract components.
    
    Expected format: {station}_{instrument}_{year}_{doy}_{date}_{time}.jpg
    Example: lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        Dictionary with parsed components or None if parsing fails
    """
    # Remove extension
    base_name = Path(filename).stem
    
    # Pattern for phenocam filename
    pattern = r'([a-zA-Z]+)_([A-Z0-9_]+)_(\d{4})_(\d{3})_(\d{8})_(\d{6})'
    match = re.match(pattern, base_name)
    
    if not match:
        logger.warning(f"Could not parse filename: {filename}")
        return None
    
    station, instrument, year, doy, date, time = match.groups()
    
    return {
        'station': station.lower(),
        'instrument': instrument,
        'year': year,
        'day_of_year': doy,
        'date': date,
        'time': time,
        'base_name': base_name
    }


def normalize_station_name(station: str) -> str:
    """Normalize station name from daily format to individual format.
    
    Args:
        station: Station name (may have special characters)
        
    Returns:
        Normalized lowercase station name
    """
    # Map special character names to standard names
    station_mapping = {
        'Lönnstorp': 'lonnstorp',
        'lonnstorp': 'lonnstorp',
        'Röbäcksdalen': 'robacksdalen',
        'robacksdalen': 'robacksdalen'
    }
    
    return station_mapping.get(station, station.lower())


def convert_daily_to_individual(
    daily_yaml_path: Path,
    output_dir: Path,
    overwrite: bool = False
) -> List[Path]:
    """Convert daily annotation YAML to individual annotation YAMLs.
    
    Args:
        daily_yaml_path: Path to daily annotation YAML file
        output_dir: Directory to save individual annotation files
        overwrite: Whether to overwrite existing files
        
    Returns:
        List of paths to created individual annotation files
    """
    logger.info(f"Converting daily annotations from: {daily_yaml_path}")
    
    # Load daily annotation file
    with open(daily_yaml_path, 'r', encoding='utf-8') as f:
        daily_data = yaml.safe_load(f)
    
    if not daily_data or 'annotations' not in daily_data:
        logger.warning(f"No annotations found in {daily_yaml_path}")
        return []
    
    # Extract metadata from daily file
    daily_station = daily_data.get('station', '')
    daily_instrument = daily_data.get('instrument', '')
    daily_doy = daily_data.get('day_of_year', '')
    created_time = daily_data.get('created', datetime.now().isoformat())
    
    created_files = []
    
    # Process each image in the daily annotations
    for filename, image_annotations in daily_data['annotations'].items():
        if not image_annotations:
            logger.debug(f"Skipping {filename} - no annotations")
            continue
        
        # Parse filename to get components
        components = parse_filename_components(filename)
        if not components:
            logger.warning(f"Skipping {filename} - could not parse filename")
            continue
        
        # Use parsed components, fall back to daily data if needed
        station = normalize_station_name(components.get('station', daily_station))
        instrument = components.get('instrument', daily_instrument)
        year = components.get('year', '')
        doy = components.get('day_of_year', daily_doy)
        date = components.get('date', '')
        time = components.get('time', '')
        
        # Create individual annotation filename
        individual_filename = f"{station}_{instrument}_{year}_{doy}_{date}_{time}_annotations.yaml"
        individual_path = output_dir / individual_filename
        
        # Skip if file exists and not overwriting
        if individual_path.exists() and not overwrite:
            logger.debug(f"Skipping {individual_filename} - already exists")
            continue
        
        # Convert annotations format
        individual_annotations = []
        for roi_annotation in image_annotations:
            # Add missing fields for individual format
            converted_annotation = {
                'roi_name': roi_annotation.get('roi_name', ''),
                'discard': roi_annotation.get('discard', False),
                'snow_presence': roi_annotation.get('snow_presence', False),
                'flags': roi_annotation.get('flags', []),
                'not_needed': False,  # Default value
                '_flag_selector': ''  # Default value
            }
            individual_annotations.append(converted_annotation)
        
        # Create individual annotation data
        individual_data = {
            'created': created_time,
            'last_modified': datetime.now().isoformat(),
            'filename': filename,
            'day_of_year': doy,
            'year': year,
            'station': station,
            'instrument': instrument,
            'annotation_time_minutes': 0.0,  # Default since we don't have this info
            'status': 'completed',  # Assume completed since coming from daily
            'annotations': individual_annotations
        }
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write individual annotation file
        with open(individual_path, 'w', encoding='utf-8') as f:
            yaml.dump(individual_data, f, default_flow_style=False, allow_unicode=True)
        
        created_files.append(individual_path)
        logger.debug(f"Created individual annotation: {individual_filename}")
    
    logger.info(f"Created {len(created_files)} individual annotation files")
    return created_files


def convert_station_daily_annotations(
    station_daily_dir: Path,
    station_individual_dir: Path,
    overwrite: bool = False
) -> Dict[str, List[Path]]:
    """Convert all daily annotation files for a station to individual format.
    
    Args:
        station_daily_dir: Directory containing daily annotation YAML files
        station_individual_dir: Directory to save individual annotation files
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary mapping daily file names to lists of created individual files
    """
    if not station_daily_dir.exists():
        logger.error(f"Daily annotations directory not found: {station_daily_dir}")
        return {}
    
    logger.info(f"Converting daily annotations from: {station_daily_dir}")
    logger.info(f"Output directory: {station_individual_dir}")
    
    # Find all daily annotation YAML files (exclude individual files)
    daily_files = list(station_daily_dir.glob("annotations_*.yaml")) + list(station_daily_dir.glob("annotations_*.yml"))
    
    if not daily_files:
        logger.warning(f"No YAML files found in {station_daily_dir}")
        return {}
    
    logger.info(f"Found {len(daily_files)} daily annotation files")
    
    conversion_results = {}
    total_created = 0
    
    for daily_file in sorted(daily_files):
        logger.info(f"Processing: {daily_file.name}")
        
        try:
            created_files = convert_daily_to_individual(
                daily_yaml_path=daily_file,
                output_dir=station_individual_dir,
                overwrite=overwrite
            )
            
            conversion_results[daily_file.name] = created_files
            total_created += len(created_files)
            
            logger.info(f"  Created {len(created_files)} individual files")
            
        except Exception as e:
            logger.error(f"Failed to process {daily_file.name}: {str(e)}")
            conversion_results[daily_file.name] = []
    
    logger.info(f"Conversion complete. Total individual files created: {total_created}")
    return conversion_results


def main():
    """Main function to convert daily annotations for both stations."""
    # Base directory
    base_dir = Path("/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning")
    
    # Station directories - they already exist in the base directory structure
    stations = {
        'lonnstorp': {
            'daily': base_dir / 'lonnstorp' / 'master_annotation_pool',
            'individual': base_dir / 'lonnstorp' / 'master_annotation_pool'
        },
        'robacksdalen': {
            'daily': base_dir / 'robacksdalen' / 'master_annotation_pool',
            'individual': base_dir / 'robacksdalen' / 'master_annotation_pool'
        }
    }
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Convert annotations for each station
    for station_name, paths in stations.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing station: {station_name.upper()}")
        logger.info(f"{'='*50}")
        
        results = convert_station_daily_annotations(
            station_daily_dir=paths['daily'],
            station_individual_dir=paths['individual'],
            overwrite=False  # Don't overwrite existing files
        )
        
        if results:
            logger.info(f"\nConversion summary for {station_name}:")
            total_files = sum(len(files) for files in results.values())
            logger.info(f"  Processed {len(results)} daily files")
            logger.info(f"  Created {total_files} individual files")
            
            # Show some examples
            for daily_file, individual_files in list(results.items())[:3]:
                logger.info(f"  {daily_file} -> {len(individual_files)} individual files")
        else:
            logger.warning(f"No files processed for {station_name}")


if __name__ == "__main__":
    main()