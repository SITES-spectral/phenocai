"""
Dataset creation and management commands
"""
import click
from pathlib import Path
import pandas as pd

from ...config.setup import config
from ...data_management import (
    create_master_annotation_dataframe,
    create_multi_station_dataset,
    load_multi_station_dataset,
    filter_dataset_by_criteria
)


@click.group()
def dataset():
    """Create and manage datasets."""
    pass


@dataset.command()
@click.option('--output', '-o', type=click.Path(), help='Output path for dataset')
@click.option('--include-unannotated', is_flag=True, help='Include unannotated ROIs')
@click.option('--test-size', default=0.2, help='Test set fraction (default: 0.2)')
@click.option('--val-size', default=0.1, help='Validation set fraction (default: 0.1)')
@click.option('--format', type=click.Choice(['csv', 'parquet']), default='csv', help='Output format')
@click.option('--instrument', '-i', help='Instrument ID (overrides current instrument)')
@click.option('--complete-rois-only/--no-complete-rois-only', default=True, help='Only include images with all ROIs annotated')
@click.option('--min-day', type=int, help='Minimum day of year to include (e.g., 32)')
@click.option('--roi-filter', multiple=True, help='Only include specific ROIs (e.g., --roi-filter ROI_00)')
def create(output, include_unannotated, test_size, val_size, format, instrument, complete_rois_only, min_day, roi_filter):
    """Create dataset from current station's annotations."""
    # Handle instrument switching if provided
    if instrument:
        try:
            config.switch_instrument(instrument)
            click.echo(f"Using instrument: {instrument}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            return 1
    
    click.echo(f"Creating dataset for {config.current_station} - {config.current_instrument}...")
    
    # Set output path with intelligent naming
    if output is None:
        # Generate filename based on station, instrument, year, and options
        suffix = '.parquet' if format == 'parquet' else '.csv'
        filename_parts = [
            config.current_station
        ]
        
        # Add instrument if there are multiple for this station
        available_instruments = config.list_available_instruments()
        if len(available_instruments) > 1:
            # Extract just the instrument number part (e.g., PHE01 from LON_AGR_PL01_PHE01)
            inst_parts = config.current_instrument.split('_')
            if len(inst_parts) >= 4:
                inst_suffix = inst_parts[-1]  # Get PHE01
                filename_parts.append(inst_suffix)
        
        filename_parts.append(f'dataset_{config.current_year}')
        
        if include_unannotated:
            filename_parts.append('with_unannotated')
            
        # Add filtering info to filename
        if roi_filter:
            roi_str = '_'.join(roi_filter).lower()
            filename_parts.append(roi_str)
        elif complete_rois_only or min_day is not None:
            if min_day or (config.current_station == 'lonnstorp' and complete_rois_only):
                day_filter = min_day or 32
                filename_parts.append(f'from_day{day_filter}')
            else:
                filename_parts.append('complete_rois')
        
        # Add split info to filename
        filename_parts.append(f'splits_{int(test_size*100)}_{int(val_size*100)}')
        
        filename = '_'.join(filename_parts) + suffix
        output = config.experimental_data_dir / filename
        
        # Ensure experimental data directory exists
        config.experimental_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(output)
        
    # Ensure correct extension
    if format == 'parquet' and not output.suffix == '.parquet':
        output = output.with_suffix('.parquet')
    elif format == 'csv' and not output.suffix == '.csv':
        output = output.with_suffix('.csv')
    
    try:
        # Create dataset
        df, stats = create_master_annotation_dataframe(
            config.master_annotation_pool_dir,
            output_path=None,  # Don't save yet - we'll add splits first
            include_unannotated=include_unannotated
        )
        
        # Filter to complete ROI sets if requested
        if complete_rois_only or min_day is not None:
            from ...data_management.dataset_builder import filter_complete_roi_sets
            
            # For Lönnstorp, we know that all ROIs started being annotated from day 32
            if min_day is None and config.current_station == 'lonnstorp' and not roi_filter:
                min_day = 32
                click.echo(f"Note: For Lönnstorp, filtering to images from day {min_day} onwards (when all ROIs were annotated)")
            
            # Skip complete ROI filtering if specific ROIs are requested
            if not roi_filter:
                df = filter_complete_roi_sets(df, min_day_of_year=min_day)
        
        # Filter to specific ROIs if requested
        if roi_filter:
            roi_list = list(roi_filter)
            initial_count = len(df)
            df = df[df['roi_name'].isin(roi_list)]
            filtered_count = len(df)
            click.echo(f"Filtered to ROIs {roi_list}: {initial_count} → {filtered_count} records")
        
        # Add train/test/val splits
        from ...data_management.dataset_builder import add_train_test_split
        df = add_train_test_split(
            df,
            test_size=test_size,
            val_size=val_size,
            stratify_by='snow_presence',
            random_state=42,
            group_by_day=True
        )
        
        # Update stats with split information
        stats.train_size = len(df[df['split'] == 'train'])
        stats.test_size = len(df[df['split'] == 'test'])
        stats.val_size = len(df[df['split'] == 'val'])
        
        # Save dataset with splits
        if format == 'parquet':
            df.to_parquet(output, index=False)
        else:
            df.to_csv(output, index=False)
        
        # Print statistics
        click.echo(f"\n✓ Created dataset with {len(df)} records")
        stats.print_summary()
        
        click.echo(f"\nDataset saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error creating dataset: {e}", err=True)
        return 1


@dataset.command('multi-station')
@click.option('--stations', '-s', multiple=True, help='Stations to include (defaults to primary stations)')
@click.option('--output', '-o', type=click.Path(), help='Output path for dataset')
@click.option('--include-unannotated', is_flag=True, help='Include unannotated ROIs')
@click.option('--balance', is_flag=True, help='Balance samples across stations')
@click.option('--test-size', default=0.2, help='Test set fraction')
@click.option('--val-size', default=0.1, help='Validation set fraction')
@click.option('--format', type=click.Choice(['csv', 'parquet']), default='csv', help='Output format')
@click.option('--roi-filter', multiple=True, help='Only include specific ROIs (e.g., --roi-filter ROI_00)')
@click.option('--years', multiple=True, help='Years to include (default: current year)')
def multi_station(stations, output, include_unannotated, balance, test_size, val_size, format, roi_filter, years):
    """Create multi-station dataset."""
    # Use provided stations or defaults
    if not stations:
        from ...config.station_configs import get_primary_stations
        stations = get_primary_stations()
        click.echo(f"Using primary stations: {', '.join(stations)}")
    else:
        stations = list(stations)
    
    # Set output path with intelligent naming
    if output is None:
        # Generate filename based on stations and options
        suffix = '.parquet' if format == 'parquet' else '.csv'
        station_str = '_'.join(sorted(stations))
        filename_parts = [
            'multi_station',
            station_str,
            f'dataset_{config.current_year}'
        ]
        
        if include_unannotated:
            filename_parts.append('with_unannotated')
            
        if balance:
            filename_parts.append('balanced')
            
        # Add ROI filter to filename
        if roi_filter:
            roi_str = '_'.join(roi_filter).lower()
            filename_parts.append(roi_str)
        
        # Add years if specified
        if years:
            years_str = '_'.join(sorted(years))
            filename_parts.append(years_str)
        
        # Add split info to filename
        filename_parts.append(f'splits_{int(test_size*100)}_{int(val_size*100)}')
        
        filename = '_'.join(filename_parts) + suffix
        output = config.experimental_data_dir / filename
        
        # Ensure experimental data directory exists
        config.experimental_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(output)
    
    # Ensure correct extension
    if format == 'parquet':
        output = output.with_suffix('.parquet')
    else:
        output = output.with_suffix('.csv')
    
    click.echo(f"\nCreating multi-station dataset...")
    
    try:
        df, station_stats = create_multi_station_dataset(
            stations=stations,
            output_path=output,
            include_unannotated=include_unannotated,
            test_size=test_size,
            val_size=val_size,
            balance_stations=balance,
            roi_filter=list(roi_filter) if roi_filter else None,
            years=list(years) if years else None
        )
        
        click.echo(f"\n✓ Created multi-station dataset")
        click.echo(f"Dataset saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error creating multi-station dataset: {e}", err=True)
        return 1


@dataset.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def info(dataset_path):
    """Show information about a dataset."""
    dataset_path = Path(dataset_path)
    
    click.echo(f"Loading dataset from {dataset_path}...")
    
    try:
        if dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
        
        click.echo(f"\n=== Dataset Information ===")
        click.echo(f"Total records: {len(df)}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        
        if 'station' in df.columns:
            click.echo(f"\nStations:")
            for station, count in df['station'].value_counts().items():
                click.echo(f"  • {station}: {count} records")
        
        if 'split' in df.columns:
            click.echo(f"\nData splits:")
            for split, count in df['split'].value_counts().items():
                click.echo(f"  • {split}: {count} records")
        
        if 'roi_name' in df.columns:
            click.echo(f"\nROI distribution:")
            for roi, count in df['roi_name'].value_counts().head(10).items():
                click.echo(f"  • {roi}: {count} records")
        
        if 'snow_presence' in df.columns:
            snow_count = df['snow_presence'].sum()
            click.echo(f"\nLabels:")
            click.echo(f"  • Snow present: {snow_count} ({snow_count/len(df)*100:.1f}%)")
            click.echo(f"  • No snow: {len(df) - snow_count} ({(len(df)-snow_count)/len(df)*100:.1f}%)")
        
        # Add quality flag statistics
        if 'has_flags' in df.columns or 'flag_count' in df.columns:
            if 'has_flags' not in df.columns and 'flag_count' in df.columns:
                df['has_flags'] = df['flag_count'] > 0
            
            if 'has_flags' in df.columns:
                flagged_count = df['has_flags'].sum()
                click.echo(f"\nQuality issues:")
                click.echo(f"  • With flags: {flagged_count} ({flagged_count/len(df)*100:.1f}%)")
                click.echo(f"  • Clean: {len(df) - flagged_count} ({(len(df)-flagged_count)/len(df)*100:.1f}%)")
        
        if 'flags' in df.columns:
            # Parse and count individual flags
            all_flags = []
            for flags_str in df[df['flags'].notna()]['flags']:
                if flags_str:
                    all_flags.extend(flags_str.split(','))
            
            if all_flags:
                from collections import Counter
                flag_counts = Counter(all_flags)
                click.echo(f"\nTop quality flags:")
                for flag, count in flag_counts.most_common(5):
                    click.echo(f"  • {flag}: {count}")
        
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        return 1


@dataset.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(), required=False)
@click.option('--stations', '-s', multiple=True, help='Filter by stations')
@click.option('--rois', '-r', multiple=True, help='Filter by ROI names')
@click.option('--min-year', type=int, help='Minimum year')
@click.option('--max-year', type=int, help='Maximum year')
@click.option('--exclude-flags', '-x', multiple=True, help='Exclude specific flags')
@click.option('--no-flags', is_flag=True, help='Include only samples with no quality flags')
@click.option('--annotated-only', is_flag=True, default=True, help='Include only annotated samples')
def filter(input_path, output_path, stations, rois, min_year, max_year, exclude_flags, no_flags, annotated_only):
    """Filter an existing dataset."""
    input_path = Path(input_path)
    
    # Generate intelligent output name if not provided
    if output_path is None:
        # Parse input filename to create filtered version
        stem = input_path.stem
        suffix = input_path.suffix
        
        # Add filter descriptors
        filter_parts = [stem]
        
        if no_flags:
            filter_parts.append('clean')
        elif exclude_flags:
            flags_str = '_'.join(sorted(exclude_flags))
            filter_parts.append(f'no_{flags_str}')
            
        if stations:
            stations_str = '_'.join(sorted(stations))
            filter_parts.append(f'stations_{stations_str}')
            
        if rois:
            rois_str = '_'.join(sorted(rois))
            filter_parts.append(f'rois_{rois_str}')
            
        if min_year or max_year:
            year_parts = []
            if min_year:
                year_parts.append(f'from{min_year}')
            if max_year:
                year_parts.append(f'to{max_year}')
            filter_parts.append('_'.join(year_parts))
            
        filter_parts.append('filtered')
        
        output_filename = '_'.join(filter_parts) + suffix
        output_path = input_path.parent / output_filename
    else:
        output_path = Path(output_path)
    
    click.echo(f"Loading dataset from {input_path}...")
    
    try:
        # Load dataset
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        original_size = len(df)
        
        # Apply filters
        # Handle no_flags option
        if no_flags:
            # Override exclude_flags to exclude all records with any flags
            df = df[~df['has_flags']] if 'has_flags' in df.columns else df[df['flag_count'] == 0]
        else:
            df = filter_dataset_by_criteria(
                df,
                stations=list(stations) if stations else None,
                rois=list(rois) if rois else None,
                min_year=min_year,
                max_year=max_year,
                exclude_flags=list(exclude_flags) if exclude_flags else None,
                annotated_only=annotated_only
            )
        
        # Save filtered dataset
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        click.echo(f"\n✓ Filtered dataset: {original_size} → {len(df)} records")
        click.echo(f"Saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error filtering dataset: {e}", err=True)
        return 1


@dataset.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(), required=False)
@click.option('--ratio', '-r', default=1.0, help='Ratio of majority to minority class (default: 1.0 for balanced)')
@click.option('--target-column', default='snow_presence', help='Column to balance on')
@click.option('--random-seed', default=42, type=int, help='Random seed for reproducibility')
def balance(input_path, output_path, ratio, target_column, random_seed):
    """Balance dataset by undersampling majority class.
    
    This command creates a balanced dataset by taking all samples from the 
    minority class (usually snow) and randomly sampling from the majority 
    class (usually no-snow) to achieve the desired ratio.
    
    Examples:
        # Create perfectly balanced dataset (1:1 ratio)
        phenocai dataset balance dataset.csv
        
        # Create dataset with 2 no-snow samples for each snow sample
        phenocai dataset balance dataset.csv --ratio 2.0
    """
    from ...data_management.dataset_balancer import balance_dataset_from_csv
    
    input_path = Path(input_path)
    
    # Generate output path if not provided
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        
        # Add balance info to filename
        if ratio == 1.0:
            balance_str = 'balanced'
        else:
            balance_str = f'ratio_{ratio:.1f}'.replace('.', '_')
        
        output_filename = f"{stem}_{balance_str}{suffix}"
        output_path = input_path.parent / output_filename
    else:
        output_path = Path(output_path)
    
    click.echo(f"Balancing dataset with ratio {ratio}:1 (majority:minority)...")
    
    try:
        # Balance the dataset
        result_path = balance_dataset_from_csv(
            str(input_path),
            str(output_path),
            target_column=target_column,
            ratio=ratio,
            random_seed=random_seed
        )
        
        click.echo(f"\n✓ Successfully created balanced dataset")
        click.echo(f"Output saved to: {result_path}")
        
    except Exception as e:
        click.echo(f"Error balancing dataset: {e}", err=True)
        return 1