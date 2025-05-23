"""Annotation conversion commands."""

import click
from pathlib import Path
import logging

from ...data_management.annotation_converter import (
    convert_daily_to_individual,
    convert_station_daily_annotations
)


@click.group()
def convert():
    """Convert between annotation formats."""
    pass


@convert.command()
@click.argument('daily_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def daily_to_individual(daily_file: str, output_dir: str, overwrite: bool, verbose: bool):
    """Convert a single daily annotation file to individual files.
    
    Example:
        phenocai convert daily-to-individual annotations_001.yaml ./individual/
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    daily_path = Path(daily_file)
    output_path = Path(output_dir)
    
    click.echo(f"Converting: {daily_path}")
    click.echo(f"Output directory: {output_path}")
    
    try:
        created_files = convert_daily_to_individual(
            daily_yaml_path=daily_path,
            output_dir=output_path,
            overwrite=overwrite
        )
        
        click.echo(f"\n✓ Successfully created {len(created_files)} individual annotation files")
        
        if verbose and created_files:
            click.echo("\nCreated files:")
            for file_path in created_files:
                click.echo(f"  • {file_path.name}")
                
    except Exception as e:
        click.echo(f"❌ Conversion failed: {str(e)}", err=True)
        raise click.Abort()


@convert.command()
@click.argument('daily_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def station_daily_to_individual(daily_dir: str, output_dir: str, overwrite: bool, verbose: bool):
    """Convert all daily annotation files in a directory to individual files.
    
    Example:
        phenocai convert station-daily-to-individual ./daily/lonnstorp/ ./individual/lonnstorp/
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    daily_path = Path(daily_dir)
    output_path = Path(output_dir)
    
    click.echo(f"Converting daily annotations from: {daily_path}")
    click.echo(f"Output directory: {output_path}")
    
    try:
        results = convert_station_daily_annotations(
            station_daily_dir=daily_path,
            station_individual_dir=output_path,
            overwrite=overwrite
        )
        
        total_files = sum(len(files) for files in results.values())
        click.echo(f"\n✓ Successfully processed {len(results)} daily files")
        click.echo(f"✓ Created {total_files} individual annotation files")
        
        if verbose and results:
            click.echo("\nConversion summary:")
            for daily_file, individual_files in results.items():
                click.echo(f"  {daily_file}: {len(individual_files)} individual files")
                
    except Exception as e:
        click.echo(f"❌ Conversion failed: {str(e)}", err=True)
        raise click.Abort()


@convert.command()
@click.option('--base-dir', type=click.Path(exists=True), 
              default="/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning",
              help='Base directory containing daily and individual subdirectories')
@click.option('--stations', multiple=True, default=['lonnstorp', 'robacksdalen'],
              help='Stations to process')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def all_stations(base_dir: str, stations: tuple, overwrite: bool, verbose: bool):
    """Convert daily annotations to individual format for all specified stations.
    
    Example:
        phenocai convert all-stations
        phenocai convert all-stations --stations lonnstorp --verbose
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    base_path = Path(base_dir)
    
    click.echo(f"Base directory: {base_path}")
    click.echo(f"Processing stations: {', '.join(stations)}")
    
    total_daily_files = 0
    total_individual_files = 0
    
    for station in stations:
        click.echo(f"\n{'='*50}")
        click.echo(f"Processing station: {station.upper()}")
        click.echo(f"{'='*50}")
        
        daily_dir = base_path / 'daily' / station
        individual_dir = base_path / 'individual' / station
        
        if not daily_dir.exists():
            click.echo(f"⚠️  Daily directory not found: {daily_dir}")
            continue
        
        try:
            results = convert_station_daily_annotations(
                station_daily_dir=daily_dir,
                station_individual_dir=individual_dir,
                overwrite=overwrite
            )
            
            station_individual_files = sum(len(files) for files in results.values())
            total_daily_files += len(results)
            total_individual_files += station_individual_files
            
            click.echo(f"✓ Station {station}: {len(results)} daily files -> {station_individual_files} individual files")
            
        except Exception as e:
            click.echo(f"❌ Failed to process station {station}: {str(e)}", err=True)
            continue
    
    click.echo(f"\n{'='*50}")
    click.echo(f"CONVERSION SUMMARY")
    click.echo(f"{'='*50}")
    click.echo(f"Total daily files processed: {total_daily_files}")
    click.echo(f"Total individual files created: {total_individual_files}")
    click.echo(f"✓ Conversion complete!")