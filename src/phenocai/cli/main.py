"""
PhenoCAI CLI Main Entry Point
"""
import click
import logging
from pathlib import Path

from ..config.setup import config
from ..config.station_configs import get_all_stations, get_primary_stations


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """
    PhenoCAI - Phenocam AI Analysis Tool
    
    Automated phenological camera image analysis and classification
    for agricultural and environmental monitoring.
    
    Primary stations: Lönnstorp, Röbäcksdalen
    """
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Show current configuration
    if verbose:
        click.echo(f"Current station: {config.current_station}")
        click.echo(f"Current instrument: {config.current_instrument}")
        click.echo(f"Current year: {config.current_year}")


# Import command groups
from .commands.station import station
from .commands.dataset import dataset
from .commands.train import train
from .commands.evaluate import evaluate
from .commands.predict import predict
from .commands.config import config_cmd
from .commands.convert import convert
from .commands.pipeline import pipeline
from .analyze import analyze

# Add command groups to main CLI
cli.add_command(station)
cli.add_command(dataset)
cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(predict)
cli.add_command(config_cmd)
cli.add_command(convert)
cli.add_command(pipeline)
cli.add_command(analyze)


@cli.command()
def info():
    """Show current configuration and system information."""
    click.echo("\n=== PhenoCAI System Information ===")
    click.echo(f"Project root: {config.project_root}")
    click.echo(f"Current station: {config.current_station}")
    click.echo(f"Current instrument: {config.current_instrument}")
    click.echo(f"Current year: {config.current_year}")
    
    click.echo(f"\nData directory: {config.data_dir}")
    click.echo(f"Master annotation pool: {config.master_annotation_pool_dir}")
    click.echo(f"Model output directory: {config.model_output_dir}")
    
    # Check if directories exist
    if config.master_annotation_pool_dir.exists():
        annotation_count = len(list(config.master_annotation_pool_dir.glob("*.yaml")))
        click.echo(f"\nAnnotation files found: {annotation_count}")
    else:
        click.echo("\n⚠️  Master annotation pool directory not found!")
    
    if config.image_base_dir.exists():
        click.echo("✓ Image base directory exists")
    else:
        click.echo("⚠️  Image base directory not found!")


@cli.command()
def version():
    """Show PhenoCAI version."""
    try:
        from .. import __version__
        click.echo(f"PhenoCAI version: {__version__}")
    except:
        click.echo("PhenoCAI version: development")


if __name__ == '__main__':
    cli()