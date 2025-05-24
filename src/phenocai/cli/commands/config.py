"""
Configuration management commands
"""
import click
from pathlib import Path
import yaml

from ...config.setup import config


@click.group(name='config')
def config_cmd():
    """Manage PhenoCAI configuration."""
    pass


@config_cmd.command()
def show():
    """Show current configuration."""
    click.echo("\n=== PhenoCAI Configuration ===")
    
    click.echo("\nStation Settings:")
    click.echo(f"  Current station: {config.current_station}")
    click.echo(f"  Current instrument: {config.current_instrument}")
    click.echo(f"  Current year: {config.current_year}")
    
    click.echo("\nPaths:")
    paths = config.get_all_paths()
    for name, path in paths.items():
        exists = "✓" if Path(path).exists() else "✗"
        click.echo(f"  {exists} {name}: {path}")
    
    click.echo("\nModel Parameters:")
    click.echo(f"  ROI image size: {config.roi_img_size}")
    click.echo(f"  Batch size: {config.batch_size}")
    click.echo(f"  Epochs: {config.epochs}")
    click.echo(f"  Initial learning rate: {config.initial_lr}")
    click.echo(f"  Prediction threshold: {config.prediction_threshold}")
    
    click.echo("\nHeuristic Thresholds:")
    click.echo(f"  Snow min pixel %: {config.snow_min_pixel_percentage}")
    click.echo(f"  Blur threshold: {config.discard_blur_threshold_roi}")
    click.echo(f"  Dark threshold: {config.discard_illumination_dark_threshold_roi}")
    click.echo(f"  Bright threshold: {config.discard_illumination_bright_threshold_roi}")


@config_cmd.command()
@click.argument('output_path', type=click.Path())
def export(output_path):
    """Export current configuration to YAML file."""
    output_path = Path(output_path)
    
    try:
        config.save_to_yaml(output_path)
        click.echo(f"✓ Configuration exported to: {output_path}")
    except Exception as e:
        click.echo(f"Error exporting configuration: {e}", err=True)
        return 1


@config_cmd.command()
@click.argument('input_path', type=click.Path(exists=True))
def load(input_path):
    """Load configuration from YAML file."""
    input_path = Path(input_path)
    
    try:
        from ...config.setup import PhenoCAIConfig
        loaded_config = PhenoCAIConfig.load_from_yaml(input_path)
        
        click.echo(f"✓ Loaded configuration from: {input_path}")
        click.echo(f"  Station: {loaded_config.current_station}")
        click.echo(f"  Instrument: {loaded_config.current_instrument}")
        click.echo(f"  Year: {loaded_config.current_year}")
        
        click.echo("\nNote: This loads configuration for the current session only.")
        click.echo("To make changes permanent, update env.sh or use 'phenocai station switch'")
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return 1


@config_cmd.command()
def validate():
    """Validate current configuration."""
    click.echo("Validating configuration...")
    
    warnings = config.validate_configuration()
    
    if not warnings:
        click.echo("✓ Configuration is valid!")
    else:
        click.echo("\n⚠️  Configuration warnings:")
        for warning in warnings:
            click.echo(f"  - {warning}")
        return 1


@config_cmd.command()
def init():
    """Initialize directories and configuration."""
    click.echo("Initializing PhenoCAI directories...")
    
    try:
        config.setup_directories()
        click.echo("✓ All directories created/verified")
        
        # Save example configuration
        example_path = config.experimental_data_dir / 'phenocai_config.yaml'
        config.save_to_yaml(example_path)
        click.echo(f"✓ Example configuration saved to: {example_path}")
        
    except Exception as e:
        click.echo(f"Error during initialization: {e}", err=True)
        return 1


@config_cmd.command()
@click.option('--sample-image', type=click.Path(exists=True),
              help='Sample image to calculate ROI_00 from')
@click.option('--station', help='Specific station to update')
@click.option('--instrument', help='Specific instrument to update')
@click.option('--method', type=click.Choice(['gradient', 'color', 'fixed']),
              default='gradient', help='Horizon detection method')
@click.option('--force', is_flag=True, help='Force recalculation even if ROI_00 exists')
def add_roi_00(sample_image, station, instrument, method, force):
    """Add ROI_00 (full image minus sky) to stations.yaml."""
    from ...roi_calculator import add_roi_00_to_station_config, update_all_stations_roi_00
    
    stations_yaml_path = Path(config.roi_config_file_path)
    
    if station and instrument:
        # Update specific instrument
        click.echo(f"Adding ROI_00 to {station} - {instrument}...")
        
        sample_path = Path(sample_image) if sample_image else None
        
        try:
            config_dict = add_roi_00_to_station_config(
                stations_yaml_path,
                station,
                instrument,
                sample_path,
                method,
                force=force
            )
            
            # Save updated configuration
            import yaml
            with open(stations_yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            click.echo(f"✓ Added ROI_00 to {station} - {instrument}")
            
        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            return 1
    else:
        # Update all primary stations
        click.echo("Updating all primary stations with ROI_00...")
        
        sample_dir = Path(sample_image).parent if sample_image else None
        
        try:
            update_all_stations_roi_00(stations_yaml_path, sample_dir, force=force)
            click.echo("✓ Updated all stations with ROI_00")
            
        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            return 1