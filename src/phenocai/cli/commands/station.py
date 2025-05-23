"""
Station management commands
"""
import click
import os
from pathlib import Path

from ...config.station_configs import (
    get_all_stations, 
    get_primary_stations,
    get_station_config,
    validate_instrument_for_station
)
from ...config.setup import config


@click.group()
def station():
    """Manage station configurations."""
    pass


@station.command()
def list():
    """List all available stations."""
    click.echo("\n=== Available Stations ===")
    
    primary = get_primary_stations()
    all_stations = get_all_stations()
    
    click.echo("\nPrimary stations:")
    for s in primary:
        cfg = get_station_config(s)
        click.echo(f"  • {cfg['full_name']} ({s})")
        click.echo(f"    Location: {cfg['latitude']}°N, {cfg['longitude']}°E")
        click.echo(f"    Instruments: {', '.join(cfg['instruments'])}")
    
    click.echo("\nAdditional stations:")
    for s in all_stations:
        if s not in primary:
            cfg = get_station_config(s)
            click.echo(f"  • {cfg['full_name']} ({s})")


@station.command()
@click.argument('station_name', type=click.Choice(get_all_stations()))
@click.option('--instrument', '-i', help='Instrument ID (uses default if not specified)')
@click.option('--year', '-y', help='Year (uses default if not specified)')
def switch(station_name, instrument, year):
    """Switch to a different station configuration."""
    from ...config.setup import config as current_config
    
    # Get station configuration
    station_cfg = get_station_config(station_name)
    
    # Use defaults if not provided
    if instrument is None:
        instrument = station_cfg['default_instrument']
    else:
        # Validate instrument
        if not validate_instrument_for_station(station_name, instrument):
            click.echo(f"Error: Invalid instrument '{instrument}' for station '{station_name}'", err=True)
            click.echo(f"Valid instruments: {', '.join(station_cfg['instruments'])}", err=True)
            return 1
    
    if year is None:
        from ...config.station_configs import get_default_years
        year = get_default_years().get(station_name, '2024')
    
    click.echo(f"\nSwitching to {station_cfg['full_name']} ({station_name})...")
    
    # Update environment variables
    os.environ['PHENOCAI_CURRENT_STATION'] = station_name
    os.environ['PHENOCAI_CURRENT_INSTRUMENT'] = instrument
    os.environ['PHENOCAI_CURRENT_YEAR'] = year
    
    # Update env.sh file
    env_path = Path(__file__).parent.parent.parent / 'config' / 'env.sh'
    
    if env_path.exists():
        # Read current env.sh
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update relevant lines
        updated_lines = []
        for line in lines:
            if line.startswith('export PHENOCAI_CURRENT_STATION='):
                updated_lines.append(f'export PHENOCAI_CURRENT_STATION="{station_name}"\n')
            elif line.startswith('export PHENOCAI_CURRENT_INSTRUMENT='):
                updated_lines.append(f'export PHENOCAI_CURRENT_INSTRUMENT="{instrument}"\n')
            elif line.startswith('export PHENOCAI_CURRENT_YEAR='):
                updated_lines.append(f'export PHENOCAI_CURRENT_YEAR="{year}"\n')
            else:
                updated_lines.append(line)
        
        # Write updated env.sh
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        click.echo(f"✓ Updated env.sh")
    
    click.echo(f"✓ Switched to {station_cfg['full_name']}")
    click.echo(f"  Instrument: {instrument}")
    click.echo(f"  Year: {year}")
    
    click.echo(f"\nTo apply changes in your shell, run:")
    click.echo(f"  source {env_path}")


@station.command()
@click.argument('station_name', type=click.Choice(get_all_stations()))
def info(station_name):
    """Show detailed information about a station."""
    cfg = get_station_config(station_name)
    
    click.echo(f"\n=== {cfg['full_name']} ({station_name}) ===")
    click.echo(f"Description: {cfg['description']}")
    click.echo(f"Station code: {cfg['station_code']}")
    click.echo(f"Location: {cfg['latitude']}°N, {cfg['longitude']}°E")
    click.echo(f"Timezone: {cfg['timezone']}")
    
    click.echo(f"\nInstruments:")
    for inst in cfg['instruments']:
        is_default = " (default)" if inst == cfg['default_instrument'] else ""
        click.echo(f"  • {inst}{is_default}")
    
    click.echo(f"\nTypical ROIs: {', '.join(cfg['typical_rois'])}")
    
    if station_name in get_primary_stations():
        click.echo(f"\n✓ This is a PRIMARY station for the PhenoCAI project")