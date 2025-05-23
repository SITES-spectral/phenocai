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
    
    # Handle year first
    if year is None:
        from ...config.station_configs import get_default_years
        year = get_default_years().get(station_name, '2024')
    
    # Update year in environment
    os.environ['PHENOCAI_CURRENT_YEAR'] = year
    current_config.current_year = year
    
    try:
        # Use the new switch_station method which validates against stations.yaml
        current_config.switch_station(station_name, instrument)
        
        # Get the actual instrument used (in case default was selected)
        instrument = current_config.current_instrument
        
        click.echo(f"\nSwitched to {station_cfg['full_name']} ({station_name})")
        click.echo(f"  Instrument: {instrument}")
        click.echo(f"  Year: {year}")
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        
        # Show available instruments from registry
        available = current_config.list_available_instruments()
        if available:
            click.echo(f"Available instruments: {', '.join(available)}", err=True)
        return 1
    
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
    from ...config.setup import config as current_config
    
    cfg = get_station_config(station_name)
    
    click.echo(f"\n=== {cfg['full_name']} ({station_name}) ===")
    click.echo(f"Description: {cfg['description']}")
    click.echo(f"Station code: {cfg['station_code']}")
    click.echo(f"Location: {cfg['latitude']}°N, {cfg['longitude']}°E")
    click.echo(f"Timezone: {cfg['timezone']}")
    
    click.echo(f"\nInstruments:")
    
    # Try to get instruments from registry if available
    try:
        # Temporarily switch to check instruments (without updating env vars)
        registry = current_config._registry
        if registry:
            station_info = registry.get_station(station_name)
            if station_info:
                for inst_id, inst in station_info.instruments.items():
                    status = f" [{inst.status}]" if inst.status != "Active" else ""
                    is_default = " (default)" if inst_id == cfg['default_instrument'] else ""
                    click.echo(f"  • {inst_id}{is_default}{status}")
                    if inst.ecosystem:
                        click.echo(f"    Ecosystem: {inst.ecosystem}")
                    if inst.viewing_direction:
                        click.echo(f"    Viewing: {inst.viewing_direction} ({inst.azimuth}°)")
            else:
                # Fallback to config file
                for inst in cfg['instruments']:
                    is_default = " (default)" if inst == cfg['default_instrument'] else ""
                    click.echo(f"  • {inst}{is_default}")
        else:
            # Fallback to config file
            for inst in cfg['instruments']:
                is_default = " (default)" if inst == cfg['default_instrument'] else ""
                click.echo(f"  • {inst}{is_default}")
    except:
        # Fallback to config file
        for inst in cfg['instruments']:
            is_default = " (default)" if inst == cfg['default_instrument'] else ""
            click.echo(f"  • {inst}{is_default}")
    
    click.echo(f"\nTypical ROIs: {', '.join(cfg['typical_rois'])}")
    
    if station_name in get_primary_stations():
        click.echo(f"\n✓ This is a PRIMARY station for the PhenoCAI project")


@station.command()
def instruments():
    """List instruments for the current station."""
    from ...config.setup import config as current_config
    
    click.echo(f"\n=== Instruments for {current_config.current_station} ===")
    
    available = current_config.list_available_instruments()
    if not available:
        click.echo("No instruments found or registry not loaded.")
        return
    
    current = current_config.current_instrument
    
    for inst in available:
        is_current = " (current)" if inst == current else ""
        click.echo(f"  • {inst}{is_current}")
    
    click.echo(f"\nTo switch instrument, use:")
    click.echo(f"  uv run phenocai station switch {current_config.current_station} --instrument INSTRUMENT_ID")