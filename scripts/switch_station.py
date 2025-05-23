#!/usr/bin/env python3
"""
Switch Station Configuration Script

This script helps switch between different stations (primarily Lönnstorp and Röbäcksdalen)
by updating environment variables and creating station-specific configuration files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phenocai.config.station_configs import (
    get_station_config, 
    get_all_stations,
    get_primary_stations,
    get_default_years
)


def update_env_file(station: str, instrument: str = None, year: str = None):
    """
    Update the env.sh file with new station configuration.
    
    Args:
        station: Station name
        instrument: Instrument ID (optional, uses default if not provided)
        year: Year (optional, uses default if not provided)
    """
    # Get station configuration
    config = get_station_config(station)
    
    # Use defaults if not provided
    if instrument is None:
        instrument = config['default_instrument']
    if year is None:
        default_years = get_default_years()
        year = default_years.get(station, '2024')
    
    # Path to env.sh
    env_path = Path(__file__).parent.parent / 'src' / 'phenocai' / 'config' / 'env.sh'
    
    # Read current env.sh
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update relevant lines
    updated_lines = []
    for line in lines:
        if line.startswith('export PHENOCAI_CURRENT_STATION='):
            updated_lines.append(f'export PHENOCAI_CURRENT_STATION="{station}"\n')
        elif line.startswith('export PHENOCAI_CURRENT_INSTRUMENT='):
            updated_lines.append(f'export PHENOCAI_CURRENT_INSTRUMENT="{instrument}"\n')
        elif line.startswith('export PHENOCAI_CURRENT_YEAR='):
            updated_lines.append(f'export PHENOCAI_CURRENT_YEAR="{year}"\n')
        else:
            updated_lines.append(line)
    
    # Write updated env.sh
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"✓ Updated env.sh for {config['full_name']} ({station})")
    print(f"  Instrument: {instrument}")
    print(f"  Year: {year}")


def create_station_env_file(station: str, output_dir: Path = None):
    """
    Create a station-specific environment file.
    
    Args:
        station: Station name
        output_dir: Output directory for env file
    """
    config = get_station_config(station)
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'src' / 'phenocai' / 'config'
    
    output_path = output_dir / f'env_{station}.sh'
    
    env_content = f"""#!/bin/bash

# PhenoCAI Environment Configuration for {config['full_name']}

# Station Information
export PHENOCAI_CURRENT_STATION="{station}"
export PHENOCAI_CURRENT_INSTRUMENT="{config['default_instrument']}"
export PHENOCAI_CURRENT_YEAR="{get_default_years().get(station, '2024')}"

# Station Details
export PHENOCAI_STATION_FULL_NAME="{config['full_name']}"
export PHENOCAI_STATION_CODE="{config['station_code']}"
export PHENOCAI_STATION_LAT="{config['latitude']}"
export PHENOCAI_STATION_LON="{config['longitude']}"

# Load base configuration
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

echo "Loaded {config['full_name']} configuration"
"""
    
    with open(output_path, 'w') as f:
        f.write(env_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"✓ Created station-specific env file: {output_path}")
    return output_path


def show_current_configuration():
    """Display current configuration from environment."""
    print("\n=== Current Configuration ===")
    print(f"Station: {os.getenv('PHENOCAI_CURRENT_STATION', 'Not set')}")
    print(f"Instrument: {os.getenv('PHENOCAI_CURRENT_INSTRUMENT', 'Not set')}")
    print(f"Year: {os.getenv('PHENOCAI_CURRENT_YEAR', 'Not set')}")
    print(f"Data Dir: {os.getenv('PHENOCAI_DATA_DIR', 'Not set')}")


def main():
    parser = argparse.ArgumentParser(
        description='Switch between different station configurations for PhenoCAI'
    )
    
    parser.add_argument(
        'station',
        nargs='?',
        choices=get_all_stations(),
        help='Station to switch to'
    )
    
    parser.add_argument(
        '--instrument', '-i',
        help='Instrument ID (uses default if not specified)'
    )
    
    parser.add_argument(
        '--year', '-y',
        help='Year (uses default if not specified)'
    )
    
    parser.add_argument(
        '--create-env-file', '-c',
        action='store_true',
        help='Create station-specific environment file'
    )
    
    parser.add_argument(
        '--primary-only',
        action='store_true',
        help='Show only primary stations (Lönnstorp and Röbäcksdalen)'
    )
    
    parser.add_argument(
        '--show-current',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    args = parser.parse_args()
    
    if args.show_current:
        show_current_configuration()
        return
    
    if args.station is None:
        parser.error("Station argument is required unless using --show-current")
        return 1
    
    # Get station config
    try:
        config = get_station_config(args.station)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"\nSwitching to {config['full_name']} ({args.station})...")
    
    # Update main env.sh
    update_env_file(args.station, args.instrument, args.year)
    
    # Create station-specific env file if requested
    if args.create_env_file:
        env_path = create_station_env_file(args.station)
        print(f"\nTo use this configuration, run:")
        print(f"  source {env_path}")
    else:
        print(f"\nTo apply this configuration, run:")
        print(f"  source src/phenocai/config/env.sh")
    
    # Show available instruments
    print(f"\nAvailable instruments for {config['full_name']}:")
    for inst in config['instruments']:
        print(f"  - {inst}")
    
    # Show note for primary stations
    if args.station in get_primary_stations():
        print(f"\n✓ {config['full_name']} is a PRIMARY station for this project")


if __name__ == "__main__":
    main()