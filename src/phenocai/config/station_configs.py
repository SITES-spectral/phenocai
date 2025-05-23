"""
Station-specific configurations for PhenoCAI

This module contains predefined configurations for the main stations:
- Lönnstorp (lonnstorp)
- Röbäcksdalen (robacksdalen)
"""

from typing import Dict, Any


# Station configurations with their typical instruments
STATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    'lonnstorp': {
        'full_name': 'Lönnstorp',
        'station_code': 'LON',
        'default_instrument': 'LON_AGR_PL01_PHE01',
        'instruments': [
            'LON_AGR_PL01_PHE01',
            'LON_AGR_PL01_PHE02',  # If exists
        ],
        'description': 'Agricultural research station in southern Sweden',
        'typical_rois': ['ROI_00', 'ROI_01', 'ROI_02', 'ROI_03', 'ROI_06'],
        'latitude': 55.6686,
        'longitude': 13.1073,
        'timezone': 'Europe/Stockholm'
    },
    'robacksdalen': {
        'full_name': 'Röbäcksdalen', 
        'station_code': 'RBD',
        'default_instrument': 'RBD_AGR_PL01_PHE01',
        'instruments': [
            'RBD_AGR_PL01_PHE01',
            'RBD_FOR_FL01_PHE01',  # Forest site if exists
        ],
        'description': 'Agricultural research station in northern Sweden',
        'typical_rois': ['ROI_00', 'ROI_01', 'ROI_02', 'ROI_03'],
        'latitude': 63.8111,
        'longitude': 20.2394,
        'timezone': 'Europe/Stockholm'
    },
    'abisko': {
        'full_name': 'Abisko',
        'station_code': 'ANS',
        'default_instrument': 'ANS_FOR_BL01_PHE01',
        'instruments': [
            'ANS_FOR_BL01_PHE01',
            'ANS_FOR_BL02_PHE01',
        ],
        'description': 'Subarctic research station',
        'typical_rois': ['ROI_00', 'ROI_01'],
        'latitude': 68.3549,
        'longitude': 18.8153,
        'timezone': 'Europe/Stockholm'
    },
    'grimso': {
        'full_name': 'Grimsö',
        'station_code': 'GRI',
        'default_instrument': 'GRI_FOR_FL01_PHE01',
        'instruments': [
            'GRI_FOR_FL01_PHE01',
        ],
        'description': 'Forest research station',
        'typical_rois': ['ROI_00', 'ROI_01'],
        'latitude': 59.7275,
        'longitude': 15.4679,
        'timezone': 'Europe/Stockholm'
    },
    'skogaryd': {
        'full_name': 'Skogaryd',
        'station_code': 'SKO',
        'default_instrument': 'SKO_MIR_FL01_PHE01',
        'instruments': [
            'SKO_MIR_FL01_PHE01',
            'SKO_FOR_FL01_PHE01',
        ],
        'description': 'Mixed forest and wetland station',
        'typical_rois': ['ROI_00', 'ROI_01'],
        'latitude': 58.3817,
        'longitude': 12.1455,
        'timezone': 'Europe/Stockholm'
    },
    'svartberget': {
        'full_name': 'Svartberget',
        'station_code': 'SVB',
        'default_instrument': 'SVB_FOR_FL01_PHE01',
        'instruments': [
            'SVB_FOR_FL01_PHE01',
        ],
        'description': 'Boreal forest research station',
        'typical_rois': ['ROI_00', 'ROI_01'],
        'latitude': 64.2561,
        'longitude': 19.7747,
        'timezone': 'Europe/Stockholm'
    }
}


def get_station_config(station_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific station.
    
    Args:
        station_name: Station name (case-insensitive)
        
    Returns:
        Station configuration dictionary
        
    Raises:
        ValueError: If station not found
    """
    station_name = station_name.lower()
    
    if station_name not in STATION_CONFIGS:
        available = ', '.join(STATION_CONFIGS.keys())
        raise ValueError(f"Unknown station '{station_name}'. Available: {available}")
    
    return STATION_CONFIGS[station_name]


def get_default_years() -> Dict[str, str]:
    """Get default years for each station based on typical data availability."""
    return {
        'lonnstorp': '2024',
        'robacksdalen': '2024',
        'abisko': '2023',
        'grimso': '2023',
        'skogaryd': '2023',
        'svartberget': '2023'
    }


def validate_instrument_for_station(station_name: str, instrument: str) -> bool:
    """
    Validate if an instrument belongs to a station.
    
    Args:
        station_name: Station name
        instrument: Instrument ID
        
    Returns:
        True if valid, False otherwise
    """
    try:
        config = get_station_config(station_name)
        # Check if instrument starts with station code
        if not instrument.startswith(config['station_code']):
            return False
        # Check if in known instruments list
        return instrument in config['instruments']
    except ValueError:
        return False


def get_all_stations() -> list:
    """Get list of all available station names."""
    return list(STATION_CONFIGS.keys())


def get_primary_stations() -> list:
    """Get list of primary stations for this project."""
    return ['lonnstorp', 'robacksdalen']


if __name__ == "__main__":
    # Display station information
    print("=== PhenoCAI Station Configurations ===\n")
    
    primary_stations = get_primary_stations()
    print(f"Primary stations for this project: {', '.join(primary_stations)}\n")
    
    for station in primary_stations:
        config = get_station_config(station)
        print(f"{config['full_name']} ({station}):")
        print(f"  Code: {config['station_code']}")
        print(f"  Default instrument: {config['default_instrument']}")
        print(f"  Location: {config['latitude']}°N, {config['longitude']}°E")
        print(f"  Typical ROIs: {', '.join(config['typical_rois'])}")
        print()
    
    print("\nAll available stations:")
    for station in get_all_stations():
        print(f"  - {station}")