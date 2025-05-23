"""PhenoCAI configuration module."""

from .setup import config, PhenoCAIConfig
from .station_configs import get_all_stations, get_primary_stations, STATION_CONFIGS

__all__ = [
    'config',
    'PhenoCAIConfig', 
    'get_all_stations',
    'get_primary_stations',
    'STATION_CONFIGS'
]