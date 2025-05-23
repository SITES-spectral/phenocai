"""
Station Registry - Dynamic loading and validation from stations.yaml

This module provides the source of truth for station and instrument configurations
by parsing the stations.yaml file directly.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class InstrumentInfo:
    """Information about a phenocam instrument."""
    id: str
    station: str
    ecosystem: str
    location: str
    instrument_number: str
    status: str = "Active"
    platform_type: str = ""
    platform_height: float = 0.0
    viewing_direction: str = ""
    azimuth: float = 0.0
    rois: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if instrument is active."""
        return self.status.lower() == "active"


@dataclass 
class StationInfo:
    """Information about a research station."""
    normalized_name: str
    name: str
    acronym: str
    instruments: Dict[str, InstrumentInfo] = field(default_factory=dict)
    
    @property
    def active_instruments(self) -> List[str]:
        """Get list of active instrument IDs."""
        return [inst_id for inst_id, inst in self.instruments.items() if inst.is_active]
    
    @property
    def default_instrument(self) -> Optional[str]:
        """Get default instrument (first active one)."""
        active = self.active_instruments
        return active[0] if active else None


class StationRegistry:
    """Registry for loading and validating station/instrument configurations from stations.yaml."""
    
    def __init__(self, yaml_path: Optional[Path] = None):
        """Initialize registry with stations.yaml path."""
        if yaml_path is None:
            # Try to find stations.yaml relative to this file
            yaml_path = Path(__file__).parent / "stations.yaml"
        
        self.yaml_path = Path(yaml_path)
        self._stations: Dict[str, StationInfo] = {}
        self._load_stations()
    
    def _load_stations(self) -> None:
        """Load and parse stations.yaml file."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"stations.yaml not found at {self.yaml_path}")
        
        try:
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse stations
            for station_key, station_data in data.items():
                if station_key == "stations":
                    # Handle nested structure if present
                    for sk, sd in station_data.items():
                        self._parse_station(sk, sd)
                elif isinstance(station_data, dict) and 'normalized_name' in station_data:
                    # Direct station entry
                    self._parse_station(station_key, station_data)
                    
        except Exception as e:
            logger.error(f"Error loading stations.yaml: {e}")
            raise
    
    def _parse_station(self, station_key: str, station_data: Dict[str, Any]) -> None:
        """Parse a single station entry."""
        station = StationInfo(
            normalized_name=station_data.get('normalized_name', station_key),
            name=station_data.get('name', station_key),
            acronym=station_data.get('acronym', '')
        )
        
        # Parse phenocams/instruments
        if 'phenocams' in station_data:
            phenocams = station_data['phenocams']
            if 'platforms' in phenocams:
                # Nested platform structure
                for platform_type, platform_data in phenocams['platforms'].items():
                    if 'instruments' in platform_data:
                        for inst_id, inst_data in platform_data['instruments'].items():
                            instrument = self._parse_instrument(
                                inst_id, inst_data, station.normalized_name, platform_type
                            )
                            station.instruments[inst_id] = instrument
            elif 'instruments' in phenocams:
                # Direct instruments structure
                for inst_id, inst_data in phenocams['instruments'].items():
                    instrument = self._parse_instrument(
                        inst_id, inst_data, station.normalized_name, "Unknown"
                    )
                    station.instruments[inst_id] = instrument
        
        self._stations[station.normalized_name] = station
    
    def _parse_instrument(self, inst_id: str, inst_data: Dict[str, Any], 
                         station_name: str, platform_type: str) -> InstrumentInfo:
        """Parse a single instrument entry."""
        return InstrumentInfo(
            id=inst_data.get('id', inst_id),
            station=station_name,
            ecosystem=inst_data.get('ecosystem', ''),
            location=inst_data.get('location', ''),
            instrument_number=inst_data.get('instrument_number', ''),
            status=inst_data.get('status', 'Active'),
            platform_type=platform_type,
            platform_height=inst_data.get('platform_height_in_meters_above_ground', 0.0),
            viewing_direction=inst_data.get('instrument_viewing_direction', ''),
            azimuth=inst_data.get('instrument_azimuth_in_degrees', 0.0),
            rois=inst_data.get('rois', {})
        )
    
    def get_station(self, station_name: str) -> Optional[StationInfo]:
        """Get station info by name (case-insensitive)."""
        station_name = station_name.lower()
        return self._stations.get(station_name)
    
    def get_instrument(self, instrument_id: str) -> Optional[Tuple[StationInfo, InstrumentInfo]]:
        """Get instrument info by ID, returns (station, instrument) tuple."""
        for station in self._stations.values():
            if instrument_id in station.instruments:
                return station, station.instruments[instrument_id]
        return None
    
    def validate_instrument(self, station_name: str, instrument_id: str) -> bool:
        """Validate if an instrument belongs to a station and is active."""
        station = self.get_station(station_name)
        if not station:
            return False
        
        instrument = station.instruments.get(instrument_id)
        if not instrument:
            return False
            
        return instrument.is_active
    
    def list_stations(self) -> List[str]:
        """List all station names."""
        return list(self._stations.keys())
    
    def list_instruments(self, station_name: str, active_only: bool = True) -> List[str]:
        """List instruments for a station."""
        station = self.get_station(station_name)
        if not station:
            return []
        
        if active_only:
            return station.active_instruments
        else:
            return list(station.instruments.keys())
    
    def get_instrument_info(self, station_name: str, instrument_id: str) -> Optional[InstrumentInfo]:
        """Get detailed instrument information."""
        station = self.get_station(station_name)
        if not station:
            return None
        return station.instruments.get(instrument_id)
    
    def get_default_instrument(self, station_name: str) -> Optional[str]:
        """Get default instrument for a station (first active one)."""
        station = self.get_station(station_name)
        if not station:
            return None
        return station.default_instrument


# Global registry instance
_registry: Optional[StationRegistry] = None


def get_registry() -> StationRegistry:
    """Get or create the global station registry."""
    global _registry
    if _registry is None:
        _registry = StationRegistry()
    return _registry


def reload_registry(yaml_path: Optional[Path] = None) -> StationRegistry:
    """Force reload of the station registry."""
    global _registry
    _registry = StationRegistry(yaml_path)
    return _registry