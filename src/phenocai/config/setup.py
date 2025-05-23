"""
PhenoCAI Configuration Module

This module provides centralized configuration management for the PhenoCAI system,
including paths, model parameters, and heuristic thresholds.
"""
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml


@dataclass
class PhenoCAIConfig:
    """Central configuration class for PhenoCAI system."""
    
    # --- Environment Variables ---
    project_root: Path = field(default_factory=lambda: Path(
        os.getenv('PHENOCAI_PROJECT_ROOT', 
                  Path(__file__).parent.parent.parent.parent.resolve())
    ))
    current_station: str = field(default_factory=lambda: os.getenv('PHENOCAI_CURRENT_STATION', 'lonnstorp'))
    current_instrument: str = field(default_factory=lambda: os.getenv('PHENOCAI_CURRENT_INSTRUMENT', 'LON_AGR_PL01_PHE01'))
    current_year: str = field(default_factory=lambda: os.getenv('PHENOCAI_CURRENT_YEAR', '2024'))
    
    # --- Paths ---
    @property
    def data_dir(self) -> Path:
        """Main data directory for current station."""
        return Path(os.getenv('PHENOCAI_DATA_DIR', 
                              self.project_root / self.current_station))
    
    @property
    def master_annotation_pool_dir(self) -> Path:
        """Directory containing annotation YAML files."""
        return Path(os.getenv('PHENOCAI_MASTER_ANNOTATION_POOL_DIR', 
                              self.data_dir / 'master_annotation_pool'))
    
    @property
    def experimental_data_dir(self) -> Path:
        """Directory for experimental data and intermediate files."""
        return Path(os.getenv('PHENOCAI_EXPERIMENTAL_DATA_DIR', 
                              self.data_dir / 'experimental_data'))
    
    @property
    def master_df_with_splits_path(self) -> Path:
        """Path to master annotations CSV with train/test splits."""
        return Path(os.getenv('PHENOCAI_MASTER_DF_WITH_SPLITS_PATH', 
                              self.experimental_data_dir / 'master_annotations_with_splits.csv'))
    
    @property
    def annotation_root_dir_for_heuristics(self) -> Path:
        """Directory for heuristic-based training pool."""
        return Path(os.getenv('PHENOCAI_ANNOTATION_ROOT_DIR_FOR_HEURISTICS', 
                              self.experimental_data_dir / 'heuristic_train_pool'))
    
    @property
    def image_base_dir(self) -> Path:
        """Base directory for phenocam images."""
        default_path = Path('/home/jobelund/lu2024-12-46/SITES/Spectral/data') / \
                      self.current_station / 'phenocams' / 'products' / \
                      self.current_instrument / 'L1' / self.current_year
        return Path(os.getenv('PHENOCAI_IMAGE_BASE_DIR', str(default_path)))
    
    @property
    def roi_config_file_path(self) -> Path:
        """Path to stations.yaml configuration file."""
        default_path = self.project_root / 'src' / 'phenocai' / 'config' / 'stations.yaml'
        return Path(os.getenv('PHENOCAI_ROI_CONFIG_FILE_PATH', str(default_path)))
    
    @property
    def model_output_dir(self) -> Path:
        """Directory for trained model outputs."""
        return Path(os.getenv('PHENOCAI_MODEL_OUTPUT_DIR', 
                              self.data_dir / 'trained_models' / 'experimental_models_final_df_split'))
    
    @property
    def output_dir_for_new_annotations(self) -> Path:
        """Directory for model-generated annotations."""
        return Path(os.getenv('PHENOCAI_OUTPUT_DIR_FOR_NEW_ANNOTATIONS', 
                              self.data_dir / 'model_generated_annotations_df_split'))
    
    # --- Model & Training Configuration ---
    roi_img_size: Tuple[int, int] = (128, 128)
    batch_size: int = 32
    epochs: int = 20
    initial_lr: float = 0.001
    fine_tune_lr: float = 0.00005
    fine_tune_epochs: int = 10
    fine_tune_at_layer: int = 100  # For MobileNetV2
    sky_line_y_ratio_for_roi00: float = 0.4
    prediction_threshold: float = 0.5
    
    # --- Data Processing ---
    max_images_per_batch: int = 100  # For memory management
    enable_garbage_collection: bool = True
    log_label_source_info: bool = True
    
    # --- Heuristic Configuration ---
    snow_lower_hsv: np.ndarray = field(default_factory=lambda: np.array([0, 0, 170]))
    snow_upper_hsv: np.ndarray = field(default_factory=lambda: np.array([180, 60, 255]))
    snow_min_pixel_percentage: float = 0.40
    
    discard_blur_threshold_roi: float = 70.0
    discard_illumination_dark_threshold_roi: int = 35
    discard_illumination_bright_threshold_roi: int = 230
    discard_low_contrast_threshold_roi: int = 15
    
    # --- Logging Configuration ---
    log_level: str = field(default_factory=lambda: os.getenv('PHENOCAI_LOG_LEVEL', 'INFO'))
    debug_roi_lookup: bool = False
    debug_data_loading: bool = False
    
    # --- Performance Monitoring ---
    enable_performance_tracking: bool = True
    performance_log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize derived paths and validate configuration."""
        if self.performance_log_dir is None:
            self.performance_log_dir = self.experimental_data_dir / 'performance_logs'
    
    def setup_directories(self) -> None:
        """Creates necessary directories if they don't exist."""
        dirs_to_create = [
            self.data_dir,
            self.master_annotation_pool_dir,
            self.experimental_data_dir,
            self.annotation_root_dir_for_heuristics,
            self.model_output_dir,
            self.output_dir_for_new_annotations,
            self.performance_log_dir
        ]
        
        # Create parent directories for files
        for file_path in [self.master_df_with_splits_path]:
            dirs_to_create.append(file_path.parent)
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
        
        logging.info("Checked/created necessary directories.")
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        
        # Check if ROI config file exists
        if not self.roi_config_file_path.exists():
            warnings.append(f"ROI config file not found: {self.roi_config_file_path}")
        
        # Check if image base directory exists
        if not self.image_base_dir.exists():
            warnings.append(f"Image base directory not found: {self.image_base_dir}")
        
        # Validate numeric parameters
        if self.batch_size <= 0:
            warnings.append(f"Invalid batch_size: {self.batch_size}")
        
        if self.epochs <= 0:
            warnings.append(f"Invalid epochs: {self.epochs}")
        
        if not 0 <= self.prediction_threshold <= 1:
            warnings.append(f"Invalid prediction_threshold: {self.prediction_threshold}")
        
        return warnings
    
    def get_all_paths(self) -> Dict[str, Path]:
        """Return dictionary of all configured paths."""
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'master_annotation_pool_dir': self.master_annotation_pool_dir,
            'experimental_data_dir': self.experimental_data_dir,
            'master_df_with_splits_path': self.master_df_with_splits_path,
            'annotation_root_dir_for_heuristics': self.annotation_root_dir_for_heuristics,
            'image_base_dir': self.image_base_dir,
            'roi_config_file_path': self.roi_config_file_path,
            'model_output_dir': self.model_output_dir,
            'output_dir_for_new_annotations': self.output_dir_for_new_annotations,
            'performance_log_dir': self.performance_log_dir
        }
    
    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'station': self.current_station,
            'instrument': self.current_instrument,
            'year': self.current_year,
            'paths': {k: str(v) for k, v in self.get_all_paths().items()},
            'model_params': {
                'roi_img_size': list(self.roi_img_size),
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'initial_lr': self.initial_lr,
                'fine_tune_lr': self.fine_tune_lr,
                'fine_tune_epochs': self.fine_tune_epochs,
                'prediction_threshold': self.prediction_threshold
            },
            'heuristics': {
                'snow_lower_hsv': self.snow_lower_hsv.tolist(),
                'snow_upper_hsv': self.snow_upper_hsv.tolist(),
                'snow_min_pixel_percentage': self.snow_min_pixel_percentage,
                'discard_blur_threshold': self.discard_blur_threshold_roi,
                'discard_illumination_dark': self.discard_illumination_dark_threshold_roi,
                'discard_illumination_bright': self.discard_illumination_bright_threshold_roi,
                'discard_low_contrast': self.discard_low_contrast_threshold_roi
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls, path: Path) -> 'PhenoCAIConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create instance with loaded values
        instance = cls(
            current_station=config_dict.get('station', 'lonnstorp'),
            current_instrument=config_dict.get('instrument', 'LON_AGR_PL01_PHE01'),
            current_year=config_dict.get('year', '2024')
        )
        
        # Update model parameters if present
        if 'model_params' in config_dict:
            mp = config_dict['model_params']
            instance.roi_img_size = tuple(mp.get('roi_img_size', [128, 128]))
            instance.batch_size = mp.get('batch_size', 32)
            instance.epochs = mp.get('epochs', 20)
            instance.initial_lr = mp.get('initial_lr', 0.001)
            instance.prediction_threshold = mp.get('prediction_threshold', 0.5)
        
        return instance


# Create global configuration instance
config = PhenoCAIConfig()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Legacy compatibility - export variables for backward compatibility
PHENOCAI_PROJECT_ROOT = str(config.project_root)
PHENOCAI_CURRENT_STATION = config.current_station
PHENOCAI_CURRENT_INSTRUMENT = config.current_instrument
PHENOCAI_CURRENT_YEAR = config.current_year
PHENOCAI_DATA_DIR = str(config.data_dir)
MASTER_ANNOTATION_POOL_DIR = str(config.master_annotation_pool_dir)
EXPERIMENTAL_DATA_DIR = str(config.experimental_data_dir)
MASTER_DF_WITH_SPLITS_PATH = str(config.master_df_with_splits_path)
ANNOTATION_ROOT_DIR_FOR_HEURISTICS = str(config.annotation_root_dir_for_heuristics)
IMAGE_BASE_DIR = str(config.image_base_dir)
ROI_CONFIG_FILE_PATH = str(config.roi_config_file_path)
MODEL_OUTPUT_DIR = str(config.model_output_dir)
OUTPUT_DIR_FOR_NEW_ANNOTATIONS = str(config.output_dir_for_new_annotations)

# Model parameters
ROI_IMG_SIZE = config.roi_img_size
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
INITIAL_LR = config.initial_lr
FINE_TUNE_LR = config.fine_tune_lr
FINE_TUNE_EPOCHS = config.fine_tune_epochs
FINE_TUNE_AT_LAYER = config.fine_tune_at_layer
SKY_LINE_Y_RATIO_FOR_ROI00 = config.sky_line_y_ratio_for_roi00
PREDICTION_THRESHOLD = config.prediction_threshold

# Heuristic parameters
LOG_LABEL_SOURCE_INFO = config.log_label_source_info
SNOW_LOWER_HSV = config.snow_lower_hsv
SNOW_UPPER_HSV = config.snow_upper_hsv
SNOW_MIN_PIXEL_PERCENTAGE = config.snow_min_pixel_percentage
DISCARD_BLUR_THRESHOLD_ROI = config.discard_blur_threshold_roi
DISCARD_ILLUMINATION_DARK_THRESHOLD_ROI = config.discard_illumination_dark_threshold_roi
DISCARD_ILLUMINATION_BRIGHT_THRESHOLD_ROI = config.discard_illumination_bright_threshold_roi
DISCARD_LOW_CONTRAST_THRESHOLD_ROI = config.discard_low_contrast_threshold_roi

# Debug flags
DEBUG_ROI_LOOKUP = config.debug_roi_lookup
DEBUG_DATA_LOADING = config.debug_data_loading


def setup_directories():
    """Legacy function for directory setup."""
    config.setup_directories()


if __name__ == '__main__':
    # Setup directories
    config.setup_directories()
    
    # Validate configuration
    warnings = config.validate_configuration()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Display configuration
    print(f"Project Root: {config.project_root}")
    print(f"Current Station: {config.current_station}")
    print(f"Current Instrument: {config.current_instrument}")
    print(f"Current Year: {config.current_year}")
    print(f"Master Annotations CSV: {config.master_df_with_splits_path}")
    print(f"Model Output Dir: {config.model_output_dir}")
    
    # Save example configuration
    example_config_path = config.experimental_data_dir / 'example_config.yaml'
    config.save_to_yaml(example_config_path)
    print(f"\nExample configuration saved to: {example_config_path}")