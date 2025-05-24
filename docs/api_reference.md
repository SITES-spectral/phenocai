# PhenoCAI API Reference

## Core Modules

### phenocai.config

#### `PhenoCAIConfig`
Central configuration class for the PhenoCAI system with dynamic instrument support.

```python
from phenocai.config.setup import config

# Access configuration
config.current_station  # Current station name
config.current_instrument  # Current instrument ID
config.batch_size       # Training batch size
config.snow_min_pixel_percentage  # Snow detection threshold

# Setup directories
config.setup_directories()

# Validate configuration (now includes instrument validation)
warnings = config.validate_configuration()

# Dynamic station/instrument switching
config.switch_station('lonnstorp', 'LON_AGR_PL01_PHE02')
config.switch_instrument('LON_AGR_PL01_PHE01')

# List available instruments
instruments = config.list_available_instruments()
```

#### Station Registry
Access to stations.yaml validation and information.

```python
from phenocai.config.station_registry import get_registry

registry = get_registry()

# Validate instrument for station
is_valid = registry.validate_instrument('lonnstorp', 'LON_AGR_PL01_PHE01')

# Get station information
station_info = registry.get_station('lonnstorp')
print(f"Station: {station_info.name}")
print(f"Instruments: {station_info.active_instruments}")

# Get instrument details
instrument_info = registry.get_instrument_info('lonnstorp', 'LON_AGR_PL01_PHE01')
print(f"Ecosystem: {instrument_info.ecosystem}")
print(f"Viewing direction: {instrument_info.viewing_direction}")
```

### phenocai.utils

#### Image Processing Functions

```python
from phenocai.utils import (
    load_image,
    parse_image_filename,
    extract_roi_sub_image,
    load_roi_config_from_yaml,
    get_roi_points_from_config
)

# Load and process image
image = load_image('path/to/image.jpg', as_rgb=True)
metadata = parse_image_filename('lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg')

# Extract ROI
roi_config = load_roi_config_from_yaml('stations.yaml')
roi_points = get_roi_points_from_config(roi_config, 'lonnstorp', 'LON_AGR_PL01_PHE01', 'ROI_01')
roi_image = extract_roi_sub_image(image, roi_points)

# ROI_00 is pre-calculated in stations.yaml
roi_00_points = get_roi_points_from_config(roi_config, 'lonnstorp', 'LON_AGR_PL01_PHE01', 'ROI_00')
# ROI_00 excludes sky region automatically
```

### phenocai.data_management

#### Dataset Creation

```python
from phenocai.data_management import (
    create_master_annotation_dataframe,
    add_train_test_split,
    create_multi_station_dataset
)

# Create single-station dataset
df, stats = create_master_annotation_dataframe(
    annotation_dir='path/to/annotations',
    include_unannotated=False
)
# The dataset includes the following fields:
# - image_filename: Original filename (e.g., 'lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg')
# - file_path: Full path including day subdirectory (e.g., '/base/path/102/filename.jpg')
# - image_id: Unique identifier
# - station, instrument, year, day_of_year: Metadata
# - roi_name: Region of interest identifier
# - discard: Boolean for discarded ROIs
# - snow_presence: Boolean for snow detection
# - flags: Comma-separated quality flags
# - flag_count: Number of quality flags
# - has_flags: Boolean indicating presence of any quality flags
# - split: Train/test/val assignment (after calling add_train_test_split)

# Add train/test/val splits with grouped stratification
df_split = add_train_test_split(
    df, 
    test_size=0.2,      # 20% for test set
    val_size=0.1,       # 10% for validation (from training set)
    stratify_by='snow_presence',  # Maintain snow/no-snow ratio
    group_by_day=True,  # Keep same-day images together
    random_state=42     # For reproducibility
)
# Result: ~70% train, 20% test, 10% validation
# All sets maintain the same snow/no-snow proportion
# Same-day images are kept together in the same split

# Create multi-station dataset
df_multi, stats_dict = create_multi_station_dataset(
    stations=['lonnstorp', 'robacksdalen'],
    balance_stations=True
)
```

#### Annotation Loading

```python
from phenocai.data_management import (
    load_individual_annotation,
    load_daily_annotations,
    AnnotationData,
    ROIAnnotation
)

# Load individual annotation file
annotation = load_individual_annotation('path/to/annotation.yaml')

# Load daily annotations
daily_anns = load_daily_annotations('annotations_001.yaml')

# Access annotation data
for roi in annotation.annotations:
    print(f"{roi.roi_name}: snow={roi.snow_presence}, flags={roi.flags}")
```

### phenocai.heuristics

#### Snow Detection

```python
from phenocai.heuristics import detect_snow_hsv

# Basic snow detection
has_snow, snow_percentage = detect_snow_hsv(image)

# Custom thresholds
has_snow, snow_percentage = detect_snow_hsv(
    image,
    lower_hsv=np.array([0, 0, 170]),
    upper_hsv=np.array([180, 60, 255]),
    min_pixel_percentage=0.4
)
```

#### Image Quality Assessment

```python
from phenocai.heuristics import (
    detect_blur,
    detect_low_brightness,
    detect_high_brightness,
    should_discard_roi
)

# Check individual quality issues
is_blurry, blur_metric = detect_blur(image, threshold=70.0)
is_dark, brightness = detect_low_brightness(image, threshold=35)
is_bright, brightness = detect_high_brightness(image, threshold=230)

# Comprehensive quality check
should_discard, metrics = should_discard_roi(
    image,
    check_blur=True,
    check_brightness=True,
    check_contrast=True
)
```

## CLI Module

### Command Structure

```python
# Main CLI group
@click.group()
def cli():
    """PhenoCAI - Phenocam AI Analysis Tool"""
    pass

# Station commands
@cli.group()
def station():
    """Manage station configurations."""
    pass

@station.command()
def list():
    """List all available stations."""
    pass

@station.command()
@click.argument('station_name')
@click.option('--instrument', '-i')
def switch(station_name, instrument):
    """Switch station with optional instrument."""
    pass

@station.command()
def instruments():
    """List instruments for current station."""
    pass

# Dataset commands
@cli.group()
def dataset():
    """Create and manage datasets."""
    pass

@dataset.command()
@click.option('--output', '-o', type=click.Path())
@click.option('--include-unannotated', is_flag=True)
@click.option('--instrument', '-i', help='Instrument ID')
@click.option('--roi-filter', help='Only include specific ROI')
@click.option('--complete-rois-only/--no-complete-rois-only', default=True)
def create(output, include_unannotated, instrument, roi_filter, complete_rois_only):
    """Create dataset from annotations.
    
    Auto-generates filename if output not specified:
    - Format: {station}_{instrument}_dataset_{year}_splits_{test}_{val}.csv
    - Example: lonnstorp_PHE01_dataset_2024_splits_20_10.csv
    - Saved to: {station}/experimental_data/
    - Validates instrument against stations.yaml
    - --roi-filter ROI_00: For cross-station compatible datasets
    - --complete-rois-only: Only include images with all ROIs annotated
    """
    pass

@dataset.command()
@click.argument('input_path')
@click.argument('output_path', required=False)
@click.option('--no-flags', is_flag=True)
def filter(input_path, output_path, no_flags):
    """Filter dataset with intelligent naming.
    
    Auto-generates filename if output not specified:
    - Adds descriptors: clean, no_fog, stations_X, etc.
    - Example: dataset_clean_filtered.csv
    """
    pass
```

## Data Classes

### AnnotationData

```python
@dataclass
class AnnotationData:
    filename: str
    station: str
    instrument: str
    year: int
    day_of_year: int
    annotations: List[ROIAnnotation]
    created: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    annotation_time_minutes: float = 0.0
    status: str = 'pending'
    
    @property
    def is_complete(self) -> bool:
        """Check if all ROIs are annotated."""
        return all(roi.is_annotated for roi in self.annotations)
```

### ROIAnnotation

```python
@dataclass
class ROIAnnotation:
    roi_name: str
    discard: bool = False
    snow_presence: bool = False
    flags: List[str] = field(default_factory=list)
    not_needed: bool = False
    
    @property
    def is_annotated(self) -> bool:
        """Check if ROI has any annotation."""
        return (
            self.discard or 
            self.snow_presence or 
            len(self.flags) > 0 or 
            self.not_needed
        )
```

### ImageMetadata

```python
@dataclass
class ImageMetadata:
    station: str
    instrument: str
    year: int
    day_of_year: int
    date_str: str
    time_str: str
    full_datetime: datetime
    filename: str
    
    @property
    def image_id(self) -> str:
        """Generate unique image ID."""
        return f"{self.station}_{self.instrument}_{self.year}_{self.day_of_year:03d}_{self.time_str}"
```

## phenocai.inference

### Model Prediction

```python
from phenocai.inference import ModelPredictor, BatchPredictor, PredictionResult

# Initialize predictor
predictor = ModelPredictor(
    model_path='trained_models/mobilenet/final_model.h5',
    threshold=0.5,
    batch_size=32,
    use_heuristics=True
)

# Predict single image
predictions = predictor.predict_image('path/to/image.jpg')
for pred in predictions:
    print(f"{pred.roi_name}: {pred.snow_presence} (confidence: {pred.confidence:.2f})")
    if pred.discard:
        print(f"  Discard recommended due to: {pred.quality_flags}")

# Batch processing
batch_predictor = BatchPredictor(predictor)
stats = batch_predictor.process_directory(
    directory='images/',
    output_dir='predictions/',
    output_format='yaml'  # or 'csv', 'json'
)

# Process date range
stats = batch_predictor.process_date_range(
    start_day=100,
    end_day=200,
    year=2024,
    output_dir='seasonal_predictions/'
)
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    filename: str
    roi_name: str
    snow_probability: float
    snow_presence: bool
    confidence: float
    quality_flags: List[str]
    discard: bool
    has_flags: bool
    processing_time: float
```

### Convenience Functions

```python
from phenocai.inference import (
    process_single_image,
    process_image_directory,
    process_date_range
)

# Quick single image prediction
results = process_single_image(
    model_path='model.h5',
    image_path='image.jpg',
    threshold=0.5
)

# Process entire directory
stats = process_image_directory(
    model_path='model.h5',
    directory='images/',
    output_dir='results/',
    output_format='csv'
)

# Process date range
stats = process_date_range(
    model_path='model.h5',
    start_day=1,
    end_day=365,
    output_dir='year_predictions/',
    year=2024
)
```

## Station Configuration

### Available Stations

```python
from phenocai.config.station_configs import (
    get_all_stations,
    get_primary_stations,
    get_station_config
)

# Get station lists
all_stations = get_all_stations()  # All 6 stations
primary = get_primary_stations()   # ['lonnstorp', 'robacksdalen']

# Get station details
config = get_station_config('lonnstorp')
# Returns: {
#     'full_name': 'Lönnstorp',
#     'station_code': 'LON',
#     'default_instrument': 'LON_AGR_PL01_PHE01',
#     'instruments': [...],
#     'typical_rois': ['ROI_00', 'ROI_01', ...],
#     'latitude': 55.6686,
#     'longitude': 13.1073
# }
```

## Usage Examples

### Complete Dataset Creation Pipeline

```python
from phenocai.config.setup import config
from phenocai.data_management import create_master_annotation_dataframe, add_train_test_split
from phenocai.heuristics import detect_snow_hsv
import pandas as pd

# 1. Setup configuration
config.setup_directories()

# 2. Create dataset
df, stats = create_master_annotation_dataframe(
    config.master_annotation_pool_dir,
    include_unannotated=False
)
# Note: The 'file_path' column contains the full path to images,
# including day-of-year subdirectories (e.g., /base/path/102/filename.jpg)

# 3. Add splits
df_split = add_train_test_split(df, test_size=0.2, val_size=0.1, group_by_day=True)

# 4. Filter by quality
clean_df = df_split[~df_split['has_flags']]

# 5. Save
clean_df.to_csv('training_data.csv', index=False)

# 6. Access file paths for training
for _, row in clean_df[clean_df['split'] == 'train'].iterrows():
    image_path = row['file_path']  # Full path including day subdirectory
    # Load and process image...
```

### Quality Analysis

```python
# Analyze quality issues
flagged = df[df['has_flags']]
flag_counts = flagged['flags'].str.split(',').explode().value_counts()

# Filter specific flags
no_fog = df[~df['flags'].str.contains('fog', na=False)]
no_blur = df[~df['flags'].str.contains('blur', na=False)]

# Get clean samples per ROI
clean_by_roi = df[~df['has_flags']].groupby('roi_name').size()
```

## Error Handling

All functions include comprehensive error handling:

```python
try:
    image = load_image('path/to/image.jpg')
except FileNotFoundError:
    print("Image not found")
except IOError as e:
    print(f"Failed to load image: {e}")

try:
    metadata = parse_image_filename('invalid_filename.jpg')
except ValueError as e:
    print(f"Invalid filename format: {e}")
```

## phenocai.roi_calculator

### ROI_00 Calculation and Management

```python
from phenocai.roi_calculator import (
    calculate_roi_00,
    serialize_polygons,
    deserialize_polygons,
    add_roi_00_to_station_config
)

# Calculate ROI_00 for an image
import cv2
image = cv2.imread('path/to/image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate using gradient-based sky detection
roi_00_points = calculate_roi_00(image_rgb, horizon_method='gradient')
# Returns: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

# Serialize/deserialize for YAML storage
roi_data = {
    'ROI_00': {
        'points': roi_00_points,
        'color': (255, 255, 255),
        'thickness': 7,
        'description': 'Full image excluding sky',
        'auto_generated': True
    }
}

# Convert to YAML-friendly format (tuples → lists)
yaml_data = serialize_polygons(roi_data)

# Convert back from YAML (lists → tuples)
original_data = deserialize_polygons(yaml_data)

# Add ROI_00 to station configuration
from pathlib import Path
add_roi_00_to_station_config(
    Path('stations.yaml'),
    'lonnstorp',
    'LON_AGR_PL01_PHE01',
    sample_image_path=Path('sample.jpg'),
    horizon_method='gradient',
    force=True
)
```

### CLI Commands for ROI_00

```python
@config_cmd.command()
@click.option('--sample-image', type=click.Path(exists=True))
@click.option('--station', help='Specific station to update')
@click.option('--instrument', help='Specific instrument to update')
@click.option('--method', type=click.Choice(['gradient', 'color', 'fixed']), default='gradient')
@click.option('--force', is_flag=True, help='Force recalculation')
def add_roi_00(sample_image, station, instrument, method, force):
    """Add ROI_00 to stations.yaml configuration."""
    pass
```

## Performance Considerations

1. **Batch Processing**: Process images in batches to manage memory
2. **Garbage Collection**: Enabled by default between batches
3. **In-place Operations**: Use numpy's `out` parameter for efficiency
4. **Array Cleanup**: Always delete large arrays after use
5. **ROI_00 Caching**: Pre-calculated in stations.yaml for performance

```python
# Memory-efficient processing
for batch in batched(images, batch_size=100):
    process_batch(batch)
    gc.collect()  # Force garbage collection
```