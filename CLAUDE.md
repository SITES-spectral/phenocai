# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhenoCAI is a Python package for automated phenocam (phenological camera) image analysis and classification for agricultural and environmental monitoring across Swedish research stations. It uses machine learning to detect weather conditions, image quality issues, and environmental features in time-lapse camera images.

### Primary Stations
The project focuses on two main stations:
- **Lönnstorp (lonnstorp)**: Agricultural research station in southern Sweden
- **Röbäcksdalen (robacksdalen)**: Agricultural research station in northern Sweden

## Development Commands

```bash
# Install dependencies
uv sync

# Switch between stations
python scripts/switch_station.py lonnstorp  # Switch to Lönnstorp
python scripts/switch_station.py robacksdalen  # Switch to Röbäcksdalen

# Run configuration setup
source src/phenocai/config/env.sh  # Or use station-specific: env_lonnstorp.sh
python src/phenocai/config/setup.py

# Run the main application (once implemented)
uv run phenocai
```

## Architecture

The project follows a modular architecture with clear separation of concerns:

1. **Configuration Layer** (`src/phenocai/config/`)
   - `setup.py`: Central configuration management, paths, model parameters
   - `stations.yaml`: Station metadata and ROI (Region of Interest) definitions
   - `flags.yaml`: Quality flag taxonomy for image classification
   - `env.sh`: Environment variables for data paths

2. **Core Components** (planned modules):
   - `utils.py`: Image I/O, ROI extraction, filename parsing utilities
   - `data_management/`: Dataset creation, train/test splits, data loading
   - `heuristics/`: Rule-based detectors (snow, blur, darkness)
   - `models/`: CNN architectures (MobileNetV2 transfer learning + custom models)
   - `evaluation/`: Model performance metrics and validation
   - `inference/`: Apply trained models to new images

3. **Data Flow**:
   - Load station configuration and ROIs
   - Apply heuristics for initial labeling
   - Create train/test datasets with DuckDB/Polars
   - Train ML models with TensorFlow
   - Apply models to classify new images

## Key Technical Details

- **Python 3.12** with `uv` package manager
- **Image Processing**: OpenCV for ROI extraction and preprocessing
- **ML Framework**: TensorFlow for deep learning models
- **Data Management**: DuckDB for efficient dataset storage, Polars for data manipulation
- **Primary Stations**: Lönnstorp, Röbäcksdalen
- **Additional Stations**: Abisko, Grimsö, Skogaryd, Svartberget

## Configuration Notes

- Station configurations define camera positions, ROIs, and metadata in `stations.yaml`
- Quality flags in `flags.yaml` cover weather (snow, rain, fog), obstructions, and sensor issues
- The `setup.py` config class centralizes all paths and parameters
- Environment setup required via `env.sh` before running

## Memory Management Best Practices

### Array Operations
- Use in-place numpy operations with `out` and `where` parameters to minimize memory allocation
- Pre-allocate result arrays before processing
- Calculate denominators inline to avoid intermediate arrays
- Use `np.nan_to_num` with `copy=False` for in-place NaN handling

### Memory Cleanup
- Always use `del` statements after arrays are no longer needed
- Use context managers (`with` statements) for PIL Image objects
- Clean up intermediate arrays immediately after use
- Use try/finally blocks to ensure cleanup even on errors

### Batch Processing
- Process images in configurable batches (default 100) to manage memory usage
- Force garbage collection between batches with `gc.collect()`
- Track memory usage and adjust batch sizes dynamically
- Implement hierarchical progress tracking (processor → station → instrument)

### Storage Optimization
- Store vegetation indices as 8-bit (uint8) instead of 32-bit float (75% size reduction)
- Scale indices appropriately: [-1,1] → [0,255] or [0,1] → [0,255]
- Include color tables in TIFF files for immediate visual analysis
- Apply histogram equalization for enhanced contrast

## Database Guidelines

### Data Types
- **ALWAYS use BIGINT for file sizes and large counters** (never INTEGER - prevents overflow)
- Use GUIDs as PRIMARY KEYs for unique identification
- Follow strict data type guidelines to prevent overflow errors

### DuckDB Best Practices
- Use centralized configuration for connection settings
- Optimize based on system resources (memory, CPU cores)
- Use transaction manager for all database operations
- Implement batch operations with `executemany`
- Export to Parquet for analytics workloads

## Code Organization Principles

### Architecture
- Use abstract base classes for extensibility
- Implement clear separation of concerns
- Pass dependencies as parameters (dependency injection)
- Use dataclasses for data models

### Error Handling
- Create custom exception hierarchies
- Use structured logging with `get_logger`
- Implement comprehensive error isolation
- Track and report processing statistics

### Path Management
- Never hardcode paths - always use configuration
- Support both absolute and relative path resolution
- Use environment variables like `PHENOCAMS_BASE_DIR` for base directory configuration

## Performance Tracking

- Track processing time at multiple levels (total, per-station, per-file)
- Calculate throughput metrics (files/sec, MB/sec)
- Generate formatted performance reports
- Monitor memory usage throughout processing

## Dataset Fields

The dataset CSV files include the following fields:
- `image_filename`: Original image filename
- `image_id`: Unique identifier for the image
- `station`: Station name
- `instrument`: Instrument ID
- `year`, `day_of_year`: Temporal information
- `roi_name`: Region of interest identifier
- `discard`: Boolean for discarded ROIs
- `snow_presence`: Boolean for snow detection
- `flags`: Comma-separated quality flags
- `flag_count`: Number of quality flags
- `has_flags`: Boolean indicating presence of any quality flags (NEW)
- `split`: Train/test/val assignment

## Annotation File Schemas

PhenoCAI works with two types of annotation YAML files from the master annotation pool (`$PHENOCAI_MASTER_ANNOTATION_POOL_DIR`):

### 1. Daily Annotation Files (Legacy Format)
Named as `annotations_099.yaml` where `_099` is the day of year:

```yaml
created: '2025-05-15T22:19:03.447031'
day_of_year: '001'
station: "Lönnstorp"
instrument: LON_AGR_PL01_PHE01
annotation_time_minutes: 0.0
annotations:
  image_filename.jpg:
    - roi_name: ROI_00
      discard: false
      snow_presence: true
      flags: ['fog', 'high_brightness']  # Quality flags
    - roi_name: ROI_01
      discard: false
      snow_presence: false
      flags: []
```

### 2. Individual Annotation Files (Current Format)
Named as `lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003_annotations.yaml`:

```yaml
created: '2025-05-16T22:02:03.229554'
last_modified: '2025-05-17T14:32:11.901534'
filename: lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg
day_of_year: '102'
year: '2024'
station: lonnstorp
instrument: LON_AGR_PL01_PHE01
annotation_time_minutes: 52.54
status: in_progress  # or 'completed'
annotations:
  - roi_name: ROI_00  # Full image ROI
    discard: false
    snow_presence: false
    flags: []
    not_needed: false  # Separate field, not a flag
    _flag_selector: ''  # UI state field
  - roi_name: ROI_01  # Custom ROI
    discard: false
    snow_presence: true
    flags: ['snow', 'clouds']
```

### Annotation Fields
- **discard**: Boolean indicating if ROI should be excluded from analysis
- **snow_presence**: Boolean for snow detection
- **flags**: List of quality flags from `flags.yaml`
- **not_needed**: Boolean indicating no annotation required (not a flag)
- **status**: 'in_progress' or 'completed' (individual files only)

### Completion Criteria
An annotation is complete when at least one of:
- `discard` is true
- `snow_presence` is true
- `flags` list is not empty
- `not_needed` is true