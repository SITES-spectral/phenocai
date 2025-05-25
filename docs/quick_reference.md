# PhenoCAI Quick Reference

## Common Commands

### Setup
```bash
# Install dependencies
uv sync

# Check configuration
uv run phenocai info

# Initialize directories
uv run phenocai config init

# Add ROI_00 to all stations (one-time setup for cross-station compatibility)
uv run phenocai config add-roi-00
```

### Station Management
```bash
# List all stations
uv run phenocai station list

# Switch station (uses default instrument)
uv run phenocai station switch lonnstorp

# Switch station with specific instrument
uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE02

# List instruments for current station
uv run phenocai station instruments

# Get station details (shows all instruments)
uv run phenocai station info lonnstorp
```

### Dataset Creation
```bash
# Create single-station dataset (auto-named)
uv run phenocai dataset create
# Creates: lonnstorp/experimental_data/lonnstorp_dataset_2024_splits_20_10.csv
# If multiple instruments: lonnstorp_PHE01_dataset_2024_splits_20_10.csv

# Create with specific instrument
uv run phenocai dataset create --instrument LON_AGR_PL01_PHE02
# Creates: lonnstorp_PHE02_dataset_2024_splits_20_10.csv

# Create with custom output path
uv run phenocai dataset create --output custom_name.csv

# Create dataset with only ROI_00 (for cross-station work)
uv run phenocai dataset create --roi-filter ROI_00
# Creates: lonnstorp_PHE01_dataset_2024_roi_00_splits_20_10.csv

# Create dataset with all images (no ROI completeness filtering)
uv run phenocai dataset create --no-complete-rois-only

# Create multi-station dataset (auto-named)
uv run phenocai dataset multi-station \
    --stations lonnstorp robacksdalen
# Creates: experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv

# Get dataset information
uv run phenocai dataset info dataset.csv
```

### Quality Analysis and Filtering
```bash
# Analyze quality issues
python scripts/analyze_quality_issues.py dataset.csv

# Filter dataset (auto-named output)
uv run phenocai dataset filter input.csv \
    --exclude-flags fog high_brightness
# Creates: input_no_fog_high_brightness_filtered.csv

# Filter with explicit output
uv run phenocai dataset filter input.csv output.csv \
    --min-year 2024 \
    --rois ROI_00 ROI_01

# Create clean dataset (no quality flags)
uv run phenocai dataset filter input.csv --no-flags
# Creates: input_clean_filtered.csv
```

### Dataset Preparation & Splitting
```bash
# Create dataset with automatic train/test/val splits (70/20/10 by default)
source src/phenocai/config/env.sh
uv run phenocai dataset create --output dataset_with_splits.csv

# Custom split ratios with grouped stratification
uv run phenocai dataset create \
    --output dataset.csv \
    --test-size 0.2 \
    --val-size 0.1
# Note: Images from the same day are kept together in the same split

### Cross-Station Evaluation
```bash
# Setup ROI_00 for cross-station compatibility
uv run phenocai config add-roi-00

# Create ROI_00 dataset for training (use all data)
uv run phenocai dataset create --roi-filter ROI_00 --test-size 0.0 --val-size 0.2

# Switch station for evaluation
uv run phenocai station switch robacksdalen
uv run phenocai dataset create --roi-filter ROI_00

# Note: Use regular evaluate command for cross-station evaluation
# Train on one station, then evaluate on another station's dataset

# Complete cross-station pipeline with annotation generation  
uv run phenocai cross-station pipeline \
    --train-stations lonnstorp \
    --eval-stations robacksdalen abisko \
    --years 2023 2024 \
    --annotation-years 2022 2025 \
    --use-heuristics
```
# The file_path field includes day-of-year subdirectories
```

### Model Training
```bash
# Quick training with MobileNetV2
uv run phenocai train model dataset.csv --preset mobilenet_quick

# Full training
uv run phenocai train model dataset.csv --preset mobilenet_full

# Custom training parameters
uv run phenocai train model dataset.csv \
    --model-type mobilenet \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001

# List available presets
uv run phenocai train list-presets
```

### Complete Pipeline
```bash
# Run entire pipeline: dataset → training → evaluation → prediction
uv run phenocai pipeline full
# Uses defaults: mobilenet_full, clean data, 2023+2024 predictions

# Customized complete pipeline
uv run phenocai pipeline full \
    --station lonnstorp \
    --instrument LON_AGR_PL01_PHE01 \
    --year 2024 \
    --prediction-years 2023 2024 2025 \
    --model-type mobilenet \
    --preset mobilenet_full \
    --clean-only

# Check pipeline status
uv run phenocai pipeline status --station lonnstorp

# Dry run (preview without executing)
uv run phenocai pipeline full --dry-run
```

### Cross-Station Pipeline
```bash
# Train on Lönnstorp, evaluate on Röbäcksdalen using ROI_00
uv run phenocai cross-station pipeline \
    --train-stations lonnstorp \
    --eval-stations robacksdalen \
    --years 2023 2024 \
    --roi-filter ROI_00

# Multi-station training and evaluation
uv run phenocai cross-station pipeline \
    --train-stations lonnstorp robacksdalen \
    --eval-stations abisko grimso \
    --annotation-years 2022 2023 2024 2025 \
    --use-heuristics
```

### Model Evaluation
```bash
# Evaluate on test set
uv run phenocai evaluate model trained_models/mobilenet/final_model.h5 dataset.csv

# Compare with manual annotations
uv run phenocai evaluate compare dataset.csv predictions.json

# Benchmark multiple models
uv run phenocai evaluate benchmark --dataset-path dataset.csv
```

### Prediction/Inference
```bash
# Predict single image with quality detection
uv run phenocai predict apply model.h5 \
    --image path/to/image.jpg

# Batch process directory with heuristics
uv run phenocai predict batch model.h5 \
    --directory /2024/ \
    --output-dir predictions/ \
    --format yaml \
    --use-heuristics

# Process specific date range
uv run phenocai predict batch model.h5 \
    --start-day 100 \
    --end-day 200 \
    --year 2024 \
    --output-dir seasonal_predictions/

# Export predictions to multiple formats
uv run phenocai predict export predictions/ \
    --format csv \
    --output results.csv
# Supported formats: yaml, csv, json
```

### ROI_00 Configuration
```bash
# Add ROI_00 to all primary stations
uv run phenocai config add-roi-00

# Add ROI_00 with specific sample image
uv run phenocai config add-roi-00 \
    --sample-image /path/to/sample.jpg

# Add ROI_00 to specific instrument
uv run phenocai config add-roi-00 \
    --station lonnstorp \
    --instrument LON_AGR_PL01_PHE01 \
    --force

# Choose horizon detection method
uv run phenocai config add-roi-00 \
    --method gradient  # or 'color', 'fixed'
```

## Quality Flags Reference

### Weather Conditions
- `fog` - Foggy conditions
- `clouds` / `cloudy` - Cloud coverage
- `rain` / `heavy_rain` - Precipitation
- `snow` / `lens_snow` - Snow conditions
- `sunny` - Bright sunny conditions

### Illumination Issues
- `high_brightness` - Overexposed areas
- `low_brightness` - Underexposed/dark
- `shadows` - Strong shadows present
- `heterogeneous_illumination` - Uneven lighting
- `glare` - Light reflection issues
- `sun_altitude_low_20deg` - Low sun angle

### Lens Problems
- `lens_dirt` - Dirty lens
- `lens_water_drops` - Water droplets on lens
- `lens_ice` - Ice formation on lens

### Image Quality
- `blur` - Blurry/out of focus
- `bluish_dominated` - Color cast issues
- `haze` - Atmospheric haze
- `unusable` - Generally unusable image

### Other
- `birds` - Birds in frame
- `small_wildlife` / `large_wildlife` - Animals present
- `land_management_practice` - Human activity
- `wet_patches` - Wet areas visible

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_filename` | str | Original image filename |
| `file_path` | str | Full path including day subdirectory (e.g., /base/102/image.jpg) |
| `image_id` | str | Unique image identifier |
| `station` | str | Station name |
| `instrument` | str | Instrument ID |
| `year` | int | Year |
| `day_of_year` | int | Day of year (1-365) |
| `roi_name` | str | ROI identifier (ROI_00, ROI_01, etc.) |
| `discard` | bool | Should ROI be discarded |
| `snow_presence` | bool | Snow detected |
| `flags` | str | Comma-separated quality flags |
| `flag_count` | int | Number of flags |
| `has_flags` | bool | Any quality issues present |
| `split` | str | train/test/val assignment |
| `not_needed` | bool | No annotation needed flag |

### ROI_00 Special Properties
- **Definition**: Full image excluding sky region
- **Calculation**: Automatic using HSV-based sky detection
- **Storage**: Pre-calculated in stations.yaml for efficiency
- **Use case**: Cross-station model compatibility

### Additional Prediction Fields
| Field | Type | Description |
|-------|------|-------------|
| `snow_probability` | float | Model confidence (0-1) |
| `confidence` | float | Prediction certainty |
| `model_predicted` | bool | Generated by model |
| `processing_time` | float | Inference time (seconds) |

## Interactive Notebook (Marimo)

```bash
# Start interactive pipeline notebook
marimo edit notebooks/phenocai_pipeline.py

# Or run in view mode
marimo run notebooks/phenocai_pipeline.py

# Features:
# - Interactive station/instrument selection
# - Visual pipeline configuration
# - Real-time execution monitoring
# - Results browsing and analysis
```

## Working with Multiple Instruments

```bash
# Check available instruments for a station
uv run phenocai station info lonnstorp

# Switch to specific instrument
uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE02

# Create dataset for specific instrument
uv run phenocai dataset create --instrument LON_AGR_PL01_PHE01

# The system validates instruments against stations.yaml
# Invalid instruments will show available options
```

## Environment Variables

```bash
export PHENOCAI_CURRENT_STATION="lonnstorp"
export PHENOCAI_CURRENT_INSTRUMENT="LON_AGR_PL01_PHE01"
export PHENOCAI_CURRENT_YEAR="2024"
export PHENOCAI_LOG_LEVEL="INFO"  # or DEBUG

# Environment variables are updated dynamically when switching
```

## File Paths

### Lönnstorp
- Images: `/home/jobelund/lu2024-12-46/SITES/Spectral/data/lonnstorp/phenocams/products/LON_AGR_PL01_PHE01/L1/2024/`
- Annotations: `/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/master_annotation_pool/`

### Röbäcksdalen
- Images: `/home/jobelund/lu2024-12-46/SITES/Spectral/data/robacksdalen/phenocams/products/RBD_AGR_PL01_PHE01/L1/2024/`
- Annotations: `/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/robacksdalen/master_annotation_pool/`

## Tips

1. **Start with filtered data**: Remove fog, high_brightness, and lens_water_drops for cleaner training data
2. **Check class balance**: Use `has_flags` to separate clean vs problematic samples
3. **Use appropriate batch sizes**: Default is 100 for memory efficiency
4. **Monitor splits**: Ensure train/test/val have similar distributions
5. **Use ROI_00 for cross-station work**: Provides consistent view across different camera angles
6. **Configure ROI_00 once**: Run `uv run phenocai config add-roi-00` for automatic sky exclusion

## Troubleshooting

### Memory Issues
- Reduce batch size in configuration
- Process stations separately
- Use filtered datasets

### Missing Annotations
- Check `PHENOCAI_MASTER_ANNOTATION_POOL_DIR` is set correctly
- Verify annotation files exist for the specified year
- Use `uv run phenocai config validate`

### Configuration Problems
- Run `source src/phenocai/config/env.sh` to load environment
- Check with `uv run phenocai config show`
- Validate with `uv run phenocai config validate`