# PhenoCAI Training Guide

This guide explains how to train models for Lönnstorp (or Röbäcksdalen) using the PhenoCAI system.

## Prerequisites

1. Ensure you have the environment set up:
```bash
cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai
source src/phenocai/config/env.sh
uv sync
```

2. Configure ROI_00 for cross-station compatibility:
```bash
# Add ROI_00 (full image excluding sky) to all station configurations
uv run phenocai config add-roi-00

# ROI_00 is automatically calculated using advanced sky detection
# algorithms from phenotag/phenocams packages for consistent cross-station analysis
```

3. Verify your current station and instrument:
```bash
uv run phenocai info

# List available instruments for current station
uv run phenocai station instruments

# Switch to specific instrument if needed
uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE02
```

## Step 1: Prepare Training Data

First, create the dataset with train/test/validation splits:

```bash
# Create dataset for current station/instrument (auto-named)
uv run phenocai dataset create
# Creates: lonnstorp_PHE01_dataset_2024_splits_20_10.csv (if multiple instruments)

# Create dataset for specific instrument
uv run phenocai dataset create --instrument LON_AGR_PL01_PHE02
# Creates: lonnstorp_PHE02_dataset_2024_splits_20_10.csv

# Create dataset with only ROI_00 (for cross-station work)
uv run phenocai dataset create --roi-filter ROI_00
# Creates: lonnstorp_PHE01_dataset_2024_roi_00_splits_20_10.csv

# Create with custom name
uv run phenocai dataset create --output lonnstorp_dataset.csv

# Or using the preparation script (recommended)
python scripts/01_prepare_training.py --station lonnstorp
```

This will:
- Load all annotations from the master annotation pool
- Create train/test/validation splits (70%/20%/10%)
- Save the dataset to the experimental data directory
- Add `has_flags` field for easy quality filtering

### Analyze Quality Issues

Before training, analyze the quality of your data:

```bash
# Detailed quality analysis
python scripts/analyze_quality_issues.py lonnstorp_dataset.csv

# Filter out problematic images
uv run phenocai dataset filter lonnstorp_dataset.csv lonnstorp_clean.csv \
    --exclude-flags fog high_brightness lens_water_drops
```

### Optional: Apply Heuristics

To see how well simple heuristics perform:

```bash
python scripts/01_prepare_training.py --station lonnstorp --apply-heuristics --sample-size 100
```

This will apply HSV-based snow detection and image quality checks to a sample of images.

## Step 2: Explore the Data

Check the dataset statistics:

```bash
uv run phenocai dataset info lonnstorp_dataset_with_splits.csv
```

## Step 3: Train Models

Train your models using the prepared dataset:

```bash
# Train MobileNetV2 model (recommended)
uv run phenocai train model lonnstorp_dataset_with_splits.csv \
    --model-type mobilenet \
    --epochs 20 \
    --batch-size 32

# Train simple CNN (faster, lower accuracy)
uv run phenocai train model lonnstorp_dataset_with_splits.csv \
    --model-type simple-cnn \
    --epochs 30 \
    --batch-size 64

# Train with custom parameters
uv run phenocai train model lonnstorp_dataset_with_splits.csv \
    --model-type mobilenet \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --output-dir trained_models/custom/
```

The training process will:
- Load and preprocess images automatically
- Apply data augmentation to training samples
- Save the best model based on validation accuracy
- Generate training history plots
- Save model checkpoints during training

## Current Status

### Completed:
- Dataset creation and management with ROI_00 support
- Train/test/validation splitting with grouped stratification
- Heuristic methods for snow detection and quality assessment
- CLI infrastructure with dynamic instrument validation
- Multi-station support with cross-station capabilities
- Model architectures (MobileNetV2, Custom CNN)
- Complete training pipeline with TensorFlow
- Data augmentation strategies
- Evaluation metrics and visualization
- Full inference/prediction pipeline
- Batch processing for directories and date ranges
- Multi-format export (YAML, CSV, JSON)
- ROI_00 automatic calculation and storage
- Cross-station training and evaluation

### In Progress:
- Performance optimization
- Additional model architectures

### TODO:
- Real-time prediction API
- Model versioning system
- Distributed training support

## Dataset Overview for Lönnstorp

Based on the current data:
- **Total images**: 1,467
- **Total ROI annotations**: 5,559
- **Snow present**: 756 cases (13.6%)
- **Discarded ROIs**: 1,326 cases (23.8%)
- **Images with quality flags**: 89.9%
- **Common quality issues**: high_brightness (35.9%), fog (26.1%), sunny (9.7%)

### Filtered Dataset Statistics
After removing fog, high_brightness, and lens_water_drops:
- **Total annotations**: 2,163 (61% reduction)
- **Snow present**: 31.6% (better class balance)
- **Clean samples**: 26.1% (vs 10.1% in full dataset)

## ROI Distribution
- ROI_00 (full image): 1,460 annotations
- ROI_01: 1,231 annotations
- ROI_02: 944 annotations
- ROI_03: 957 annotations
- ROI_06: 967 annotations

## Next Steps

1. **Review the data**: Examine the CSV files to understand the annotation distribution
2. **Implement models**: The model architectures need to be implemented in `src/phenocai/models/`
3. **Create training pipeline**: Implement the actual training loop with TensorFlow
4. **Add data augmentation**: Implement augmentation strategies for better generalization
5. **Evaluate performance**: Create evaluation scripts to assess model performance

## Tips for Training

1. **Start simple**: Begin with the heuristics to establish a baseline
2. **Monitor class imbalance**: Snow is present in only ~14% of cases, so consider:
   - Class weighting
   - Oversampling minority class
   - Using appropriate metrics (precision, recall, F1)
3. **Use appropriate ROIs**: ROI_00 is the full image, others are specific regions
4. **Consider temporal aspects**: Images from the same day should stay in the same split
5. **Memory management**: Process images in batches to avoid memory issues

## Multi-Station Training

To train on both Lönnstorp and Röbäcksdalen:

```bash
# Create multi-station dataset
uv run phenocai dataset multi-station \
    --stations lonnstorp robacksdalen \
    --output multi_station_dataset.csv

# Train on multi-station data
uv run phenocai train model multi_station_dataset.csv \
    --model-type mobilenet \
    --epochs 30 \
    --batch-size 32
```

## Cross-Station Training with ROI_00

ROI_00 enables training on one station and applying to others:

```bash
# Train on Lönnstorp with ROI_00 only
uv run phenocai dataset create --roi-filter ROI_00
uv run phenocai train model lonnstorp_roi_00_dataset.csv --preset mobilenet_full

# Switch to Röbäcksdalen and evaluate
uv run phenocai station switch robacksdalen
uv run phenocai dataset create --roi-filter ROI_00
uv run phenocai evaluate model /path/to/model.h5 robacksdalen_roi_00_dataset.csv

# Use the cross-station pipeline for automation
uv run phenocai cross-station pipeline \
    --train-stations lonnstorp \
    --eval-stations robacksdalen abisko \
    --roi-filter ROI_00
```

### Why ROI_00 for Cross-Station Work?

- **Universal Coverage**: ROI_00 represents the full image minus sky
- **Automatic Calculation**: Sky exclusion using advanced HSV detection
- **Pre-calculated**: Stored in stations.yaml for performance
- **Consistent View**: Same perspective across different camera angles

## Step 4: Evaluate Your Model

After training, evaluate model performance:

```bash
# Evaluate on test set
uv run phenocai evaluate model trained_models/mobilenet/final_model.h5 \
    lonnstorp_dataset_with_splits.csv \
    --split test

# Generate detailed metrics
uv run phenocai evaluate model trained_models/mobilenet/final_model.h5 \
    lonnstorp_dataset_with_splits.csv \
    --save-predictions \
    --generate-plots
```

## Step 5: Apply Model to New Images

Use your trained model to predict snow presence in new images:

```bash
# Predict single image
uv run phenocai predict apply trained_models/mobilenet/final_model.h5 \
    --image /path/to/new/image.jpg

# Process entire year of data
uv run phenocai predict batch trained_models/mobilenet/final_model.h5 \
    --year 2024 \
    --output-dir predictions/2024/ \
    --format yaml \
    --use-heuristics

# Export results for analysis
uv run phenocai predict export predictions/2024/ \
    --format csv \
    --output lonnstorp_2024_predictions.csv
```

The prediction system will:
- Detect snow presence with confidence scores
- Apply quality heuristics to flag issues
- Determine if images should be discarded
- Process images in efficient batches
- Save results in your preferred format