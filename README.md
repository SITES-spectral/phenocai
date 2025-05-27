---
![SITES Spectral Thematic Center](https://h24-original.s3.amazonaws.com/231546/28893673-EQhe9.png "SITES Spectral Thematic Center")

# Swedish Infrastructure for Ecosystem Science (SITES) - Spectral | Thematic Center (SSTC)

["SITES spectral"](https://www.fieldsites.se/en-GB/sites-thematic-programs/sites-spectral-32634403)

## Maintainers

* José M. Beltrán-Abaunza, PhD | Lund University, Department of Physical Geography and Ecosystem Science | SITES spectral Research Engineer

## Contributors

* José M. Beltrán-Abaunza, PhD | Lund University, Department of Physical Geography and Ecosystem Science | SITES spectral Research Engineer
* Lars Eklundh, Professor | Lund University, Department of Physical Geography and Ecosystem Science | SITES spectral Coordinator
* Kexin Guo | Lund University, Department of Physical Geography and Ecosystem Science | Bachelor program 2022-2025 | Supported UX and code alpha testing, data analysis and codebase debugging. Thesis to be linked here.

## Development Support

This package was developed with support from [Claude](https://claude.ai/code) (Anthropic's Claude 3.5 Sonnet model), which assisted with code refactoring, documentation organization, and architectural improvements.

## Citation

If you use this package in your research, please cite it as follows:

**Chicago Style (Author-Date):**

Beltrán-Abaunza, José M., and Lars Eklundh. *PhenoCAI: A Python Package for Automated Phenological Camera Image Analysis*. Version [version]. Lund: SITES Spectral Thematic Center, Lund University, 2025. https://github.com/sites-spectral/phenocai.

**Chicago Style (Notes-Bibliography):**

Beltrán-Abaunza, José M., and Lars Eklundh. *PhenoCAI: A Python Package for Automated Phenological Camera Image Analysis*. Version [version]. Lund: SITES Spectral Thematic Center, Lund University, 2025. https://github.com/sites-spectral/phenocai.

**BibTeX:**
```bibtex
@software{beltran_abaunza_phenocai_2025,
  author = {Beltrán-Abaunza, José M. and Eklundh, Lars},
  title = {PhenoCAI: A Python Package for Automated Phenological Camera Image Analysis},
  year = {2025},
  publisher = {SITES Spectral Thematic Center, Lund University},
  address = {Lund, Sweden},
  url = {https://github.com/sites-spectral/phenocai},
  note = {Version [version]}
}
```

*Replace [version] with the specific version number you used.*

# PhenoCAI

Automated phenological camera image analysis using AI for environmental monitoring.

## Overview

PhenoCAI (Phenological Camera AI) is a Python package for automated analysis and classification of phenocam images from the SITES (Swedish Infrastructure for Ecosystem Science) research stations. It uses machine learning to detect environmental conditions like snow presence and image quality issues.

### Primary Stations
- **Lönnstorp**: Agricultural research station in southern Sweden
- **Röbäcksdalen**: Agricultural research station in northern Sweden

## Features

- **Automated phenocam image analysis** with ROI-based processing
- **Snow detection** using both heuristics and deep learning
- **ROI_00 support** for full-image analysis excluding sky regions
- **Multi-station dataset management** with automatic file path resolution
- **Quality flag detection** and filtering (23+ flag types)
- **Transfer learning** with MobileNetV2 and custom CNNs
- **Comprehensive evaluation** metrics and visualizations
- **Memory-efficient** batch processing
- **Complete training pipeline** with callbacks and monitoring
- **Annotation format conversion** (daily to individual)
- **Educational documentation** with workflow diagrams
- **Production-Ready Prediction** system for processing entire years of data
- **Grouped Stratified Splitting** for robust train/test/val datasets
- **Intelligent Dataset Naming** with station, instrument, and configuration details
- **Dynamic Instrument Validation** against stations.yaml configuration
- **Cross-station compatibility** through ROI_00 standardization

## What's New

### Dataset Management
- **Smart Dataset Balancing**: Automatic balancing of snow/no-snow samples with configurable ratios
- **Threshold Optimization**: Find optimal prediction thresholds using validation data
- **Enhanced Cross-Station Support**: Train and evaluate across different locations

### Processing Capabilities
- **Multi-Year Processing**: Process historical data efficiently
- **Batch Prediction Pipeline**: Optimized for processing large volumes of images
- **Confidence-Based Filtering**: Apply custom thresholds for production use

### Features
- **ROI_00 Automatic Calculation**: Advanced sky detection algorithm
- **Cross-Station Training**: Train on one station and evaluate on others
- **Complete Pipeline Command**: Single command automation
- **Interactive Marimo Notebook**: Visual pipeline execution
- **Annotation Generation**: Create annotations using ML and heuristics

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd phenocai

# Install dependencies using uv
uv sync

# Set up environment
source src/phenocai/config/env.sh
```

## Quick Start

### Complete Pipeline (Recommended)
```bash
# Run the entire pipeline with one command
uv run phenocai pipeline full

# Or with custom settings
uv run phenocai pipeline full \
    --station lonnstorp \
    --instrument LON_AGR_PL01_PHE01 \
    --year 2024 \
    --prediction-years 2023 2024 2025 \
    --clean-only

# Interactive notebook (recommended for beginners)
marimo edit notebooks/phenocai_pipeline.py
```

### Step-by-Step Setup
```bash
# Check system configuration
uv run phenocai info

# Initialize configuration
uv run phenocai config init

# List available stations
uv run phenocai station list

# Switch station with specific instrument
uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE01

# List instruments for current station
uv run phenocai station instruments

# Add ROI_00 to stations configuration
uv run phenocai config add-roi-00
```

### 2. Data Preparation
```bash
# Convert daily annotations to individual format
uv run phenocai convert all-stations

# Create training dataset
uv run phenocai dataset create --output my_dataset.csv

# Analyze dataset quality
uv run phenocai train analyze-dataset my_dataset.csv

# Filter problematic images
uv run phenocai dataset filter my_dataset.csv clean_dataset.csv \
    --exclude-flags fog high_brightness lens_water_drops

# Balance dataset for better model performance
uv run phenocai dataset balance my_dataset.csv balanced_dataset.csv \
    --target-ratio 0.5  # 50/50 snow/no-snow
```

### 3. Training Models
```bash
# Quick training with preset
uv run phenocai train model my_dataset.csv --preset mobilenet_quick

# Custom training configuration
uv run phenocai train model my_dataset.csv \
    --model-type custom_cnn \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# Fine-tune existing model
uv run phenocai train fine-tune models/mobilenet_v2_lonnstorp my_dataset.csv
```

### 4. Model Evaluation
```bash
# Evaluate single model
uv run phenocai evaluate model models/best_model.h5 test_dataset.csv \
    --plot-samples 16 \
    --analyze-errors

# Compare multiple models
uv run phenocai evaluate benchmark --dataset-path test_dataset.csv

# Compare predictions with ground truth
uv run phenocai evaluate compare dataset.csv predictions.json
```

### 5. Heuristic Analysis
```bash
# Analyze individual image
uv run phenocai analyze detect-snow image.jpg --visualize

# Assess image quality
uv run phenocai analyze assess-quality image.jpg

# Analyze entire dataset
uv run phenocai analyze analyze-dataset dataset.csv --sample-size 200
```

## CLI Reference

### Dataset Commands
```bash
# Dataset creation and management
uv run phenocai dataset create [OPTIONS]
uv run phenocai dataset create --roi-filter ROI_00  # Cross-station compatible
uv run phenocai dataset create --complete-rois-only  # Only images with all ROIs
uv run phenocai dataset multi-station --stations lonnstorp robacksdalen
uv run phenocai dataset filter INPUT OUTPUT [OPTIONS]
uv run phenocai dataset info DATASET_PATH
uv run phenocai dataset balance INPUT OUTPUT [OPTIONS]  # Balance snow/no-snow ratio
```

### Training Commands
```bash
# Model training
uv run phenocai train model DATASET_PATH [OPTIONS]
uv run phenocai train fine-tune MODEL_DIR DATASET_PATH [OPTIONS]
uv run phenocai train list-models
uv run phenocai train list-presets
uv run phenocai train analyze-dataset DATASET_PATH [OPTIONS]
```

### Evaluation Commands
```bash
# Model evaluation
uv run phenocai evaluate model MODEL_PATH DATASET_PATH [OPTIONS]
uv run phenocai evaluate compare DATASET_PATH PREDICTIONS_PATH [OPTIONS]
uv run phenocai evaluate benchmark [OPTIONS]
```

### Analysis Commands
```bash
# Heuristic analysis
uv run phenocai analyze detect-snow IMAGE_PATH [OPTIONS]
uv run phenocai analyze assess-quality IMAGE_PATH [OPTIONS]
uv run phenocai analyze analyze-dataset DATASET_PATH [OPTIONS]
```

### Conversion Commands
```bash
# Annotation format conversion
uv run phenocai convert daily-to-individual DAILY_FILE OUTPUT_DIR [OPTIONS]
uv run phenocai convert station-daily-to-individual DAILY_DIR OUTPUT_DIR [OPTIONS]
uv run phenocai convert all-stations [OPTIONS]
```

### Station & Config Commands
```bash
# Station management
uv run phenocai station list
uv run phenocai station info STATION_NAME
uv run phenocai station switch STATION_NAME

# Configuration
uv run phenocai config show
uv run phenocai config validate
uv run phenocai config init
uv run phenocai config add-roi-00 [OPTIONS]
```

## Model Architectures

### Transfer Learning (MobileNetV2)
- Pre-trained on ImageNet with 1000+ classes
- Fine-tuned top layers for snow detection
- Data augmentation and regularization
- Support for progressive unfreezing

### Custom CNN
- Optimized architecture for phenocam images
- Multiple filter configurations available
- Batch normalization and dropout layers
- Global average pooling

### Ensemble Models
- Combine multiple base models
- Averaging, weighted, or stacking methods
- Improved robustness and accuracy

## Training Presets

| Preset | Model | Epochs | Features |
|--------|-------|--------|----------|
| `mobilenet_quick` | MobileNetV2 | 10 | Fast development testing |
| `mobilenet_full` | MobileNetV2 | 50 | Full training with fine-tuning |
| `custom_cnn_small` | Custom CNN | 30 | Lightweight architecture |
| `custom_cnn_large` | Custom CNN | 50 | High-capacity model |
| `ensemble_simple` | Ensemble | 10 | Model averaging |
| `ensemble_stacking` | Ensemble | 20 | Meta-learner stacking |

## Quality Flags

PhenoCAI recognizes 23+ quality flags including:

### Weather Conditions
- `fog`, `clouds`, `rain`, `snow`

### Illumination Issues  
- `high_brightness`, `low_brightness`, `shadows`

### Lens Problems
- `lens_dirt`, `lens_water_drops`, `lens_ice`

### Image Quality
- `blur`, `unusable`, `contrast_low`

### Wildlife
- `birds`, `small_wildlife`, `large_wildlife`

Use the `has_flags` field to easily filter images with quality issues.

## Annotation Formats

### Daily Format (Legacy)
```yaml
created: '2025-05-15T22:19:03'
day_of_year: '001'
station: "Lönnstorp"
instrument: LON_AGR_PL01_PHE01
annotations:
  image1.jpg:
    - roi_name: ROI_00
      snow_presence: true
      flags: []
```

### Individual Format (Current)
```yaml
created: '2025-05-15T22:19:03'
last_modified: '2025-05-23T13:33:45'
filename: lonnstorp_LON_AGR_PL01_PHE01_2024_001_20240101_090002.jpg
station: lonnstorp
year: '2024'
day_of_year: '001'
annotations:
  - roi_name: ROI_00
    snow_presence: true
    flags: []
    not_needed: false
```

## Dataset Structure

The dataset CSV files include fields for:
- Image metadata (filename, station, instrument, timestamp)
- ROI information and quality flags
- Snow presence annotations
- Train/test/val split assignments

See the [Data Preparation Guide](docs/workflows/workflow_data_preparation.md) for details on dataset creation.

## Cross-Station Evaluation

PhenoCAI supports training models at one station and evaluating at another using ROI_00 standardization:

### ROI_00 Universal Approach

ROI_00 represents the full image excluding the sky region, automatically calculated using advanced sky detection algorithms from phenotag/phenocams packages.

```bash
# Ensure ROI_00 is configured for all stations
uv run phenocai config add-roi-00

# Train on Lönnstorp (southern Sweden)
uv run phenocai dataset create --roi-filter ROI_00 --test-size 0.0
uv run phenocai train model lonnstorp_roi_00_dataset.csv

# Evaluate on Röbäcksdalen (northern Sweden)
uv run phenocai station switch robacksdalen
uv run phenocai dataset create --roi-filter ROI_00
uv run phenocai evaluate model /path/to/model.h5 robacksdalen_dataset.csv
```

### Multi-Station Training
```bash
# Create combined dataset
uv run phenocai dataset create-multi \
    --stations lonnstorp robacksdalen abisko \
    --roi-filter ROI_00
```

See [Cross-Station Evaluation Guide](docs/cross_station_evaluation.md) for details.

### Automated Cross-Station Pipeline
```bash
# Complete pipeline with annotation generation
uv run phenocai cross-station pipeline \
    --train-stations lonnstorp \
    --eval-stations robacksdalen abisko \
    --years 2023 2024 \
    --annotation-years 2022 2025 \
    --use-heuristics
```

See [Annotation Generation Workflow](docs/workflow_annotation_generation.md) for expanding datasets.

## Project Structure

```
phenocai/
├── src/phenocai/
│   ├── cli/              # Command-line interface
│   │   ├── main.py       # Main CLI entry point
│   │   ├── analyze.py    # Heuristic analysis commands
│   │   └── commands/     # Command modules
│   ├── config/           # Configuration and settings
│   ├── data_management/  # Dataset creation and handling
│   │   ├── dataset_builder.py    # Main dataset creation
│   │   └── annotation_converter.py # Format conversion
│   ├── analysis/         # Heuristic methods
│   │   └── heuristics.py # Snow detection & quality assessment
│   ├── models/           # ML model architectures
│   │   ├── architectures.py # Model definitions
│   │   └── config.py     # Model configurations
│   ├── training/         # Training pipeline
│   │   ├── trainer.py    # Main training logic
│   │   └── callbacks.py  # Training callbacks
│   ├── data/             # Data loading utilities
│   │   └── dataloader.py # TensorFlow data pipeline
│   ├── evaluation/       # Model evaluation
│   │   └── metrics.py    # Metrics and visualizations
│   └── utils.py          # Utility functions
├── scripts/              # Analysis scripts
├── docs/                 # Documentation
│   ├── workflow_*.md     # Educational workflow guides
│   ├── training_guide.md # Training instructions
│   └── implementation_plan.md # Development roadmap
├── CHANGELOG.md          # Version history
├── CLAUDE.md            # AI assistant guidance
└── pyproject.toml       # Package configuration
```

## Documentation

### User Guides
- [Training Guide](docs/training_guide.md) - Complete training instructions
- [Workflow Overview](docs/workflow_overview.md) - High-level process explanation
- [Data Preparation](docs/workflow_data_preparation.md) - Dataset creation guide
- [Training Process](docs/workflow_training.md) - Neural network training
- [Model Evaluation](docs/workflow_evaluation.md) - Performance assessment
- [Prediction Pipeline](docs/workflow_prediction.md) - Using trained models
- [Complete Guide](docs/workflow_complete_guide.md) - End-to-end workflow

### Technical Documentation
- [Implementation Plan](docs/implementation_plan.md) - Development roadmap
- [CLAUDE.md](CLAUDE.md) - AI assistant guidance
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Example Workflows

### Research Workflow
```bash
# 1. Convert annotations and create dataset
uv run phenocai convert all-stations
uv run phenocai dataset multi-station --stations lonnstorp robacksdalen

# 2. Analyze and clean data
uv run phenocai train analyze-dataset multi_station_dataset.csv
uv run phenocai dataset filter multi_station_dataset.csv clean_dataset.csv

# 3. Train and compare models
uv run phenocai train model clean_dataset.csv --preset mobilenet_full
uv run phenocai train model clean_dataset.csv --preset custom_cnn_large

# 4. Evaluate and benchmark
uv run phenocai evaluate benchmark --dataset-path test_dataset.csv
```

### Development Workflow
```bash
# Quick iteration cycle
uv run phenocai train model small_dataset.csv --preset mobilenet_quick
uv run phenocai evaluate model models/mobilenet_quick/final_model.h5 test.csv
uv run phenocai analyze detect-snow sample_image.jpg --visualize
```

### Production Workflow
```bash
# Full training pipeline
uv run phenocai train model production_dataset.csv --preset mobilenet_full
uv run phenocai train fine-tune models/mobilenet_full new_data.csv
uv run phenocai evaluate model models/final_model.h5 holdout_test.csv --analyze-errors
```

## Performance

### Heuristic Methods
- **Snow Detection**: Fast rule-based detection for clear images
- **Quality Assessment**: Identify common image quality issues
- **Speed**: Suitable for real-time processing

### Deep Learning Models
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet
- **Custom CNN**: Lightweight architectures for phenocam images
- **Ensemble**: Combine multiple models for improved robustness

### Processing Capabilities
- **GPU Support**: Accelerated training and inference
- **Batch Processing**: Efficient handling of large image datasets
- **Memory Management**: Optimized for limited resources

## Current Status

**Completed:**
- Complete training pipeline with TensorFlow
- Transfer learning with MobileNetV2
- Custom CNN architectures and ensemble methods
- Comprehensive CLI interface
- Heuristic analysis tools (snow detection, quality assessment)
- Annotation format conversion utilities
- Educational documentation with Mermaid diagrams
- Memory-efficient data loading pipeline
- Advanced evaluation metrics and visualizations
- ROI_00 automatic calculation and sky detection
- Cross-station training and evaluation support
- Complete pipeline automation
- Interactive Marimo notebook interface
- Annotation generation with ML and heuristics

**In Progress:**
- Model optimization and hyperparameter tuning
- Production deployment pipeline
- Real-time inference system

**Planned:**
- Web API for model serving
- Automated retraining pipeline
- Integration with SITES data infrastructure
- Multi-temporal analysis features
- Real-time monitoring dashboard
- Automated quality reports

## Contributing

This project is part of the SITES Spectral research infrastructure. For questions or contributions, please contact the development team.

### Development Setup
```bash
# Clone and setup development environment
git clone <repository-url>
cd phenocai
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/
```

## Citation

If you use PhenoCAI in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]

## Acknowledgments

This project is supported by the Swedish Infrastructure for Ecosystem Science (SITES) and uses data from the SITES Spectral thematic program. Special thanks to the researchers and technicians maintaining the phenocam networks at Lönnstorp and Röbäcksdalen stations.