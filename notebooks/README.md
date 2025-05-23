# PhenoCAI Notebooks

This directory contains interactive notebooks for PhenoCAI workflows.

## Available Notebooks

### `phenocai_pipeline.py` - Complete Pipeline Notebook
Interactive marimo notebook for running the complete PhenoCAI pipeline from dataset creation through prediction generation.

**Features:**
- Interactive station and instrument selection
- Configurable training parameters
- Real-time pipeline execution monitoring
- Results browsing and analysis
- Dry-run capability

**Usage:**
```bash
# Start the marimo server
marimo edit notebooks/phenocai_pipeline.py

# Or run in development mode
marimo run notebooks/phenocai_pipeline.py
```

## Getting Started

1. Ensure marimo is installed:
   ```bash
   uv add marimo
   ```

2. Start the notebook:
   ```bash
   cd /path/to/phenocai
   marimo edit notebooks/phenocai_pipeline.py
   ```

3. Open your browser to the provided URL (usually http://localhost:2718)

4. Follow the interactive interface to:
   - Select station and instrument
   - Configure training parameters
   - Execute the pipeline
   - Monitor progress and results

## Benefits of Interactive Notebooks

- **Visual Interface**: Easy-to-use dropdowns and sliders
- **Real-time Feedback**: See results as they're generated
- **Reproducible**: Save notebook state for later reference
- **Educational**: Perfect for students and new users
- **Flexible**: Modify parameters and re-run easily

## Alternative: CLI Pipeline

For scripted execution, use the CLI pipeline command:

```bash
# Complete pipeline with defaults
uv run phenocai pipeline full

# Customized pipeline
uv run phenocai pipeline full \
    --station lonnstorp \
    --instrument LON_AGR_PL01_PHE01 \
    --year 2024 \
    --prediction-years 2023 2024 \
    --model-type mobilenet \
    --preset mobilenet_full \
    --clean-only

# Check pipeline status
uv run phenocai pipeline status --station lonnstorp
```

## Troubleshooting

### Marimo Not Found
```bash
# Install marimo
uv add marimo

# Or with pip
pip install marimo
```

### Import Errors
Ensure you're running from the PhenoCAI root directory and the environment is activated:
```bash
source src/phenocai/config/env.sh
uv sync
```

### Port Already in Use
```bash
# Specify a different port
marimo edit notebooks/phenocai_pipeline.py --port 8080
```