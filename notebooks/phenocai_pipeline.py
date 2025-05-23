"""
PhenoCAI Complete Pipeline - Interactive Marimo Notebook

This notebook provides an interactive interface for running the complete
PhenoCAI pipeline from dataset creation through model training, evaluation,
and prediction generation.
"""

import marimo as mo
import sys
from pathlib import Path
import os
import subprocess
import time
from datetime import datetime
import pandas as pd
import json

# Add the PhenoCAI source to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phenocai.config.setup import config
from phenocai.config.station_registry import get_registry


def __():
    # Page title and introduction
    mo.md("""
    # 🌲 PhenoCAI Complete Pipeline
    
    This interactive notebook guides you through the complete PhenoCAI workflow:
    **Dataset Creation** → **Model Training** → **Evaluation** → **Prediction**
    
    Use this notebook to process phenocam data from start to finish with just a few clicks!
    """)


def __():
    # Configuration section
    mo.md("## ⚙️ Configuration")


def __():
    # Station and instrument selection
    registry = get_registry()
    available_stations = registry.list_stations()
    
    # Station selector
    station_selector = mo.ui.dropdown(
        options=available_stations,
        value=config.current_station,
        label="Select Station:"
    )
    
    # Get instruments for selected station
    if station_selector.value:
        instruments = registry.list_instruments(station_selector.value)
        instrument_selector = mo.ui.dropdown(
            options=instruments,
            value=instruments[0] if instruments else None,
            label="Select Instrument:"
        )
    else:
        instrument_selector = mo.ui.dropdown(options=[], label="Select Instrument:")
    
    mo.md(f"""
    **Station Configuration:**
    """), station_selector, instrument_selector


def __():
    # Year and prediction configuration
    current_year = int(config.current_year)
    
    training_year = mo.ui.number(
        start=2020, stop=2030, step=1, value=current_year,
        label="Training Data Year:"
    )
    
    prediction_years = mo.ui.multiselect(
        options=[str(y) for y in range(2020, 2031)],
        value=["2023", "2024"],
        label="Generate Predictions For Years:"
    )
    
    mo.md("**Temporal Configuration:**"), training_year, prediction_years


def __():
    # Model and training configuration
    model_type = mo.ui.dropdown(
        options=["mobilenet", "custom_cnn", "ensemble"],
        value="mobilenet",
        label="Model Architecture:"
    )
    
    preset = mo.ui.dropdown(
        options=["mobilenet_quick", "mobilenet_full", "custom_cnn_small", "custom_cnn_large"],
        value="mobilenet_full",
        label="Training Preset:"
    )
    
    clean_only = mo.ui.checkbox(
        value=True,
        label="Use only clean data (no quality flags)"
    )
    
    mo.md("**Model Configuration:**"), model_type, preset, clean_only


def __():
    # Advanced options
    test_size = mo.ui.slider(
        start=0.1, stop=0.3, step=0.05, value=0.2,
        label="Test Set Fraction:"
    )
    
    val_size = mo.ui.slider(
        start=0.05, stop=0.2, step=0.05, value=0.1,
        label="Validation Set Fraction:"
    )
    
    output_dir = mo.ui.text(
        placeholder="Leave empty for auto-generated name",
        label="Output Directory (optional):"
    )
    
    mo.md("**Advanced Options:**"), test_size, val_size, output_dir


def __():
    # Pipeline execution section
    mo.md("## 🚀 Pipeline Execution")


def __():
    # Dry run option and execute button
    dry_run = mo.ui.checkbox(
        value=True,
        label="Dry run (show what would be executed without running)"
    )
    
    execute_button = mo.ui.button(
        label="🚀 Execute Pipeline",
        disabled=False,
        kind="success"
    )
    
    mo.hstack([dry_run, execute_button])


def __():
    # Pipeline execution logic
    if execute_button.value:
        mo.md("### 📋 Pipeline Execution")
        
        # Build command
        cmd = ["uv", "run", "phenocai", "pipeline", "full"]
        
        # Add configuration options
        if station_selector.value:
            cmd.extend(["--station", station_selector.value])
        
        if instrument_selector.value:
            cmd.extend(["--instrument", instrument_selector.value])
        
        if training_year.value:
            cmd.extend(["--year", str(training_year.value)])
        
        for year in prediction_years.value:
            cmd.extend(["--prediction-years", year])
        
        cmd.extend(["--model-type", model_type.value])
        cmd.extend(["--preset", preset.value])
        cmd.extend(["--test-size", str(test_size.value)])
        cmd.extend(["--val-size", str(val_size.value)])
        
        if clean_only.value:
            cmd.append("--clean-only")
        
        if dry_run.value:
            cmd.append("--dry-run")
        
        if output_dir.value:
            cmd.extend(["--output-dir", output_dir.value])
        
        # Display command
        cmd_str = " ".join(cmd)
        mo.md(f"**Executing command:**\n```bash\n{cmd_str}\n```")
        
        # Execute command
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                mo.md(f"✅ **Pipeline completed successfully!** ({duration:.1f}s)")
                mo.md(f"**Output:**\n```\n{result.stdout}\n```")
            else:
                mo.md(f"❌ **Pipeline failed!** (exit code: {result.returncode})")
                mo.md(f"**Error:**\n```\n{result.stderr}\n```")
                mo.md(f"**Output:**\n```\n{result.stdout}\n```")
        
        except Exception as e:
            mo.md(f"❌ **Error executing pipeline:** {e}")
    else:
        mo.md("Click the **Execute Pipeline** button above to start processing.")


def __():
    # Status monitoring section
    mo.md("## 📊 Pipeline Status")


def __():
    # Status check button
    status_button = mo.ui.button(
        label="📋 Check Status",
        kind="neutral"
    )
    
    refresh_button = mo.ui.button(
        label="🔄 Refresh",
        kind="neutral"
    )
    
    mo.hstack([status_button, refresh_button])


def __():
    # Status display
    if status_button.value or refresh_button.value:
        try:
            # Check datasets
            dataset_pattern = f"{station_selector.value or config.current_station}_*_dataset_*.csv"
            datasets = list(config.experimental_data_dir.glob(dataset_pattern))
            
            # Check models
            model_dirs = list(config.experimental_data_dir.glob("models_*"))
            
            # Check predictions
            prediction_dirs = list(config.experimental_data_dir.glob("*predictions_*"))
            
            # Create status table
            status_data = []
            
            # Dataset status
            if datasets:
                for dataset in datasets:
                    status_data.append({
                        "Type": "Dataset",
                        "Name": dataset.name,
                        "Status": "✅ Available",
                        "Size": f"{dataset.stat().st_size / 1024 / 1024:.1f} MB"
                    })
            else:
                status_data.append({
                    "Type": "Dataset",
                    "Name": "No datasets found",
                    "Status": "❌ Missing",
                    "Size": "-"
                })
            
            # Model status
            if model_dirs:
                for model_dir in model_dirs:
                    model_file = model_dir / "final_model.h5"
                    if model_file.exists():
                        status_data.append({
                            "Type": "Model",
                            "Name": model_dir.name,
                            "Status": "✅ Trained",
                            "Size": f"{model_file.stat().st_size / 1024 / 1024:.1f} MB"
                        })
                    else:
                        status_data.append({
                            "Type": "Model",
                            "Name": model_dir.name,
                            "Status": "⚠️ In Progress",
                            "Size": "-"
                        })
            else:
                status_data.append({
                    "Type": "Model",
                    "Name": "No models found",
                    "Status": "❌ Missing",
                    "Size": "-"
                })
            
            # Prediction status
            if prediction_dirs:
                for pred_dir in prediction_dirs:
                    pred_files = list(pred_dir.glob("*.yaml")) + list(pred_dir.glob("*.csv"))
                    status_data.append({
                        "Type": "Predictions",
                        "Name": pred_dir.name,
                        "Status": f"✅ {len(pred_files)} files",
                        "Size": "-"
                    })
            else:
                status_data.append({
                    "Type": "Predictions",
                    "Name": "No predictions found",
                    "Status": "❌ Missing",
                    "Size": "-"
                })
            
            status_df = pd.DataFrame(status_data)
            mo.ui.table(status_df)
            
        except Exception as e:
            mo.md(f"❌ Error checking status: {e}")
    else:
        mo.md("Click **Check Status** to see current pipeline state.")


def __():
    # Results section
    mo.md("## 📈 Results & Analysis")


def __():
    # Results browser
    results_dir = mo.ui.text(
        placeholder="Enter path to pipeline results directory",
        label="Results Directory:"
    )
    
    browse_button = mo.ui.button(
        label="📁 Browse Results",
        kind="neutral"
    )
    
    mo.hstack([results_dir, browse_button])


def __():
    # Results display
    if browse_button.value and results_dir.value:
        try:
            results_path = Path(results_dir.value)
            
            if results_path.exists():
                # List available result files
                result_files = []
                
                # Look for summary
                summary_file = results_path / "pipeline_summary.txt"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary_content = f.read()
                    mo.md(f"### 📋 Pipeline Summary\n```\n{summary_content}\n```")
                
                # Look for evaluation results
                eval_files = list(results_path.glob("**/evaluation_*.txt"))
                if eval_files:
                    mo.md("### 📊 Evaluation Results")
                    for eval_file in eval_files:
                        with open(eval_file, 'r') as f:
                            eval_content = f.read()
                        mo.md(f"**{eval_file.name}:**\n```\n{eval_content[:1000]}...\n```")
                
                # Look for prediction summaries
                pred_dirs = list(results_path.glob("predictions_*"))
                if pred_dirs:
                    mo.md("### 🔮 Prediction Summaries")
                    for pred_dir in pred_dirs:
                        year = pred_dir.name.split('_')[-1]
                        pred_files = list(pred_dir.glob("*.yaml")) + list(pred_dir.glob("*.csv"))
                        mo.md(f"**{year}:** {len(pred_files)} prediction files")
                
            else:
                mo.md(f"❌ Directory not found: {results_dir.value}")
        
        except Exception as e:
            mo.md(f"❌ Error browsing results: {e}")


def __():
    # Footer
    mo.md("""
    ---
    
    ## 💡 Tips & Best Practices
    
    - **Start with dry run** to preview what will be executed
    - **Use clean data** for initial model training (better accuracy)
    - **MobileNet Full preset** provides the best balance of speed and accuracy
    - **Check status regularly** to monitor progress
    - **Save results** to organized directories for easy analysis
    
    ## 🔗 Related Documentation
    
    - [Training Guide](../docs/training_guide.md)
    - [API Reference](../docs/api_reference.md)
    - [Quick Reference](../docs/quick_reference.md)
    
    **PhenoCAI v0.2.0** - Automated Phenocam Image Analysis
    """)