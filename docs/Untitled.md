  Manual Step-by-Step Commands

  Step 1: Setup and Switch Station/Instrument

  # Check current configuration
  uv run phenocai info

  # Switch to specific station and instrument
  uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE01 --year 2024

  # Verify the switch
  uv run phenocai station instruments

  Step 2: Create Dataset

  # Create dataset with train/test/val splits
  uv run phenocai dataset create \
      --test-size 0.2 \
      --val-size 0.1
  # This creates: lonnstorp_PHE01_dataset_2024_splits_20_10.csv

  Step 3: Filter Dataset (Optional - for clean data only)

  # Filter to remove all quality flags
  uv run phenocai dataset filter \
      lonnstorp_PHE01_dataset_2024_splits_20_10.csv \
      --no-flags
  # This creates: lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv

  # Or filter specific flags
  uv run phenocai dataset filter \
      lonnstorp_PHE01_dataset_2024_splits_20_10.csv \
      --exclude-flags fog high_brightness lens_water_drops

  Step 4: Analyze Dataset Quality

  # Check dataset statistics
  uv run phenocai dataset info lonnstorp_PHE01_dataset_2024_splits_20_10.csv

  # Detailed quality analysis
  python scripts/analyze_quality_issues.py lonnstorp_PHE01_dataset_2024_splits_20_10.csv

  Step 5: Train Model

  # Train with MobileNetV2 full preset (recommended)
  uv run phenocai train model \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_full_clean

  # Or with custom parameters
  uv run phenocai train model \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --model-type mobilenet \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001 \
      --output-dir trained_models/custom_mobilenet

  Step 6: Evaluate Model

  # Evaluate on test set
  uv run phenocai evaluate model \
      trained_models/mobilenet_full_clean/final_model.h5 \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --save-predictions \
      --generate-plots

  # Get detailed metrics
  uv run phenocai evaluate model \
      trained_models/mobilenet_full_clean/final_model.h5 \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --split test

  Step 7: Generate Predictions for New Years

  # Predict for 2023
  uv run phenocai predict batch \
      trained_models/mobilenet_full_clean/final_model.h5 \
      --year 2023 \
      --output-dir predictions/2023 \
      --format yaml \
      --use-heuristics

  # Predict for 2024
  uv run phenocai predict batch \
      trained_models/mobilenet_full_clean/final_model.h5 \
      --year 2024 \
      --output-dir predictions/2024 \
      --format yaml \
      --use-heuristics

  # Predict for specific date range
  uv run phenocai predict batch \
      trained_models/mobilenet_full_clean/final_model.h5 \
      --start-day 100 \
      --end-day 200 \
      --year 2024 \
      --output-dir predictions/2024_spring \
      --format yaml

  Step 8: Export Predictions

  # Export predictions to CSV
  uv run phenocai predict export \
      predictions/2023 \
      --format csv \
      --output lonnstorp_PHE01_predictions_2023.csv

  uv run phenocai predict export \
      predictions/2024 \
      --format csv \
      --output lonnstorp_PHE01_predictions_2024.csv

  # Export to JSON for analysis
  uv run phenocai predict export \
      predictions/2024 \
      --format json \
      --output lonnstorp_PHE01_predictions_2024.json

  Complete Example Workflow

  Here's a complete example with all commands in sequence:

  # 1. Setup
  source src/phenocai/config/env.sh
  uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE01 --year 2024

  # 2. Create dataset
  uv run phenocai dataset create --test-size 0.2 --val-size 0.1

  # 3. Filter to clean data
  uv run phenocai dataset filter \
      lonnstorp_PHE01_dataset_2024_splits_20_10.csv \
      --no-flags

  # 4. Analyze
  python scripts/analyze_quality_issues.py \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv

  # 5. Train
  uv run phenocai train model \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_clean_2024

  # 6. Evaluate
  uv run phenocai evaluate model \
      trained_models/mobilenet_clean_2024/final_model.h5 \
      lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
      --save-predictions --generate-plots

  # 7. Predict for multiple years
  for year in 2023 2024 2025; do
      uv run phenocai predict batch \
          trained_models/mobilenet_clean_2024/final_model.h5 \
          --year $year \
          --output-dir predictions/${year} \
          --format yaml \
          --use-heuristics
  done

  # 8. Export all predictions
  for year in 2023 2024 2025; do
      uv run phenocai predict export \
          predictions/${year} \
          --format csv \
          --output lonnstorp_PHE01_predictions_${year}.csv
  done

  Key Differences from Pipeline Command

  The manual approach gives you:
  - More control over each step
  - Ability to inspect intermediate results
  - Custom file naming for outputs
  - Selective re-running of specific steps
  - Different parameters for each step

  The pipeline command (uv run phenocai pipeline full) automates all these steps with sensible
  defaults, but manual execution gives you complete flexibility!
  ----
  phenocai train …
      model "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstor
      p/experimental_data/lonnstorp_PHE01_dataset_2024_splits_20_10.csv" --preset mobilenet_quick
      --epochs 2