trained_models/experimental_models_final_df_split/mobilenet_lonnstorp_PHE01_dataset_2024_from_day32_splits_20_10
View training progress with: tensorboard --logdir /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/trained_models/experimental_models_final_df_split/mobilenet_lonnstorp_PHE01_dataset_2024_from_day32_splits_20_10/logs

  -----
  Looking at those results, I can see several issues that explain why cross-station evaluation is so
  important:

  Analysis of Your Previous Training Results

  1. Severe Overfitting

  - Training accuracy: 69.4%
  - Validation accuracy: 94.6%
  - Test accuracy: 94.6%

  But notice:
  - Validation AUC: 0.5 (random guessing!)
  - Validation precision/recall: 0.0 (not detecting any snow!)

  2. What Happened?

  The model learned to predict "no snow" for everything:
  - Since only 13.6% of data has snow, predicting "no snow" always gives ~86% accuracy
  - The 94.6% test accuracy is misleading - it's just predicting the majority class
  - AUC of 0.5 confirms the model has no discriminative ability

  3. Why Cross-Station + ROI_00 Will Help

  Your previous issues likely came from:

  4. ROI Imbalance: Different ROIs had different snow patterns
  5. Class Imbalance: Only 13.6% snow samples
  6. Overfitting to Station-Specific Patterns: Model memorized Lönnstorp-specific features

  Recommended Solutions

  7. Use Balanced Metrics

  # Focus on these metrics instead of accuracy:
  - F1 Score (balances precision and recall)
  - AUC (area under ROC curve)
  - Balanced Accuracy
  - Precision and Recall for snow class

 2. Apply Class Weights

  # The training already calculates class weights:
  # Class weights: {0: 0.578, 1: 3.678}
  # This gives 3.7x more importance to snow samples

  3. Use ROI_00 Dataset with Better Balance

  # Create balanced dataset
  uv run phenocai dataset create --roi-filter ROI_00

  # Consider filtering for better balance
  uv run phenocai dataset filter dataset.csv --exclude-flags fog high_brightness
  # This can increase snow percentage from 13.6% to ~30%

  4. Monitor the Right Metrics

  # Train with focus on AUC instead of accuracy
  uv run phenocai train model dataset.csv \
      --monitor-metric val_auc \
      --monitor-mode max

  5. Use Data Augmentation

  The MobileNet presets already include augmentation, which helps with limited snow samples.

  6. Expected Better Results

  With ROI_00 and cross-station evaluation, you should see:
  - More balanced predictions (not just "no snow" for everything)
  - Better AUC (>0.7 for same station, >0.6 for cross-station)
  - Non-zero precision/recall for snow detection
  - Lower but more honest accuracy (~75-85% instead of misleading 94%)

  Quick Test

  Try this to verify your model's behavior:
  # Check if model predicts any snow at all
  uv run phenocai evaluate model your_model.h5 test_dataset.csv --save-predictions

  # Look at predictions - if all are 0 (no snow), the model failed to learn

  The cross-station evaluation with ROI_00 will give you a much more realistic assessment of model
  performance!
---
● Summary

  I've created a comprehensive cross-station prediction pipeline with annotation generation
  capabilities:

  1. New CLI Command: phenocai cross-station

  - pipeline: Complete automated workflow for cross-station training, evaluation, and annotation
  - merge-predictions: Combine model predictions with heuristics
  - create-station-models: Fine-tune models for specific stations

  2. Enhanced Dataset Commands

  - Added --roi-filter option to all dataset commands
  - Added --years option for multi-year datasets
  - Multi-station dataset now supports ROI and year filtering

  3. New Documentation

  - cross_station_evaluation.md: Complete guide for cross-station work
  - workflow_annotation_generation.md: How to generate annotations for new years

  4. Key Features

  Automated Pipeline

  # Train on Lönnstorp, evaluate on others, generate new annotations
  uv run phenocai cross-station pipeline \
      --train-stations lonnstorp \
      --eval-stations robacksdalen abisko \
      --years 2023 2024 \
      --annotation-years 2022 2025 \
      --use-heuristics

  ROI_00 Universal Approach

  # Create cross-station compatible datasets
  uv run phenocai dataset create --roi-filter ROI_00
  uv run phenocai dataset multi-station --stations lonnstorp robacksdalen --roi-filter ROI_00

  Annotation Generation with Heuristics

  - Combines ML predictions with rule-based quality flags
  - Confidence-based filtering for quality control
  - Supports iterative improvement through retraining

  5. Benefits

  - Expands datasets across years and stations automatically
  - Combines ML + heuristics for complete annotations
  - Confidence thresholds ensure quality
  - Iterative improvement through expanded datasets
  - Cross-station generalization testing

  The pipeline addresses your previous overfitting issues by:
  - Using ROI_00 for consistency across stations
  - Testing on completely different locations
  - Generating more balanced datasets through annotation
  - Providing realistic performance metrics


  ---
  
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