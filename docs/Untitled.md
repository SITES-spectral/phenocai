---
"-":
---

export  DATASET_PATH="/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed_balanced.csv"

phenocai train model "$DATASET_PATH"  --preset mobilenet_full --output-dir /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset --epochs 50 --batch-size 32
----
see below for the updated banaced fix call

● I've created a comprehensive document with the findings and recommendations.  Cost: $0.2213 (10.4s)
   The document includes:

  1. Executive Summary - Quick overview of the main issue and findings
  2. Key Findings - Detailed metrics and analysis of the prediction bias
  3. Root Cause Analysis - Explanation of why validation metrics showed 0
  4. Recommendations covering:
    - Immediate fixes for the current model
    - Training improvements for future models
    - Data preprocessing suggestions
    - Model architecture adjustments
    - Evaluation best practices
  5. Deployment Guidelines - How to use the model in production
  6. Code examples throughout for practical implementation

  The document is saved at: /lunarc/nobackup/projects/sitesspec/SITES/Spectral
  /apps/phenocai/docs/model_evaluation_findings.md

  You can view it with:
  less /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/docs/m
  odel_evaluation_findings.md

  Or convert to PDF if needed:
  pandoc docs/model_evaluation_findings.md -o model_evaluation_findings.pdf

> as it is actually true that are more days with no snow than with snow, we could as another balance 
  strategy first get all the files with annotations with snow, then randomly select the same amount 
  of files with no snow, then use that as a dataset to select from for training. What do you think? 
  could we implement a command for that?

✻ Thinking…

  The user is suggesting a great strategy for handling class imbalance - instead of using class 
  weights or trying to fix it during training, they want to create a balanced dataset by:
  1. Taking all snow samples
  2. Randomly sampling an equal number of no-snow samples
  3. Creating a balanced dataset for training

  This is a common and effective approach called undersampling the majority class. It's particularly 
  good when you have plenty of data for the majority class.

  I should implement this as a command that:
  4. Reads the dataset
  5. Separates snow/no-snow samples
  6. Takes all snow samples
  7. Randomly samples equal number of no-snow samples
  8. Combines them into a balanced dataset
  9. Maintains the train/val/test splits properly

● Excellent idea! That's a very effective strategy called "undersampling the    Cost: $0.6847 (21.5s)
  majority class" - it often works better than class weights because the model
   sees a truly balanced distribution during training. Let me implement this
  as a dataset preparation command:

● Update Todos                                                                  Cost: $0.6847 (21.5s)
  ⎿  ☐ Create a balanced dataset command that undersamples no-snow images (P0)
     ☐ Add the command to the CLI (P0)
     ☐ Test the balanced dataset creation (P0)

✻ Thinking…

  I'll create a script that implements balanced dataset creation through undersampling. This will:
  1. Load the dataset CSV
  2. Separate snow and no-snow samples
  3. For each split (train/val/test), balance the classes
  4. Save the balanced dataset

  I should make sure to:
  - Preserve the original train/val/test splits
  - Use random sampling with a seed for reproducibility
  - Report statistics about the balancing
  - Allow for different balancing strategies (1:1, 1:2, etc.)

● Write(file_path: /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/ph…  Cost: $0.1847 (52.0s)
       nocai/src/phenocai/data_management/dataset_balancer.py)
  ⎿ Wrote 256 lines to /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src/phenocai/
data_management/dataset_balancer.py
     """Dataset balancing utilities for handling class imbalance."""

     import pandas as pd
     import numpy as np
     from pathlib import Path
     from typing import Optional, Tuple, Dict
     import logging
     from collections import Counter

     logger = logging.getLogger(__name__)


     class DatasetBalancer:
         """Balance dataset by undersampling majority class."""

         def __init__(self, random_seed: int = 42):
             """Initialize balancer.
             
             Args:
                 random_seed: Random seed for reproducibility
             """
             self.random_seed = random_seed
             np.random.seed(random_seed)

         def balance_dataset(
             self,
             df: pd.DataFrame,
             target_column: str = 'snow_presence',
             ratio: float = 1.0,
             strategy: str = 'undersample',
             preserve_splits: bool = True
         ) -> pd.DataFrame:
             """Balance dataset by adjusting class distribution.
             
             Args:
                 df: Input dataframe
                 target_column: Column containing class labels
                 ratio: Ratio of majority to minority class (1.0 = equal)
                 strategy: 'undersample' or 'oversample'
                 preserve_splits: Whether to balance within existing splits
                 
             Returns:
                 Balanced dataframe
             """
             if strategy != 'undersample':
                 raise NotImplementedError("Only undersample strategy is currently supported")

             if preserve_splits and 'split' in df.columns:
                 # Balance within each split
                 balanced_dfs = []

                 for split in df['split'].unique():
                     split_df = df[df['split'] == split]
                     balanced_split = self._balance_binary_split(
                         split_df, target_column, ratio
                     )
                     balanced_dfs.append(balanced_split)

                 balanced_df = pd.concat(balanced_dfs, ignore_index=True)
             else:
                 # Balance entire dataset
                 balanced_df = self._balance_binary_split(df, target_column, ratio)

             return balanced_df

         def _balance_binary_split(
             self,
             df: pd.DataFrame,
             target_column: str,
             ratio: float
         ) -> pd.DataFrame:
             """Balance a binary classification split.
             
             Args:
                 df: Dataframe to balance
                 target_column: Target column name
                 ratio: Ratio of majority to minority samples
                 
             Returns:
                 Balanced dataframe
             """
             # Separate classes
             positive_df = df[df[target_column] == True]
             negative_df = df[df[target_column] == False]

             n_positive = len(positive_df)
             n_negative = len(negative_df)

             # Determine minority and majority
             if n_positive < n_negative:
                 minority_df = positive_df
                 majority_df = negative_df
                 minority_label = "positive (snow)"
                 majority_label = "negative (no snow)"
             else:
                 minority_df = negative_df
                 majority_df = positive_df
                 minority_label = "negative (no snow)"
                 majority_label = "positive (snow)"

             # Calculate target number of majority samples
             n_minority = len(minority_df)
             n_target_majority = int(n_minority * ratio)

             # Sample from majority class
             if n_target_majority < len(majority_df):
                 # Undersample majority
                 majority_sampled = majority_df.sample(
                     n=n_target_majority,
                     random_state=self.random_seed
                 )
                 logger.info(
                     f"Undersampled {majority_label} from {len(majority_df)} to {n_target_majority}"
                 )
             else:
                 # Use all majority samples
                 majority_sampled = majority_df
                 logger.info(
                     f"Using all {len(majority_df)} {majority_label} samples"
                 )

             # Combine and shuffle
             balanced_df = pd.concat([minority_df, majority_sampled], ignore_index=True)
             balanced_df = balanced_df.sample(frac=1,
     random_state=self.random_seed).reset_index(drop=True)

             return balanced_df

         def get_balance_report(
             self,
             original_df: pd.DataFrame,
             balanced_df: pd.DataFrame,
             target_column: str = 'snow_presence'
         ) -> Dict:
             """Generate report on balancing results.
             
             Args:
                 original_df: Original dataframe
                 balanced_df: Balanced dataframe
                 target_column: Target column name
                 
             Returns:
                 Dictionary with balance statistics
             """
             report = {
                 'original': self._get_class_stats(original_df, target_column),
                 'balanced': self._get_class_stats(balanced_df, target_column),
             }

             # Add split information if available
             if 'split' in original_df.columns:
                 report['original_by_split'] = {}
                 report['balanced_by_split'] = {}

                 for split in original_df['split'].unique():
                     orig_split = original_df[original_df['split'] == split]
                     report['original_by_split'][split] = self._get_class_stats(
                         orig_split, target_column
                     )

                     if split in balanced_df['split'].unique():
                         bal_split = balanced_df[balanced_df['split'] == split]
                         report['balanced_by_split'][split] = self._get_class_stats(
                             bal_split, target_column
                         )

             return report

         def _get_class_stats(self, df: pd.DataFrame, target_column: str) -> Dict:
             """Get class distribution statistics."""
             total = len(df)
             positive = df[target_column].sum()
             negative = total - positive

             return {
                 'total': total,
                 'positive': int(positive),
                 'negative': int(negative),
                 'positive_pct': positive / total * 100 if total > 0 else 0,
                 'negative_pct': negative / total * 100 if total > 0 else 0,
                 'ratio': negative / positive if positive > 0 else float('inf')
             }


     def balance_dataset_from_csv(
         input_csv: str,
         output_csv: str,
         target_column: str = 'snow_presence',
         ratio: float = 1.0,
         random_seed: int = 42
     ) -> str:
         """Balance dataset from CSV file.
         
         Args:
             input_csv: Path to input CSV
             output_csv: Path to output CSV
             target_column: Column containing labels
             ratio: Ratio of majority to minority class
             random_seed: Random seed
             
         Returns:
             Path to output CSV
         """
         # Load dataset
         df = pd.read_csv(input_csv)
         logger.info(f"Loaded dataset with {len(df)} samples")

         # Create balancer
         balancer = DatasetBalancer(random_seed=random_seed)

         # Balance dataset
         balanced_df = balancer.balance_dataset(
             df,
             target_column=target_column,
             ratio=ratio,
             preserve_splits=True
         )

         # Get report
         report = balancer.get_balance_report(df, balanced_df, target_column)

         # Print report
         print("\n" + "="*60)
         print("Dataset Balancing Report")
         print("="*60)

         print(f"\nOriginal Dataset:")
         print(f"  Total samples: {report['original']['total']}")
         print(f"  Positive (snow): {report['original']['positive']} 
     ({report['original']['positive_pct']:.1f}%)")
         print(f"  Negative (no snow): {report['original']['negative']} 
     ({report['original']['negative_pct']:.1f}%)")

         print(f"\nBalanced Dataset:")
         print(f"  Total samples: {report['balanced']['total']}")
         print(f"  Positive (snow): {report['balanced']['positive']} 
     ({report['balanced']['positive_pct']:.1f}%)")
         print(f"  Negative (no snow): {report['balanced']['negative']} 
     ({report['balanced']['negative_pct']:.1f}%)")

         if 'original_by_split' in report:
             print(f"\nBy Split:")
             for split in report['original_by_split']:
                 orig = report['original_by_split'][split]
                 if split in report['balanced_by_split']:
                     bal = report['balanced_by_split'][split]
                     print(f"\n  {split}:")
                     print(f"    Original: {orig['total']} samples ({orig['positive_pct']:.1f}% 
     positive)")
                     print(f"    Balanced: {bal['total']} samples ({bal['positive_pct']:.1f}% 
     positive)")

         # Save balanced dataset
         balanced_df.to_csv(output_csv, index=False)
         logger.info(f"Saved balanced dataset to {output_csv}")

         # Save report
         report_path = Path(output_csv).with_suffix('.balance_report.json')
         import json
         with open(report_path, 'w') as f:
             json.dump(report, f, indent=2)

         return output_csv

✻ Thinking…

  Now I need to add this to the CLI. I'll create a new command under the dataset subcommand group.

● Update Todos                                                                   Cost: $0.1136 (9.5s)
  ⎿  ☒ Create a balanced dataset command that undersamples no-snow images (P0)
     ☐ Add the command to the CLI (P0)
     ☐ Test the balanced dataset creation (P0)

✻ Thinking…

  I need to add the balance command to the dataset CLI commands. Let me check the current dataset 
  commands structure.

● Read(file_path: "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/ph…   Cost: $0.0918 (7.0s)
      nocai/src/phenocai/cli/commands/dataset.py")
  ⎿  """
     Dataset creation and management commands
     """
     import click
     from pathlib import Path
     import pandas as pd

     from ...config.setup import config
     from ...data_management import (
         create_master_annotation_dataframe,
         create_multi_station_dataset,
         load_multi_station_dataset,
         filter_dataset_by_criteria
     )


     @click.group()
     def dataset():
         """Create and manage datasets."""
         pass


     @dataset.command()
     @click.option('--output', '-o', type=click.Path(), help='Output path for dataset')
     @click.option('--include-unannotated', is_flag=True, help='Include unannotated ROIs')
     @click.option('--test-size', default=0.2, help='Test set fraction (default: 0.2)')
     @click.option('--val-size', default=0.1, help='Validation set fraction (default: 0.1)')
     @click.option('--format', type=click.Choice(['csv', 'parquet']), default='csv', help='Output 
     format')
     @click.option('--instrument', '-i', help='Instrument ID (overrides current instrument)')
     @click.option('--complete-rois-only/--no-complete-rois-only', default=True, help='Only include 
     images with all ROIs annotated')
     @click.option('--min-day', type=int, help='Minimum day of year to include (e.g., 32)')
     @click.option('--roi-filter', multiple=True, help='Only include specific ROIs (e.g., --roi-filte
      ROI_00)')
     def create(output, include_unannotated, test_size, val_size, format, instrument, 
     complete_rois_only, min_day, roi_filter):
         """Create dataset from current station's annotations."""
         # Handle instrument switching if provided
         if instrument:
             try:
                 config.switch_instrument(instrument)
                 click.echo(f"Using instrument: {instrument}")
             except ValueError as e:
                 click.echo(f"Error: {e}", err=True)
                 return 1

         click.echo(f"Creating dataset for {config.current_station} - {config.current_instrument}..."

         # Set output path with intelligent naming
         if output is None:
             # Generate filename based on station, instrument, year, and options
             suffix = '.parquet' if format == 'parquet' else '.csv'
             filename_parts = [
                 config.current_station
             ]

             # Add instrument if there are multiple for this station
             available_instruments = config.list_available_instruments()
             if len(available_instruments) > 1:
                 # Extract just the instrument number part (e.g., PHE01 from LON_AGR_PL01_PHE01)
                 inst_parts = config.current_instrument.split('_')
                 if len(inst_parts) >= 4:
                     inst_suffix = inst_parts[-1]  # Get PHE01
                     filename_parts.append(inst_suffix)

             filename_parts.append(f'dataset_{config.current_year}')

             if include_unannotated:
                 filename_parts.append('with_unannotated')

             # Add filtering info to filename
             if roi_filter:
                 roi_str = '_'.join(roi_filter).lower()
                 filename_parts.append(roi_str)
             elif complete_rois_only or min_day is not None:
                 if min_day or (config.current_station == 'lonnstorp' and complete_rois_only):
                     day_filter = min_day or 32
                     filename_parts.append(f'from_day{day_filter}')
                 else:
                     filename_parts.append('complete_rois')

             # Add split info to filename
             filename_parts.append(f'splits_{int(test_size*100)}_{int(val_size*100)}')

             filename = '_'.join(filename_parts) + suffix
             output = config.experimental_data_dir / filename

             # Ensure experimental data directory exists
             config.experimental_data_dir.mkdir(parents=True, exist_ok=True)
         else:
             output = Path(output)

         # Ensure correct extension
         if format == 'parquet' and not output.suffix == '.parquet':
             output = output.with_suffix('.parquet')
         elif format == 'csv' and not output.suffix == '.csv':
             output = output.with_suffix('.csv')

         try:
             # Create dataset
             df, stats = create_master_annotation_dataframe(
                 config.master_annotation_pool_dir,
                 output_path=None,  # Don't save yet - we'll add splits first
                 include_unannotated=include_unannotated
             )

             # Filter to complete ROI sets if requested
             if complete_rois_only or min_day is not None:
                 from ...data_management.dataset_builder import filter_complete_roi_sets

                 # For Lönnstorp, we know that all ROIs started being annotated from day 32
                 if min_day is None and config.current_station == 'lonnstorp' and not roi_filter:
                     min_day = 32
                     click.echo(f"Note: For Lönnstorp, filtering to images from day {min_day} onwards
     (when all ROIs were annotated)")

                 # Skip complete ROI filtering if specific ROIs are requested
                 if not roi_filter:
                     df = filter_complete_roi_sets(df, min_day_of_year=min_day)

             # Filter to specific ROIs if requested
             if roi_filter:
                 roi_list = list(roi_filter)
                 initial_count = len(df)
                 df = df[df['roi_name'].isin(roi_list)]
                 filtered_count = len(df)
                 click.echo(f"Filtered to ROIs {roi_list}: {initial_count} → {filtered_count} 
     records")

             # Add train/test/val splits
             from ...data_management.dataset_builder import add_train_test_split
             df = add_train_test_split(
                 df,
                 test_size=test_size,
                 val_size=val_size,
                 stratify_by='snow_presence',
                 random_state=42,
                 group_by_day=True
             )

             # Update stats with split information
             stats.train_size = len(df[df['split'] == 'train'])
             stats.test_size = len(df[df['split'] == 'test'])
             stats.val_size = len(df[df['split'] == 'val'])

             # Save dataset with splits
             if format == 'parquet':
                 df.to_parquet(output, index=False)
             else:
                 df.to_csv(output, index=False)

             # Print statistics
             click.echo(f"\n✓ Created dataset with {len(df)} records")
             stats.print_summary()

             click.echo(f"\nDataset saved to: {output}")

         except Exception as e:
             click.echo(f"Error creating dataset: {e}", err=True)
             return 1


     @dataset.command('multi-station')
     @click.option('--stations', '-s', multiple=True, help='Stations to include (defaults to primary 
     stations)')
     @click.option('--output', '-o', type=click.Path(), help='Output path for dataset')
     @click.option('--include-unannotated', is_flag=True, help='Include unannotated ROIs')
     @click.option('--balance', is_flag=True, help='Balance samples across stations')
     @click.option('--test-size', default=0.2, help='Test set fraction')
     @click.option('--val-size', default=0.1, help='Validation set fraction')
     @click.option('--format', type=click.Choice(['csv', 'parquet']), default='csv', help='Output 
     format')
     @click.option('--roi-filter', multiple=True, help='Only include specific ROIs (e.g., --roi-filte
      ROI_00)')
     @click.option('--years', multiple=True, help='Years to include (default: current year)')
     def multi_station(stations, output, include_unannotated, balance, test_size, val_size, format, 
     roi_filter, years):
         """Create multi-station dataset."""
         # Use provided stations or defaults
         if not stations:
             from ...config.station_configs import get_primary_stations
             stations = get_primary_stations()
             click.echo(f"Using primary stations: {', '.join(stations)}")
         else:
             stations = list(stations)

         # Set output path with intelligent naming
         if output is None:
             # Generate filename based on stations and options
             suffix = '.parquet' if format == 'parquet' else '.csv'
             station_str = '_'.join(sorted(stations))
             filename_parts = [
                 'multi_station',
                 station_str,
                 f'dataset_{config.current_year}'
             ]

             if include_unannotated:
                 filename_parts.append('with_unannotated')

             if balance:
                 filename_parts.append('balanced')

             # Add ROI filter to filename
             if roi_filter:
                 roi_str = '_'.join(roi_filter).lower()
                 filename_parts.append(roi_str)

             # Add years if specified
             if years:
                 years_str = '_'.join(sorted(years))
                 filename_parts.append(years_str)

             # Add split info to filename
             filename_parts.append(f'splits_{int(test_size*100)}_{int(val_size*100)}')

             filename = '_'.join(filename_parts) + suffix
             output = config.experimental_data_dir / filename

             # Ensure experimental data directory exists
             config.experimental_data_dir.mkdir(parents=True, exist_ok=True)
         else:
             output = Path(output)

         # Ensure correct extension
         if format == 'parquet':
             output = output.with_suffix('.parquet')
         else:
             output = output.with_suffix('.csv')

         click.echo(f"\nCreating multi-station dataset...")

         try:
             df, station_stats = create_multi_station_dataset(
                 stations=stations,
                 output_path=output,
                 include_unannotated=include_unannotated,
                 test_size=test_size,
                 val_size=val_size,
                 balance_stations=balance,
                 roi_filter=list(roi_filter) if roi_filter else None,
                 years=list(years) if years else None
             )

             click.echo(f"\n✓ Created multi-station dataset")
             click.echo(f"Dataset saved to: {output}")

         except Exception as e:
             click.echo(f"Error creating multi-station dataset: {e}", err=True)
             return 1


     @dataset.command()
     @click.argument('dataset_path', type=click.Path(exists=True))
     def info(dataset_path):
         """Show information about a dataset."""
         dataset_path = Path(dataset_path)

         click.echo(f"Loading dataset from {dataset_path}...")

         try:
             if dataset_path.suffix == '.parquet':
                 df = pd.read_parquet(dataset_path)
             else:
                 df = pd.read_csv(dataset_path)

             click.echo(f"\n=== Dataset Information ===")
             click.echo(f"Total records: {len(df)}")
             click.echo(f"Columns: {', '.join(df.columns)}")

             if 'station' in df.columns:
                 click.echo(f"\nStations:")
                 for station, count in df['station'].value_counts().items():
                     click.echo(f"  • {station}: {count} records")

             if 'split' in df.columns:
                 click.echo(f"\nData splits:")
                 for split, count in df['split'].value_counts().items():
                     click.echo(f"  • {split}: {count} records")

             if 'roi_name' in df.columns:
                 click.echo(f"\nROI distribution:")
                 for roi, count in df['roi_name'].value_counts().head(10).items():
                     click.echo(f"  • {roi}: {count} records")

             if 'snow_presence' in df.columns:
                 snow_count = df['snow_presence'].sum()
                 click.echo(f"\nLabels:")
                 click.echo(f"  • Snow present: {snow_count} ({snow_count/len(df)*100:.1f}%)")
                 click.echo(f"  • No snow: {len(df) - snow_count} 
     ({(len(df)-snow_count)/len(df)*100:.1f}%)")

             # Add quality flag statistics
             if 'has_flags' in df.columns or 'flag_count' in df.columns:
                 if 'has_flags' not in df.columns and 'flag_count' in df.columns:
                     df['has_flags'] = df['flag_count'] > 0

                 if 'has_flags' in df.columns:
                     flagged_count = df['has_flags'].sum()
                     click.echo(f"\nQuality issues:")
                     click.echo(f"  • With flags: {flagged_count} ({flagged_count/len(df)*100:.1f}%)"
                     click.echo(f"  • Clean: {len(df) - flagged_count} 
     ({(len(df)-flagged_count)/len(df)*100:.1f}%)")

             if 'flags' in df.columns:
                 # Parse and count individual flags
                 all_flags = []
                 for flags_str in df[df['flags'].notna()]['flags']:
                     if flags_str:
                         all_flags.extend(flags_str.split(','))

                 if all_flags:
                     from collections import Counter
                     flag_counts = Counter(all_flags)
                     click.echo(f"\nTop quality flags:")
                     for flag, count in flag_counts.most_common(5):
                         click.echo(f"  • {flag}: {count}")

         except Exception as e:
             click.echo(f"Error loading dataset: {e}", err=True)
             return 1


     @dataset.command()
     @click.argument('input_path', type=click.Path(exists=True))
     @click.argument('output_path', type=click.Path(), required=False)
     @click.option('--stations', '-s', multiple=True, help='Filter by stations')
     @click.option('--rois', '-r', multiple=True, help='Filter by ROI names')
     @click.option('--min-year', type=int, help='Minimum year')
     @click.option('--max-year', type=int, help='Maximum year')
     @click.option('--exclude-flags', '-x', multiple=True, help='Exclude specific flags')
     @click.option('--no-flags', is_flag=True, help='Include only samples with no quality flags')
     @click.option('--annotated-only', is_flag=True, default=True, help='Include only annotated 
     samples')
     def filter(input_path, output_path, stations, rois, min_year, max_year, exclude_flags, no_flags,
     annotated_only):
         """Filter an existing dataset."""
         input_path = Path(input_path)

         # Generate intelligent output name if not provided
         if output_path is None:
             # Parse input filename to create filtered version
             stem = input_path.stem
             suffix = input_path.suffix

             # Add filter descriptors
             filter_parts = [stem]

             if no_flags:
                 filter_parts.append('clean')
             elif exclude_flags:
                 flags_str = '_'.join(sorted(exclude_flags))
                 filter_parts.append(f'no_{flags_str}')

             if stations:
                 stations_str = '_'.join(sorted(stations))
                 filter_parts.append(f'stations_{stations_str}')

             if rois:
                 rois_str = '_'.join(sorted(rois))
                 filter_parts.append(f'rois_{rois_str}')

             if min_year or max_year:
                 year_parts = []
                 if min_year:
                     year_parts.append(f'from{min_year}')
                 if max_year:
                     year_parts.append(f'to{max_year}')
                 filter_parts.append('_'.join(year_parts))

             filter_parts.append('filtered')

             output_filename = '_'.join(filter_parts) + suffix
             output_path = input_path.parent / output_filename
         else:
             output_path = Path(output_path)

         click.echo(f"Loading dataset from {input_path}...")

         try:
             # Load dataset
             if input_path.suffix == '.parquet':
                 df = pd.read_parquet(input_path)
             else:
                 df = pd.read_csv(input_path)

             original_size = len(df)

             # Apply filters
             # Handle no_flags option
             if no_flags:
                 # Override exclude_flags to exclude all records with any flags
                 df = df[~df['has_flags']] if 'has_flags' in df.columns else df[df['flag_count'] == 0
             else:
                 df = filter_dataset_by_criteria(
                     df,
                     stations=list(stations) if stations else None,
                     rois=list(rois) if rois else None,
                     min_year=min_year,
                     max_year=max_year,
                     exclude_flags=list(exclude_flags) if exclude_flags else None,
                     annotated_only=annotated_only
                 )

             # Save filtered dataset
             if output_path.suffix == '.parquet':
                 df.to_parquet(output_path, index=False)
             else:
                 df.to_csv(output_path, index=False)

             click.echo(f"\n✓ Filtered dataset: {original_size} → {len(df)} records")
             click.echo(f"Saved to: {output_path}")

         except Exception as e:
             click.echo(f"Error filtering dataset: {e}", err=True)
             return 1




✻ Thinking…

  Now I'll add the balance command to this dataset command group.

● Update(/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src…  Cost: $0.1713 (23.5s)
        phenocai/cli/commands/dataset.py)
  ⎿  Updated/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src/phenocai/cli/command
            s/dataset.py with 60 additions
     400 
     401      except Exception as e:
     402          click.echo(f"Error filtering dataset: {e}", err=True)
     403          return 1
     404  
     405  
     406  @dataset.command()
     407  @click.argument('input_path', type=click.Path(exists=True))
     408  @click.argument('output_path', type=click.Path(), required=False)
     409  @click.option('--ratio', '-r', default=1.0, help='Ratio of majority to minority class (def
         ault: 1.0 for balanced)')
     410  @click.option('--target-column', default='snow_presence', help='Column to balance on')
     411  @click.option('--random-seed', default=42, type=int, help='Random seed for reproducibility
         ')
     412  def balance(input_path, output_path, ratio, target_column, random_seed):
     413      """Balance dataset by undersampling majority class.
     414      
     415      This command creates a balanced dataset by taking all samples from the 
     416      minority class (usually snow) and randomly sampling from the majority 
     417      class (usually no-snow) to achieve the desired ratio.
     418      
     419      Examples:
     420          # Create perfectly balanced dataset (1:1 ratio)
     421          phenocai dataset balance dataset.csv
     422          
     423          # Create dataset with 2 no-snow samples for each snow sample
     424          phenocai dataset balance dataset.csv --ratio 2.0
     425      """
     426      from ...data_management.dataset_balancer import balance_dataset_from_csv
     427      
     428      input_path = Path(input_path)
     429      
     430      # Generate output path if not provided
     431      if output_path is None:
     432          stem = input_path.stem
     433          suffix = input_path.suffix
     434          
     435          # Add balance info to filename
     436          if ratio == 1.0:
     437              balance_str = 'balanced'
     438          else:
     439              balance_str = f'ratio_{ratio:.1f}'.replace('.', '_')
     440          
     441          output_filename = f"{stem}_{balance_str}{suffix}"
     442          output_path = input_path.parent / output_filename
     443      else:
     444          output_path = Path(output_path)
     445      
     446      click.echo(f"Balancing dataset with ratio {ratio}:1 (majority:minority)...")
     447      
     448      try:
     449          # Balance the dataset
     450          result_path = balance_dataset_from_csv(
     451              str(input_path),
     452              str(output_path),
     453              target_column=target_column,
     454              ratio=ratio,
     455              random_seed=random_seed
     456          )
     457          
     458          click.echo(f"\n✓ Successfully created balanced dataset")
     459          click.echo(f"Output saved to: {result_path}")
     460          
     461      except Exception as e:
     462          click.echo(f"Error balancing dataset: {e}", err=True)
     463          return 1
     464 \ No newline at end of file

✻ Thinking…

  Now I need to make sure the dataset_balancer module is properly imported in the data_management 
  init.py file.

● Update(/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src…  Cost: $0.1206 (11.3s)
        phenocai/data_management/__init__.py)
  ⎿  Error: File has not been read yet. Read it first before writing to it.

✻ Thinking…

  Let me first read the init.py file.

● Read(file_path: "/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/ph…   Cost: $0.0981 (5.8s)
      nocai/src/phenocai/data_management/__init__.py")
  ⎿  """
     PhenoCAI Data Management Module

     This module handles annotation loading, dataset creation, and data management
     for the PhenoCAI system.
     """

     from .annotation_loader import (
         load_daily_annotations,
         load_individual_annotation,
         AnnotationData,
         ROIAnnotation
     )

     from .dataset_builder import (
         create_master_annotation_dataframe,
         add_train_test_split,
         DatasetStats
     )

     from .multi_station_builder import (
         create_multi_station_dataset,
         load_multi_station_dataset,
         filter_dataset_by_criteria
     )

     __all__ = [
         'load_daily_annotations',
         'load_individual_annotation',
         'AnnotationData',
         'ROIAnnotation',
         'create_master_annotation_dataframe',
         'add_train_test_split',
         'DatasetStats',
         'create_multi_station_dataset',
         'load_multi_station_dataset',
         'filter_dataset_by_criteria'
     ]

✻ Thinking…

  Now I can update it to include the dataset_balancer imports.

● Update(/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src…  Cost: $0.1132 (11.2s)
        phenocai/data_management/__init__.py)
  ⎿  Updated/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/src/phenocai/data_manage
            ment/__init__.py with 8 additions and 1 removal
     24      filter_dataset_by_criteria
     25  )
     26 
     27  from .dataset_balancer import (
     28      DatasetBalancer,
     29      balance_dataset_from_csv
     30  )
     31  
     32  __all__ = [
     33      'load_daily_annotations',
     34      'load_individual_annotation',
     ...
     34      'DatasetStats',
     35      'create_multi_station_dataset',
     36      'load_multi_station_dataset',
     37      'filter_dataset_by_criteria'
     37      'filter_dataset_by_criteria',
     38      'DatasetBalancer',
     39      'balance_dataset_from_csv'
     40  ]
     41 \ No newline at end of file

✻ Thinking…

  Now I need to add the missing import for logging in the dataset_balancer.py file.

● Update Todos                                                                   Cost: $0.1053 (8.1s)
  ⎿  ☒ Create a balanced dataset command that undersamples no-snow images (P0)
     ☒ Add the command to the CLI (P0)
     ☐ Test the balanced dataset creation (P0)

✻ Thinking…

  Now let me test the balance command with the existing dataset.

● Bash(cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai &&…  Cost: $0.1019 (10.7s)
      phenocai dataset balance /home/jobelund/lu2024-12-46/SITES/Spectral/anal
      ysis/phenocams/transfer_learning/lonnstorp/experimental_data/multi_stati
      on_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed.csv)
  ⎿ Balancing dataset with ratio 1.0:1 (majority:minority)...

    ============================================================
    Dataset Balancing Report
    ============================================================

    Original Dataset:
      Total samples: 8963
      Positive (snow): 3449 (38.5%)
      Negative (no snow): 5514 (61.5%)

    Balanced Dataset:
      Total samples: 6898
      Positive (snow): 3449 (50.0%)
      Negative (no snow): 3449 (50.0%)

    By Split:

      train:
        Original: 6106 samples (39.7% positive)
        Balanced: 4846 samples (50.0% positive)

      val:
        Original: 851 samples (37.1% positive)
        Balanced: 632 samples (50.0% positive)

      test:
        Original: 2006 samples (35.4% positive)
        Balanced: 1420 samples (50.0% positive)

    ✓ Successfully created balanced dataset
    Output saved to:
    /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experim
    ental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed_balanced.csv
  ⎿ 2025-05-25 08:35:51.118516: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find 
    cuda drivers on your machine, GPU will not be used.
    2025-05-25 08:35:52.343720: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find 
    cuda drivers on your machine, GPU will not be used.
    2025-05-25 08:35:52.668865: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable
     to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already 
    been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1748154952.964437  258416 cuda_dnn.cc:8579] Unable to register cuDNN factory: 
    Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1748154953.006043  258416 cuda_blas.cc:1407] Unable to register cuBLAS factory: 
    Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1748154953.631166  258416 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748154953.631210  258416 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748154953.631214  258416 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748154953.631216  258416 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    2025-05-25 08:35:53.638364: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow 
    binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the 
    appropriate compiler flags.
    2025-05-25 08:36:13,360 - phenocai.data_management.dataset_balancer - INFO - Loaded dataset with 
    8963 samples
    2025-05-25 08:36:13,365 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 3683 to 2423
    2025-05-25 08:36:13,370 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 535 to 316
    2025-05-25 08:36:13,373 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 1296 to 710
    2025-05-25 08:36:13,452 - phenocai.data_management.dataset_balancer - INFO - Saved balanced 
    dataset to 
    /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experim
    ental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed_balanced.csv
    Shell cwd was reset to /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai

✻ Thinking…

  Excellent! The balance command worked perfectly. It created a balanced dataset where:
  - Each split (train/val/test) now has exactly 50% snow and 50% no-snow samples
  - The total dataset was reduced from 8,963 to 6,898 samples
  - All snow samples were kept, and no-snow samples were randomly undersampled

  Let me create one more example with a different ratio to show how it works.

● Bash(cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai &&…  Cost: $0.1159 (11.4s)
      phenocai dataset balance /home/jobelund/lu2024-12-46/SITES/Spectral/anal
      ysis/phenocams/transfer_learning/lonnstorp/experimental_data/multi_stati
      on_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed.csv --ratio
      1.5)
  ⎿ Balancing dataset with ratio 1.5:1 (majority:minority)...

    ============================================================
    Dataset Balancing Report
    ============================================================

    Original Dataset:
      Total samples: 8963
      Positive (snow): 3449 (38.5%)
      Negative (no snow): 5514 (61.5%)

    Balanced Dataset:
      Total samples: 8622
      Positive (snow): 3449 (40.0%)
      Negative (no snow): 5173 (60.0%)

    By Split:

      train:
        Original: 6106 samples (39.7% positive)
        Balanced: 6057 samples (40.0% positive)

      val:
        Original: 851 samples (37.1% positive)
        Balanced: 790 samples (40.0% positive)

      test:
        Original: 2006 samples (35.4% positive)
        Balanced: 1775 samples (40.0% positive)

    ✓ Successfully created balanced dataset
    Output saved to:
    /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experim
    ental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed_ratio_1_5.csv
  ⎿ 2025-05-25 08:37:08.654406: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find 
    cuda drivers on your machine, GPU will not be used.
    2025-05-25 08:37:08.659038: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find 
    cuda drivers on your machine, GPU will not be used.
    2025-05-25 08:37:08.667274: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable
     to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already 
    been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1748155028.680665  260589 cuda_dnn.cc:8579] Unable to register cuDNN factory: 
    Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1748155028.684879  260589 cuda_blas.cc:1407] Unable to register cuBLAS factory: 
    Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1748155028.698354  260589 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748155028.698372  260589 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748155028.698375  260589 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1748155028.698378  260589 computation_placer.cc:177] computation placer already 
    registered. Please check linkage and avoid linking the same target more than once.
    2025-05-25 08:37:08.702604: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow 
    binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the 
    appropriate compiler flags.
    2025-05-25 08:37:10,533 - phenocai.data_management.dataset_balancer - INFO - Loaded dataset with 
    8963 samples
    2025-05-25 08:37:10,539 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 3683 to 3634
    2025-05-25 08:37:10,544 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 535 to 474
    2025-05-25 08:37:10,547 - phenocai.data_management.dataset_balancer - INFO - Undersampled 
    negative (no snow) from 1296 to 1065
    2025-05-25 08:37:10,635 - phenocai.data_management.dataset_balancer - INFO - Saved balanced 
    dataset to 
    /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experim
    ental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10_fixed_ratio_1_5.csv
    Shell cwd was reset to /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai

✻ Thinking…

  Perfect! The command is working well. Let me complete the task by marking everything as done.

● Update Todos                                                                   Cost: $0.1136 (8.0s)
  ⎿  ☒ Create a balanced dataset command that undersamples no-snow images (P0)
     ☒ Add the command to the CLI (P0)
     ☒ Test the balanced dataset creation (P0)

✻ Thinking…

  Let me also show the user how to train with the balanced dataset.

● Implementation Complete!                                                      Cost: $0.1116 (15.2s)

  I've successfully implemented the dataset balancing strategy. The new
  command creates a balanced dataset by:

  1. Taking all snow samples (minority class)
  2. Randomly undersampling no-snow samples (majority class)
  3. Maintaining the original train/val/test splits

  Usage Examples:

  # Create perfectly balanced dataset (1:1 ratio)
  phenocai dataset balance dataset.csv

  # Create dataset with custom ratio (e.g., 1.5 no-snow for each snow)
  phenocai dataset balance dataset.csv --ratio 1.5

  Results:

  - Balanced dataset (1:1): 6,898 samples (50% snow, 50% no-snow)
  - Ratio 1.5 dataset: 8,622 samples (40% snow, 60% no-snow)

  Training with Balanced Dataset:

  phenocai train model \
      "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_
  learning/lonnstorp/experimental_data/multi_station_lonnstorp_robacksdalen_da
  taset_2024_splits_20_10_fixed_balanced.csv" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_balanced \
      --epochs 50 \
      --batch-size 32

  This balanced dataset approach should significantly improve model
  calibration and reduce the need for threshold adjustments!


----


● I've created a comprehensive document with the findings and recommendations. The document includes:

  1. Executive Summary - Quick overview of the main issue and findings
  2. Key Findings - Detailed metrics and analysis of the prediction bias
  3. Root Cause Analysis - Explanation of why validation metrics showed 0
  4. Recommendations covering:
    - Immediate fixes for the current model
    - Training improvements for future models
    - Data preprocessing suggestions
    - Model architecture adjustments
    - Evaluation best practices
  5. Deployment Guidelines - How to use the model in production
  6. Code examples throughout for practical implementation

  The document is saved at:
  /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/docs/model_evaluation_findings.md

  You can view it with:
  less
  /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/docs/model_evaluation_findings.md

  Or convert to PDF if needed:
  pandoc docs/model_evaluation_findings.md -o model_evaluation_findings.pdf

-----

phenocai train model "$DATASET_PATH"  --preset mobilenet_full --output-dir /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset --epochs 50 --batch-size 32
-----

   and prediction:

  Phase 1: Setup and Dataset Analysis

  # Set working directory and environment
  cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai
  source src/phenocai/config/env.sh

  # Check current configuration
  uv run phenocai info

  # Analyze the multi-station dataset first
  uv run phenocai train analyze-dataset
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  # Get detailed dataset information
  uv run phenocai dataset info
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  Phase 2: Model Training

  A. Train MobileNet Model (Full Training)

  # Train MobileNet with full preset (comprehensive training)
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_multi_station_full \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001

  # Alternative: Train with custom parameters for fine control
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --model-type mobilenet \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001 \
      --freeze-base \
      --output-dir trained_models/mobilenet_multi_station_custom

  B. Train Custom CNN Model (For Comparison)

  # Train custom CNN with full preset
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --preset custom_cnn_large \
      --output-dir trained_models/custom_cnn_multi_station \
      --epochs 40 \
      --batch-size 32

  C. Heuristics "Training" (Analysis and Baseline)

  # Analyze dataset with heuristics for baseline comparison
  uv run phenocai analyze analyze-dataset
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --sample-size 1000 \
      --save-results \
      --output-dir heuristics_analysis

  Phase 3: Model Evaluation

  A. Evaluate MobileNet Model

  # Comprehensive evaluation with visualizations
  uv run phenocai evaluate model trained_models/mobilenet_multi_station_full/final_model.h5 \
      "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/expe
  rimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --split test \
      --save-predictions \
      --generate-plots \
      --plot-samples 16 \
      --analyze-errors \
      --output-dir evaluations/mobilenet_multi_station

  B. Evaluate Custom CNN Model

  # Evaluate custom CNN
  uv run phenocai evaluate model trained_models/custom_cnn_multi_station/final_model.h5 \
      "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/expe
  rimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --split test \
      --save-predictions \
      --generate-plots \
      --output-dir evaluations/custom_cnn_multi_station

  C. Benchmark All Models

  # Compare all models on the same test dataset
  uv run phenocai evaluate benchmark \
      --dataset-path "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning
  /lonnstorp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --models-dir trained_models \
      --output-dir benchmark_results

  Phase 4: Predictions on Other Years

  A. Generate Predictions for Multiple Years

  # Set up for predictions (use the best model from evaluation)
  BEST_MODEL="trained_models/mobilenet_multi_station_full/final_model.h5"

  # Predict for 2022 data
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/2022 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Predict for 2023 data  
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2023 \
      --output-dir predictions/2023 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Predict for 2025 data (if available)
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2025 \
      --output-dir predictions/2025 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Export predictions to CSV for analysis
  uv run phenocai predict export predictions/2022 \
      --format csv \
      --output predictions_2022.csv

  uv run phenocai predict export predictions/2023 \
      --format csv \
      --output predictions_2023.csv

  B. Cross-Station Predictions (If you want to test generalization)

  # Switch to Lönnstorp and predict
  uv run phenocai station switch lonnstorp
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/lonnstorp_2022 \
      --format yaml \
      --use-heuristics

  # Switch to Röbäcksdalen and predict  
  uv run phenocai station switch robacksdalen
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/robacksdalen_2022 \
      --format yaml \
      --use-heuristics

  Phase 5: Evaluate Predictions

  A. Create Ground Truth Datasets for Other Years (if annotations exist)

  # If you have annotations for 2022, create a test dataset
  uv run phenocai station switch lonnstorp
  uv run phenocai dataset create \
      --roi-filter ROI_00 \
      --year 2022 \
      --output lonnstorp_2022_test.csv

  # Same for Röbäcksdalen 2022
  uv run phenocai station switch robacksdalen
  uv run phenocai dataset create \
      --roi-filter ROI_00 \
      --year 2022 \
      --output robacksdalen_2022_test.csv

  B. Compare Predictions with Ground Truth

  # Compare 2022 predictions with ground truth (if available)
  uv run phenocai evaluate compare \
      lonnstorp_2022_test.csv \
      predictions_2022.csv \
      --output-dir prediction_evaluation/2022 \
      --generate-plots

  # Evaluate prediction quality over time
  uv run phenocai evaluate compare \
      robacksdalen_2022_test.csv \
      predictions/robacksdalen_2022 \
      --output-dir prediction_evaluation/robacksdalen_2022

  Phase 6: Advanced Analysis and Fine-Tuning

  A. Fine-tune Model on New Years (if you get good predictions)

  # If predictions for 2022 look good, you can fine-tune the model
  uv run phenocai train fine-tune trained_models/mobilenet_multi_station_full \
      lonnstorp_2022_test.csv \
      --epochs 10 \
      --learning-rate 0.0001 \
      --output-dir trained_models/mobilenet_finetuned_2022

  B. Complete Cross-Station Pipeline (Alternative Automated Approach)

  # Run the complete automated pipeline for cross-station analysis
  uv run phenocai cross-station pipeline \
      --train-stations lonnstorp robacksdalen \
      --eval-stations abisko \
      --years 2024 \
      --annotation-years 2022 2023 2025 \
      --use-heuristics \
      --roi-filter ROI_00 \
      --output-dir cross_station_analysis

  Phase 7: Results Analysis and Monitoring

  # Check training progress with TensorBoard (if you have it installed)
  tensorboard --logdir trained_models/mobilenet_multi_station_full/logs

  # List all trained models
  uv run phenocai train list-models

  # Get training presets for reference
  uv run phenocai train list-presets

  # Check pipeline status
  uv run phenocai pipeline status

  Expected Workflow Execution Order:

  1. Start with Phase 1 (Dataset analysis)
  2. Run Phase 2A (MobileNet training) - This will take the longest
  3. While training runs, prepare Phase 2C (Heuristics analysis)
  4. Run Phase 3A (Evaluate MobileNet)
  5. Optional: Phase 2B (Custom CNN) if you want comparison
  6. Run Phase 4A (Generate predictions)
  7. Run Phase 5 (Evaluate predictions if ground truth available)

  Key Parameters Explained:

  - --preset mobilenet_full: Uses optimized hyperparameters for full training
  - --epochs 50: Full training cycles (will take several hours)
  - --batch-size 32: Good balance for memory and training speed
  - --use-heuristics: Combines ML with rule-based quality detection
  - --roi-filter ROI_00: Ensures cross-station compatibility
  - --save-predictions: Saves predictions for later analysis
  - --generate-plots: Creates evaluation visualizations

  This pipeline will give you comprehensive training, evaluation, and prediction results across
  multiple years and stations!

> phenocai train analyze-dataset
    
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
    ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"
  Usage: phenocai train analyze-dataset [OPTIONS] DATASET_PATH
  Try 'phenocai train analyze-dataset --help' for help.

  Error: Missing argument 'DATASET_PATH'.
  bash: 
  /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
    ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv: No such file or 
  directory

● I see the issue! The command line is breaking the path due to line wrapping. Let me provide you
  with the corrected commands using proper shell escaping and line continuation:

  Fixed Commands with Proper Path Handling

  # Set working directory and environment
  cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai
  source src/phenocai/config/env.sh

  # Define the dataset path as a variable to avoid line breaking issues
  DATASET_PATH="/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonns
  torp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  # Verify the file exists
  ls -la "$DATASET_PATH"

  Phase 1: Setup and Dataset Analysis

  # Check current configuration
  uv run phenocai info

  # Analyze the multi-station dataset first
  uv run phenocai train analyze-dataset "$DATASET_PATH"

  # Get detailed dataset information
  uv run phenocai dataset info "$DATASET_PATH"

  Phase 2: Model Training

  A. Train MobileNet Model (Full Training)

  # Train MobileNet with full preset (comprehensive training)
  uv run phenocai train model "$DATASET_PATH" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_multi_station_full \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001

  B. Train Custom CNN Model (For Comparison)

  # Train custom CNN with full preset
  uv run phenocai train model "$DATASET_PATH" \
      --preset custom_cnn_large \
      --output-dir trained_models/custom_cnn_multi_station \
      --epochs 40 \
      --batch-size 32

  C. Heuristics Analysis (Baseline)

  # Analyze dataset with heuristics for baseline comparison
  uv run phenocai analyze analyze-dataset "$DATASET_PATH" \
      --sample-size 1000 \
      --save-results \
      --output-dir heuristics_analysis

  Phase 3: Model Evaluation

  # Comprehensive evaluation with visualizations
  uv run phenocai evaluate model trained_models/mobilenet_multi_station_full/final_model.h5 \
      "$DATASET_PATH" \
      --split test \
      --save-predictions \
      --generate-plots \
      --plot-samples 16 \
      --analyze-errors \
      --output-dir evaluations/mobilenet_multi_station

  # Evaluate custom CNN
  uv run phenocai evaluate model trained_models/custom_cnn_multi_station/final_model.h5 \
      "$DATASET_PATH" \
      --split test \
      --save-predictions \
      --generate-plots \
      --output-dir evaluations/custom_cnn_multi_station

  # Compare all models on the same test dataset
  uv run phenocai evaluate benchmark \
      --dataset-path "$DATASET_PATH" \
      --models-dir trained_models \
      --output-dir benchmark_results

  Phase 4: Predictions on Other Years

  # Set up for predictions (use the best model from evaluation)
  BEST_MODEL="trained_models/mobilenet_multi_station_full/final_model.h5"

  # Predict for 2022 data
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/2022 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Predict for 2023 data  
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2023 \
      --output-dir predictions/2023 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Export predictions to CSV for analysis
  uv run phenocai predict export predictions/2022 \
      --format csv \
      --output predictions_2022.csv

  uv run phenocai predict export predictions/2023 \
      --format csv \
      --output predictions_2023.csv

  Start with These Commands First:

  # 1. Set up environment and variables
  cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai
  source src/phenocai/config/env.sh
  DATASET_PATH="/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonns
  torp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  # 2. Verify file exists
  ls -la "$DATASET_PATH"

  # 3. Check current configuration
  uv run phenocai info

  # 4. Analyze the dataset
  uv run phenocai train analyze-dataset "$DATASET_PATH"

  The key fixes:
  1. Used shell variable DATASET_PATH to avoid path breaking
  2. Proper quoting with "$DATASET_PATH"
  3. Verified file existence before running commands
  4. Removed line breaks within the path argument

  Try running these commands in order, and let me know the results from the dataset analysis!

> === Dataset Analysis: multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv ===
  Total samples: 8963

  Class distribution:
    Snow: 3449 (38.5%)
    No snow: 5514 (61.5%)

  ✓ Dataset is reasonably balanced

  Quality issues:
    Images with flags: 6117 (68.2%)

  ⚠️  WARNING: Many images have quality issues!
    Consider filtering dataset before training

  === Recommendations ===
  ✓ Dataset size (8963) is sufficient for training

● Update Todos
  ⎿  ☐ Train MobileNet model on multi-station dataset
     ☐ Train heuristics model
     ☐ Evaluate both models
     ☐ Generate predictions for other years
     ☐ Evaluate predictions on other years

● Great! The dataset analysis shows you have a good dataset for training:

  📊 Dataset Summary:
  - 8,963 samples - Excellent size for training
  - 38.5% snow, 61.5% no snow - Reasonably balanced (ideal for classification)
  - 68.2% have quality flags - This is expected and manageable

  Since you have quality issues in 68% of images, I recommend two training approaches:

  Option 1: Train on Full Dataset (Recommended for robustness)

  # Train MobileNet on the full dataset (includes all quality conditions)
  uv run phenocai train model "$DATASET_PATH"  --preset mobilenet_full --output-dir /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/trained_models/mobilenet_full_dataset --epochs 50 --batch-size 32

  Option 2: Train on Clean Dataset (Higher accuracy, less robustness)

  # First, create a filtered clean dataset
  uv run phenocai dataset filter "$DATASET_PATH" \
      --no-flags \
      --output clean_dataset.csv

  # Then train on clean data
  uv run phenocai train model clean_dataset.csv \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_clean_dataset \
      --epochs 50 \
      --batch-size 32

  Recommended Next Steps:

  Step 1: Start Training (Choose your approach)

  For maximum robustness (trains on all data including quality issues):
  uv run phenocai train model "$DATASET_PATH" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_multi_station_robust \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001

  Step 2: While training runs, start heuristics analysis:

  # Run heuristics analysis (this is fast and can run while training)
  uv run phenocai analyze analyze-dataset "$DATASET_PATH" \
      --sample-size 2000 \
      --save-results \
      --output-dir heuristics_baseline

  Step 3: Check training progress:

  # List available presets to see what mobilenet_full includes
  uv run phenocai train list-presets

  # Monitor training (if you want to see what's happening)
  # The training will show progress automatically

  Step 4: Once training completes (will take several hours), evaluate:

  # Evaluate the trained model
  uv run phenocai evaluate model trained_models/mobilenet_multi_station_robust/final_model.h5 \
      "$DATASET_PATH" \
      --split test \
      --save-predictions \
      --generate-plots \
      --plot-samples 16 \
      --analyze-errors \
      --output-dir evaluations/mobilenet_robust

  Which approach do you want to take?

  1. Train on full dataset (recommended) - More robust, handles quality issues better
  2. Train on clean dataset - Higher accuracy on perfect images, but less robust
  3. Train both for comparison - Takes longer but gives you complete analysis

  I recommend starting with Option 1 (full dataset) since your real-world predictions will include
  images with quality issues, and you want a robust model.

  Start with this command:
  uv run phenocai train model "$DATASET_PATH" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_multi_station_robust \
      --epochs 50 \
      --batch-size 32

  This will train for several hours. While it's training, you can run the heuristics analysis in
  parallel!

---

phenocai train analyze-dataset "$DATASET_PATH"

=== Dataset Analysis: multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv ===
Total samples: 8963

Class distribution:
  Snow: 3449 (38.5%)
  No snow: 5514 (61.5%)

✓ Dataset is reasonably balanced

Quality issues:
  Images with flags: 6117 (68.2%)

⚠️  WARNING: Many images have quality issues!
  Consider filtering dataset before training

=== Recommendations ===
✓ Dataset size (8963) is sufficient for training


---
  Commands That Actually Work:

  ✅ Verified Working Commands:

  Main Commands:
  - phenocai info ✅
  - phenocai version ✅

  Analyze Commands:
  - phenocai analyze detect-snow ✅
  - phenocai analyze assess-quality ✅
  - phenocai analyze analyze-dataset ✅

  Dataset Commands:
  - phenocai dataset create ✅
  - phenocai dataset multi-station ✅ (NOT create-multi)
  - phenocai dataset filter ✅
  - phenocai dataset info ✅

  Train Commands:
  - phenocai train model ✅
  - phenocai train fine-tune ✅
  - phenocai train list-models ✅ (just fixed)
  - phenocai train list-presets ✅
  - phenocai train analyze-dataset ✅

  Evaluate Commands:
  - phenocai evaluate model ✅
  - phenocai evaluate compare ✅
  - phenocai evaluate benchmark ✅

  Predict Commands:
  - phenocai predict apply ✅
  - phenocai predict batch ✅
  - phenocai predict export ✅

  Station Commands:
  - phenocai station list ✅
  - phenocai station switch ✅
  - phenocai station info ✅
  - phenocai station instruments ✅

  Config Commands:
  - phenocai config show ✅
  - phenocai config validate ✅
  - phenocai config init ✅
  - phenocai config add-roi-00 ✅

  Convert Commands:
  - phenocai convert daily-to-individual ✅
  - phenocai convert station-daily-to-individual ✅
  - phenocai convert all-stations ✅

  Cross-Station Commands:
  - phenocai cross-station pipeline ✅
  - phenocai cross-station merge-predictions ✅
  - phenocai cross-station create-station-models ✅

  Pipeline Commands:
  - phenocai pipeline full ✅
  - phenocai pipeline status ✅

  Files Updated:

  1. README.md - Fixed create-multi → multi-station, removed dataset validate
  2. docs/quick_reference.md - Fixed command references and cross-station evaluation
  3. docs/workflow_complete_guide.md - Updated cross-station evaluation examples
  4. docs/training_guide.md - Fixed create-multi → multi-station
  5. docs/implementation_plan.md - Fixed command references
  6. CHANGELOG.md - Updated command references

  The CLI documentation now accurately reflects only the commands that actually exist and work. Users
   won't encounter "No such command" errors when following the documentation!

> I have this multi-station dataset 
  `/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv` I want to train a 
  mobilnet model and an heuristics model too, then evaluate the models and make prediction on other 
  years and evaluate the predictions, please use give the phenocai cli commands with all the 
  parameters and full training

● I'll help you create a complete training and evaluation pipeline for your multi-station dataset.
  Let me provide you with the full sequence of commands with proper parameters.

● Update Todos
  ⎿  ☐ Train MobileNet model on multi-station dataset
     ☐ Train heuristics model
     ☐ Evaluate both models
     ☐ Generate predictions for other years
     ☐ Evaluate predictions on other years

● Here's the complete pipeline with all the necessary PhenoCAI CLI commands for training, evaluation,
   and prediction:

  Phase 1: Setup and Dataset Analysis

  # Set working directory and environment
  cd /lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai
  source src/phenocai/config/env.sh

  # Check current configuration
  uv run phenocai info

  # Analyze the multi-station dataset first
  uv run phenocai train analyze-dataset
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  # Get detailed dataset information
  uv run phenocai dataset info
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv"

  Phase 2: Model Training

  A. Train MobileNet Model (Full Training)

  # Train MobileNet with full preset (comprehensive training)
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --preset mobilenet_full \
      --output-dir trained_models/mobilenet_multi_station_full \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001

  # Alternative: Train with custom parameters for fine control
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --model-type mobilenet \
      --epochs 50 \
      --batch-size 32 \
      --learning-rate 0.001 \
      --freeze-base \
      --output-dir trained_models/mobilenet_multi_station_custom

  B. Train Custom CNN Model (For Comparison)

  # Train custom CNN with full preset
  uv run phenocai train model
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --preset custom_cnn_large \
      --output-dir trained_models/custom_cnn_multi_station \
      --epochs 40 \
      --batch-size 32

  C. Heuristics "Training" (Analysis and Baseline)

  # Analyze dataset with heuristics for baseline comparison
  uv run phenocai analyze analyze-dataset
  "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experime
  ntal_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --sample-size 1000 \
      --save-results \
      --output-dir heuristics_analysis

  Phase 3: Model Evaluation

  A. Evaluate MobileNet Model

  # Comprehensive evaluation with visualizations
  uv run phenocai evaluate model trained_models/mobilenet_multi_station_full/final_model.h5 \
      "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/expe
  rimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --split test \
      --save-predictions \
      --generate-plots \
      --plot-samples 16 \
      --analyze-errors \
      --output-dir evaluations/mobilenet_multi_station

  B. Evaluate Custom CNN Model

  # Evaluate custom CNN
  uv run phenocai evaluate model trained_models/custom_cnn_multi_station/final_model.h5 \
      "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/expe
  rimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --split test \
      --save-predictions \
      --generate-plots \
      --output-dir evaluations/custom_cnn_multi_station

  C. Benchmark All Models

  # Compare all models on the same test dataset
  uv run phenocai evaluate benchmark \
      --dataset-path "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning
  /lonnstorp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv" \
      --models-dir trained_models \
      --output-dir benchmark_results

  Phase 4: Predictions on Other Years

  A. Generate Predictions for Multiple Years

  # Set up for predictions (use the best model from evaluation)
  BEST_MODEL="trained_models/mobilenet_multi_station_full/final_model.h5"

  # Predict for 2022 data
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/2022 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Predict for 2023 data  
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2023 \
      --output-dir predictions/2023 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Predict for 2025 data (if available)
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2025 \
      --output-dir predictions/2025 \
      --format yaml \
      --use-heuristics \
      --roi-filter ROI_00

  # Export predictions to CSV for analysis
  uv run phenocai predict export predictions/2022 \
      --format csv \
      --output predictions_2022.csv

  uv run phenocai predict export predictions/2023 \
      --format csv \
      --output predictions_2023.csv

  B. Cross-Station Predictions (If you want to test generalization)

  # Switch to Lönnstorp and predict
  uv run phenocai station switch lonnstorp
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/lonnstorp_2022 \
      --format yaml \
      --use-heuristics

  # Switch to Röbäcksdalen and predict  
  uv run phenocai station switch robacksdalen
  uv run phenocai predict batch "$BEST_MODEL" \
      --year 2022 \
      --output-dir predictions/robacksdalen_2022 \
      --format yaml \
      --use-heuristics

  Phase 5: Evaluate Predictions

  A. Create Ground Truth Datasets for Other Years (if annotations exist)

  # If you have annotations for 2022, create a test dataset
  uv run phenocai station switch lonnstorp
  uv run phenocai dataset create \
      --roi-filter ROI_00 \
      --year 2022 \
      --output lonnstorp_2022_test.csv

  # Same for Röbäcksdalen 2022
  uv run phenocai station switch robacksdalen
  uv run phenocai dataset create \
      --roi-filter ROI_00 \
      --year 2022 \
      --output robacksdalen_2022_test.csv

  B. Compare Predictions with Ground Truth

  # Compare 2022 predictions with ground truth (if available)
  uv run phenocai evaluate compare \
      lonnstorp_2022_test.csv \
      predictions_2022.csv \
      --output-dir prediction_evaluation/2022 \
      --generate-plots

  # Evaluate prediction quality over time
  uv run phenocai evaluate compare \
      robacksdalen_2022_test.csv \
      predictions/robacksdalen_2022 \
      --output-dir prediction_evaluation/robacksdalen_2022

  Phase 6: Advanced Analysis and Fine-Tuning

  A. Fine-tune Model on New Years (if you get good predictions)

  # If predictions for 2022 look good, you can fine-tune the model
  uv run phenocai train fine-tune trained_models/mobilenet_multi_station_full \
      lonnstorp_2022_test.csv \
      --epochs 10 \
      --learning-rate 0.0001 \
      --output-dir trained_models/mobilenet_finetuned_2022

  B. Complete Cross-Station Pipeline (Alternative Automated Approach)

  # Run the complete automated pipeline for cross-station analysis
  uv run phenocai cross-station pipeline \
      --train-stations lonnstorp robacksdalen \
      --eval-stations abisko \
      --years 2024 \
      --annotation-years 2022 2023 2025 \
      --use-heuristics \
      --roi-filter ROI_00 \
      --output-dir cross_station_analysis

  Phase 7: Results Analysis and Monitoring

  # Check training progress with TensorBoard (if you have it installed)
  tensorboard --logdir trained_models/mobilenet_multi_station_full/logs

  # List all trained models
  uv run phenocai train list-models

  # Get training presets for reference
  uv run phenocai train list-presets

  # Check pipeline status
  uv run phenocai pipeline status

  Expected Workflow Execution Order:

  1. Start with Phase 1 (Dataset analysis)
  2. Run Phase 2A (MobileNet training) - This will take the longest
  3. While training runs, prepare Phase 2C (Heuristics analysis)
  4. Run Phase 3A (Evaluate MobileNet)
  5. Optional: Phase 2B (Custom CNN) if you want comparison
  6. Run Phase 4A (Generate predictions)
  7. Run Phase 5 (Evaluate predictions if ground truth available)

  Key Parameters Explained:

  - --preset mobilenet_full: Uses optimized hyperparameters for full training
  - --epochs 50: Full training cycles (will take several hours)
  - --batch-size 32: Good balance for memory and training speed
  - --use-heuristics: Combines ML with rule-based quality detection
  - --roi-filter ROI_00: Ensures cross-station compatibility
  - --save-predictions: Saves predictions for later analysis
  - --generate-plots: Creates evaluation visualizations

  This pipeline will give you comprehensive training, evaluation, and prediction results across
  multiple years and stations!


---

===========================================================
MULTI-STATION DATASET SUMMARY
============================================================

Total records: 8963
Total images: 3209
Stations: lonnstorp, robacksdalen

Per-station breakdown:

lonnstorp:
  Records: 5559
  Images: 1467
  Days: 285
  Snow present: 756
  Discarded: 1326

robacksdalen:
  Records: 3404
  Images: 1742
  Days: 303
  Snow present: 2693
  Discarded: 107

Split distribution:
station  lonnstorp  robacksdalen
split                           
test          1287           719
train         3737          2369
val            535           316

Label distribution by station:
snow_presence  False  True 
station                    
lonnstorp       4803    756
robacksdalen     711   2693

ROI distribution:
  ROI_00: 3198
  ROI_01: 2897
  ROI_06: 967
  ROI_03: 957
  ROI_02: 944

✓ Created multi-station dataset
Dataset saved to: /home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experimental_data/multi_station_lonnstorp_robacksdalen_dataset_2024_splits_20_10.csv

---

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