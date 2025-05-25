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
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
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
    print(f"  Positive (snow): {report['original']['positive']} ({report['original']['positive_pct']:.1f}%)")
    print(f"  Negative (no snow): {report['original']['negative']} ({report['original']['negative_pct']:.1f}%)")
    
    print(f"\nBalanced Dataset:")
    print(f"  Total samples: {report['balanced']['total']}")
    print(f"  Positive (snow): {report['balanced']['positive']} ({report['balanced']['positive_pct']:.1f}%)")
    print(f"  Negative (no snow): {report['balanced']['negative']} ({report['balanced']['negative_pct']:.1f}%)")
    
    if 'original_by_split' in report:
        print(f"\nBy Split:")
        for split in report['original_by_split']:
            orig = report['original_by_split'][split]
            if split in report['balanced_by_split']:
                bal = report['balanced_by_split'][split]
                print(f"\n  {split}:")
                print(f"    Original: {orig['total']} samples ({orig['positive_pct']:.1f}% positive)")
                print(f"    Balanced: {bal['total']} samples ({bal['positive_pct']:.1f}% positive)")
    
    # Save balanced dataset
    balanced_df.to_csv(output_csv, index=False)
    logger.info(f"Saved balanced dataset to {output_csv}")
    
    # Save report
    report_path = Path(output_csv).with_suffix('.balance_report.json')
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_csv