"""
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

from .dataset_balancer import (
    DatasetBalancer,
    balance_dataset_from_csv
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
    'filter_dataset_by_criteria',
    'DatasetBalancer',
    'balance_dataset_from_csv'
]