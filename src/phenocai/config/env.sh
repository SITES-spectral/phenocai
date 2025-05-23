#!/bin/bash

# PhenoCAI Environment Configuration

# Project and Station Information
export PHENOCAI_PROJECT_ROOT="/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning"
export PHENOCAI_CURRENT_STATION="lonnstorp"
export PHENOCAI_CURRENT_INSTRUMENT="LON_AGR_PL01_PHE01"
export PHENOCAI_CURRENT_YEAR="2024"

# Data Directory Structure
export PHENOCAI_DATA_DIR="${PHENOCAI_PROJECT_ROOT}/${PHENOCAI_CURRENT_STATION}"
export PHENOCAI_MASTER_ANNOTATION_POOL_DIR="${PHENOCAI_DATA_DIR}/master_annotation_pool"
export PHENOCAI_EXPERIMENTAL_DATA_DIR="${PHENOCAI_DATA_DIR}/experimental_data"
export PHENOCAI_MASTER_DF_WITH_SPLITS_PATH="${PHENOCAI_EXPERIMENTAL_DATA_DIR}/master_annotations_with_splits.csv"
export PHENOCAI_ANNOTATION_ROOT_DIR_FOR_HEURISTICS="${PHENOCAI_EXPERIMENTAL_DATA_DIR}/heuristic_train_pool"

# Image and Configuration Paths
export PHENOCAI_IMAGE_BASE_DIR="/home/jobelund/lu2024-12-46/SITES/Spectral/data/${PHENOCAI_CURRENT_STATION}/phenocams/products/${PHENOCAI_CURRENT_INSTRUMENT}/L1/${PHENOCAI_CURRENT_YEAR}"
export PHENOCAI_ROI_CONFIG_FILE_PATH="/home/jobelund/lu2024-12-46/SITES/Spectral/apps/phenocai/src/phenocai/config/stations.yaml"

# Model and Output Directories
export PHENOCAI_MODEL_OUTPUT_DIR="${PHENOCAI_DATA_DIR}/trained_models/experimental_models_final_df_split"
export PHENOCAI_OUTPUT_DIR_FOR_NEW_ANNOTATIONS="${PHENOCAI_DATA_DIR}/model_generated_annotations_df_split"

# Optional: Add any additional environment variables here
# export PHENOCAI_DEBUG="true"
# export PHENOCAI_LOG_LEVEL="INFO"

# Print confirmation message
echo "PhenoCAI environment variables have been set up."
echo "Current station: ${PHENOCAI_CURRENT_STATION}"
echo "Current instrument: ${PHENOCAI_CURRENT_INSTRUMENT}"
echo "Current year: ${PHENOCAI_CURRENT_YEAR}"
echo "Data directory: ${PHENOCAI_DATA_DIR}"
echo "Image base directory: ${PHENOCAI_IMAGE_BASE_DIR}"
echo "ROI config file path: ${PHENOCAI_ROI_CONFIG_FILE_PATH}"
echo "Model output directory: ${PHENOCAI_MODEL_OUTPUT_DIR}"
echo "Output directory for new annotations: ${PHENOCAI_OUTPUT_DIR_FOR_NEW_ANNOTATIONS}"