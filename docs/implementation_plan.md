# PhenoCAI Implementation Plan

## Phase 1: Project Setup and Core Infrastructure
1. **Environment and Project Structure**
   - [x] Set up environment variables (env.sh)
   - [x] Create basic project structure
   - [x] Development environment with uv (already configured)
   - [ ] Initialize git repository

2. **Configuration Module**
   - [ ] Implement `setup.py` with all required paths and parameters
   - [ ] Add environment variable support
   - [ ] Implement `setup_directories()` function
   - [ ] Add configuration validation

3. **Utility Functions**
   - [ ] Implement `load_roi_config_from_yaml()`
   - [ ] Implement `parse_image_filename()`
   - [ ] Implement `get_roi_points_from_config()`
   - [ ] Implement `extract_roi_sub_image()`
   - [ ] Implement `load_image()`
   - [ ] Add unit tests for utility functions

## Phase 2: Data Management
1. **Master Dataset Creation**
   - [ ] Implement `create_master_annotation_dataframe()`
   - [ ] Add YAML file scanning functionality
   - [ ] Implement train/test split logic
   - [ ] Add data validation and error handling
   - [ ] Create data loading utilities

2. **Data Loading System**
   - [ ] Implement `load_data_for_model()`
   - [ ] Add support for both DataFrame and YAML scanning
   - [ ] Implement preprocessing pipeline
   - [ ] Add metadata tracking
   - [ ] Create data validation tests

## Phase 3: Heuristic Implementation
1. **Heuristic Detectors**
   - [ ] Implement `heuristic_snow_detection()`
   - [ ] Implement `heuristic_discard_detection()`
   - [ ] Add HSV color thresholding
   - [ ] Implement blur detection
   - [ ] Add illumination/contrast checks
   - [ ] Create validation tests for heuristics

## Phase 4: Model Development
1. **Model Architecture**
   - [ ] Implement `build_mobilenetv2_model()`
   - [ ] Implement `build_simple_cnn_model()`
   - [ ] Add transfer learning support
   - [ ] Create model configuration system

2. **Training System**
   - [ ] Implement `train_model_runner()`
   - [ ] Add early stopping
   - [ ] Implement fine-tuning logic
   - [ ] Add model checkpointing
   - [ ] Create training monitoring

## Phase 5: Evaluation System
1. **Metrics and Evaluation**
   - [ ] Implement `evaluate_model_performance()`
   - [ ] Add comprehensive metrics calculation
   - [ ] Create visualization tools
   - [ ] Implement stratified evaluation
   - [ ] Add performance reporting

## Phase 6: Inference System
1. **Prediction Pipeline**
   - [ ] Implement `apply_models_to_new_images()`
   - [ ] Add batch processing support
   - [ ] Implement YAML output generation
   - [ ] Add progress tracking
   - [ ] Create validation checks

## Phase 7: Scripts and Automation
1. **Utility Scripts**
   - [ ] Create `01_create_master_manifest.py`
   - [ ] Create `02_train_models.py`
   - [ ] Create `03_evaluate_models.py`
   - [ ] Create `04_apply_models_to_new_data.py`
   - [ ] Add command-line interfaces

## Phase 8: Testing and Documentation
1. **Testing**
   - [ ] Write unit tests for all modules
   - [ ] Create integration tests
   - [ ] Add performance benchmarks
   - [ ] Implement CI/CD pipeline

2. **Documentation**
   - [ ] Write API documentation
   - [ ] Create usage examples
   - [ ] Add configuration guide
   - [ ] Write troubleshooting guide

## Phase 9: Future Expansion Preparation
1. **Extensibility**
   - [ ] Design plugin system for new flags
   - [ ] Create model registry
   - [ ] Add configuration templates
   - [ ] Document extension process

## Timeline and Dependencies
- Phase 1: 1 week
- Phase 2: 2 weeks
- Phase 3: 1 week
- Phase 4: 2 weeks
- Phase 5: 1 week
- Phase 6: 1 week
- Phase 7: 1 week
- Phase 8: 2 weeks
- Phase 9: 1 week

Total estimated time: 12 weeks

## Priority Order
1. Core infrastructure (Phase 1)
2. Data management (Phase 2)
3. Model development (Phase 4)
4. Evaluation system (Phase 5)
5. Heuristic implementation (Phase 3)
6. Inference system (Phase 6)
7. Scripts and automation (Phase 7)
8. Testing and documentation (Phase 8)
9. Future expansion (Phase 9)

## Success Criteria
- All core functionality implemented and tested
- Documentation complete and up-to-date
- Test coverage > 80%
- Performance benchmarks established
- Extension system documented and tested 