# Complete PhenoCAI Workflow Guide

## The Full Journey: From Photos to Knowledge 🌟

This guide shows how all the pieces fit together in the PhenoCAI system.

```mermaid
graph TB
    subgraph Input
        A[📷 Phenocam Photos<br/>Every 3 hours]
        B[👤 Human Labels<br/>Snow, Flags]
    end
    
    subgraph Data Preparation
        C[📊 Create Dataset]
        D[✂️ Split Data<br/>70/10/20]
        E[🔍 Quality Analysis]
    end
    
    subgraph Training
        F[🧠 Neural Network<br/>MobileNetV2]
        G[📚 Learn Patterns]
        H[💾 Save Model]
    end
    
    subgraph Evaluation
        I[🎯 Test Performance]
        J[📈 Calculate Metrics]
        K[✅ Validate Results]
    end
    
    subgraph Production
        L[🚀 Deploy Model]
        M[📸 New Photos]
        N[🏷️ Auto-Label]
        O[📊 Research Data]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style H fill:#c5e1a5
    style K fill:#4caf50,color:#fff
    style O fill:#ffd700
```

## Quick Reference: Commands for Each Stage

### 1️⃣ Setup and Configuration
```bash
# Install and configure
uv sync
source src/phenocai/config/env.sh

# Check setup
uv run phenocai info
uv run phenocai config validate
```

### 2️⃣ Data Preparation
```bash
# Create dataset with train/test/val splits
uv run phenocai dataset create --output data.csv \
    --test-size 0.2 \
    --val-size 0.1

# The dataset now includes:
# - file_path: Full path with day-of-year subdirectory
# - has_flags: Boolean for quick quality filtering
# - split: train/test/val assignment

# Analyze quality
python scripts/analyze_quality_issues.py data.csv

# Filter if needed
uv run phenocai dataset filter data.csv clean_data.csv \
    --exclude-flags fog high_brightness
```

### 3️⃣ Training
```bash
# Train model with MobileNetV2
uv run phenocai train model clean_data.csv \
    --model-type mobilenet \
    --epochs 20 \
    --batch-size 32

# The model learns to predict:
# - snow_presence: Binary classification
# - Quality issues for discard detection
```

### 4️⃣ Evaluation
```bash
# Evaluate model
uv run phenocai evaluate model saved_model.h5 test_data.csv \
    --save-predictions \
    --generate-plots

# Generates:
# - Accuracy, precision, recall metrics
# - Confusion matrices
# - Performance by quality condition
```

### 5️⃣ Prediction
```bash
# Predict single image
uv run phenocai predict apply saved_model.h5 \
    --image path/to/image.jpg

# Batch process entire directory
uv run phenocai predict batch saved_model.h5 \
    --directory /2024/ \
    --output-dir predictions/ \
    --format yaml

# Process specific date range
uv run phenocai predict batch saved_model.h5 \
    --start-day 100 \
    --end-day 200 \
    --year 2024 \
    --output-dir seasonal_predictions/

# Export predictions
uv run phenocai predict export predictions/ \
    --format csv \
    --output results.csv
```

## Key Decision Points 🤔

### 1. Data Quality Decisions

```mermaid
graph TD
    A[Dataset Analysis] --> B{Quality Issues?}
    B -->|Many Issues<br/>90%| C[Filter Dataset]
    B -->|Few Issues<br/>10%| D[Use Full Dataset]
    
    C --> E[Choose Flags<br/>to Exclude]
    E --> F[fog, high_brightness,<br/>lens_water_drops]
    
    style B fill:#fff3e0
    style C fill:#ff9800,color:#fff
    style D fill:#4caf50,color:#fff
```

### 2. Model Selection

```mermaid
graph TD
    A[Choose Model] --> B{Priority?}
    B -->|Accuracy| C[MobileNetV2<br/>Transfer Learning]
    B -->|Speed| D[Simple CNN<br/>From Scratch]
    B -->|Both| E[Fine-tuned<br/>MobileNetV2]
    
    style C fill:#4caf50,color:#fff
    style D fill:#2196f3,color:#fff
    style E fill:#9c27b0,color:#fff
```

### 3. Threshold Settings

```mermaid
graph TD
    A[Confidence Threshold] --> B{Use Case?}
    B -->|Research<br/>High Accuracy| C[>80%<br/>Conservative]
    B -->|Monitoring<br/>Catch All| D[>50%<br/>Liberal]
    B -->|Balanced| E[>65%<br/>Middle Ground]
    
    style C fill:#f44336,color:#fff
    style D fill:#4caf50,color:#fff
    style E fill:#ff9800,color:#fff
```

## Real-World Example: Complete Pipeline

### Processing a Full Year of Lönnstorp Data

```bash
# 1. Setup environment
source src/phenocai/config/env_lonnstorp.sh
python src/phenocai/config/setup.py

# 1b. Check available instruments and switch if needed
uv run phenocai station instruments
uv run phenocai station switch lonnstorp --instrument LON_AGR_PL01_PHE01

# 2. Create dataset with annotations (auto-named)
uv run phenocai dataset create \
    --test-size 0.2 \
    --val-size 0.1
# Creates: lonnstorp_PHE01_dataset_2024_splits_20_10.csv

# 3. Check data quality
python scripts/analyze_quality_issues.py lonnstorp_PHE01_dataset_2024_splits_20_10.csv
# Output: 1,547 total samples
#   - 404 clean (no flags): 26.1%
#   - 1,143 with quality issues: 73.9%

# 4. Train on clean subset first (auto-generates filename)
uv run phenocai dataset filter lonnstorp_PHE01_dataset_2024_splits_20_10.csv --no-flags
# Creates: lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv
    
uv run phenocai train model lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv \
    --model-type mobilenet \
    --epochs 20 \
    --output-dir models/clean_baseline/

# 5. Evaluate performance
uv run phenocai evaluate model models/clean_baseline/final_model.h5 \
    lonnstorp_PHE01_dataset_2024_splits_20_10_clean_filtered.csv --split test

# 6. Apply to full year of new images
uv run phenocai predict batch models/clean_baseline/final_model.h5 \
    --year 2024 \
    --output-dir predictions/2024/ \
    --format yaml \
    --use-heuristics

# 7. Export results for analysis
uv run phenocai predict export predictions/2024/ \
    --format csv \
    --output lonnstorp_PHE01_2024_predictions.csv
```

### Results You Can Expect

```yaml
# Example prediction output (predictions/2024/102/predictions.yaml)
lonnstorp_LON_AGR_PL01_PHE01_2024_102_20240411_080003.jpg:
  - roi_name: ROI_00
    snow_presence: false
    snow_probability: 0.12
    confidence: 0.88
    quality_flags: []
    discard: false
    has_flags: false
    
  - roi_name: ROI_01
    snow_presence: true
    snow_probability: 0.95
    confidence: 0.95
    quality_flags: ['partial_snow']
    discard: false
    has_flags: true
```

## Common Workflows

### Workflow A: Clean Data First-Time Training

```mermaid
graph LR
    A[Full Dataset] --> B[Heavy Filtering]
    B --> C[Clean Subset<br/>26% of data]
    C --> D[Initial Training]
    D --> E[Good Baseline<br/>Model]
    
    style C fill:#4caf50,color:#fff
    style E fill:#ffd700
```

**Best for**: Getting started quickly with reliable results

### Workflow B: Progressive Training

```mermaid
graph LR
    A[Clean Data] --> B[Train Model 1]
    B --> C[Add Medium<br/>Quality Data]
    C --> D[Train Model 2]
    D --> E[Add All Data]
    E --> F[Train Model 3]
    
    style B fill:#c5e1a5
    style D fill:#81c784
    style F fill:#4caf50,color:#fff
```

**Best for**: Gradually improving model robustness

### Workflow C: Condition-Specific Models

```mermaid
graph TD
    A[All Data] --> B[Group by Condition]
    B --> C[Clear Weather<br/>Model]
    B --> D[Fog<br/>Model]
    B --> E[Bright<br/>Model]
    
    F[New Image] --> G{Condition?}
    G -->|Clear| C
    G -->|Foggy| D
    G -->|Bright| E
    
    style C fill:#4caf50,color:#fff
    style D fill:#e1f5fe
    style E fill:#fff9c4
```

**Best for**: Maximum accuracy across all conditions

## Performance Expectations 📊

### By Data Quality

```mermaid
graph TD
    subgraph Expected Accuracy
        A[Clean Images] --> B[90-95%]
        C[Light Issues] --> D[80-85%]
        E[Heavy Issues] --> F[60-70%]
        G[Multiple Issues] --> H[40-50%]
    end
    
    style B fill:#4caf50,color:#fff
    style D fill:#81c784
    style F fill:#ff9800,color:#fff
    style H fill:#f44336,color:#fff
```

### By Training Data Size

```mermaid
graph LR
    A[500 samples] --> B[~70% accuracy]
    C[2,000 samples] --> D[~80% accuracy]
    E[5,000 samples] --> F[~85% accuracy]
    G[10,000+ samples] --> H[~90% accuracy]
    
    style B fill:#ffecb3
    style D fill:#fff176
    style F fill:#aed581
    style H fill:#4caf50,color:#fff
```

## Troubleshooting Guide 🔧

```mermaid
graph TD
    A[Problem] --> B{Type?}
    
    B -->|Low Accuracy| C[Check Data Quality]
    C --> D[More Training Data]
    C --> E[Better Filtering]
    
    B -->|Slow Training| F[Reduce Batch Size]
    F --> G[Use Smaller Model]
    
    B -->|Overfitting| H[Add Dropout]
    H --> I[Data Augmentation]
    H --> J[Simpler Model]
    
    B -->|Poor on New Data| K[Check Distribution]
    K --> L[Retrain Regularly]
    
    style A fill:#f44336,color:#fff
    style D fill:#4caf50,color:#fff
    style E fill:#4caf50,color:#fff
    style G fill:#4caf50,color:#fff
    style I fill:#4caf50,color:#fff
    style L fill:#4caf50,color:#fff
```

## Tips for Success 🎯

### 1. Start Simple
- ✅ Use filtered dataset first
- ✅ Train basic model
- ✅ Get baseline results
- ✅ Then add complexity

### 2. Monitor Everything
- 📊 Track accuracy over time
- 📈 Log confidence scores
- 🔍 Review random samples
- 📝 Document changes

### 3. Iterate Frequently
- 🔄 Small improvements
- 🧪 Test each change
- 📊 Measure impact
- 🎯 Keep what works

## The Big Picture 🌍

```mermaid
graph TD
    A[Individual Photos] --> B[PhenoCAI System]
    B --> C[Automated Analysis]
    C --> D[Large-Scale Data]
    D --> E[Climate Research]
    E --> F[Environmental Understanding]
    F --> G[Better Decisions]
    
    style B fill:#9c27b0,color:#fff
    style E fill:#4caf50,color:#fff
    style G fill:#ffd700
```

## Remember: You're Part of Something Important! 🌟

Every photo labeled, every model trained, and every prediction made contributes to:
- 🌡️ Understanding climate change
- 🌱 Monitoring ecosystem health
- 📊 Providing data for research
- 🌍 Protecting our environment

## Next Steps

1. **Practice**: Try the commands with sample data
2. **Experiment**: Test different settings
3. **Learn**: Understand what works best
4. **Contribute**: Share your findings

## Need Help?

- 📖 Review individual workflow guides
- 💬 Ask the development team
- 🐛 Report issues on GitHub
- 📚 Check the documentation

Good luck with your PhenoCAI journey! 🚀