# Prediction Workflow: Using Your Trained Model on New Photos

## Overview

Now that we have a trained and tested model, it's time to put it to work! This is like having a trained assistant who can label thousands of photos automatically.

```mermaid
graph TD
    A[New Photos] --> B[Trained Model]
    B --> C[Predictions]
    C --> D[Save Results]
    D --> E[Scientists Use Data]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#c5e1a5
    style D fill:#81c784
    style E fill:#4caf50,color:#fff
```

## The Prediction Pipeline

### Step 1: Collect New Photos

```mermaid
graph LR
    A[Camera] --> B[Today's Photos]
    B --> C[No Labels Yet]
    C --> D[Ready for<br/>Prediction]
    
    style A fill:#e3f2fd
    style B fill:#fff9c4
    style C fill:#ffecb3
    style D fill:#c5e1a5
```

These photos are completely new - taken after the model was trained.

### Step 2: Load the Model

```mermaid
graph TD
    A[Saved Model File<br/>model.h5] --> B[Load into Memory]
    B --> C[Model Ready]
    C --> D[Contains All<br/>Learned Knowledge]
    
    style A fill:#81c784
    style B fill:#f3e5f5
    style C fill:#9c27b0,color:#fff
    style D fill:#e1bee7
```

### Step 3: Process Images

```mermaid
graph TD
    A[New Image] --> B[Extract ROIs]
    B --> C[ROI_00<br/>Full Image]
    B --> D[ROI_01<br/>Sky Area]
    B --> E[ROI_02<br/>Ground Area]
    
    C --> F[Predict Each ROI]
    D --> F
    E --> F
    
    F --> G[Combine Results]
    
    style A fill:#e3f2fd
    style F fill:#f3e5f5
    style G fill:#c5e1a5
```

### Step 4: Make Predictions

For each image and ROI, the model outputs probabilities:

```mermaid
graph LR
    A[ROI Image] --> B[Model]
    B --> C[Snow: 85%<br/>No Snow: 15%]
    C --> D{Threshold<br/>Check}
    D -->|>50%| E[Prediction:<br/>Snow Present]
    
    style B fill:#f3e5f5
    style C fill:#fff9c4
    style E fill:#4caf50,color:#fff
```

## Batch Processing for Efficiency

Instead of one image at a time, we process in batches:

```mermaid
graph TD
    A[1000 New Images] --> B[Batch 1<br/>Images 1-32]
    A --> C[Batch 2<br/>Images 33-64]
    A --> D[...]
    A --> E[Batch 32<br/>Images 993-1000]
    
    B --> F[Process<br/>in Parallel]
    C --> F
    E --> F
    
    F --> G[All Predictions<br/>Complete]
    
    style A fill:#e3f2fd
    style F fill:#f3e5f5
    style G fill:#81c784
```

**Why Batches?**
- Much faster than one-by-one
- Better memory usage
- GPU processes multiple images at once

## Confidence Scores

The model doesn't just say "yes" or "no" - it gives confidence levels:

```mermaid
graph TD
    subgraph High Confidence
        A[Clear Snow] --> B[Snow: 95%]
    end
    
    subgraph Medium Confidence
        C[Light Snow] --> D[Snow: 72%]
    end
    
    subgraph Low Confidence
        E[Ambiguous] --> F[Snow: 53%]
    end
    
    style B fill:#4caf50,color:#fff
    style D fill:#ffc107
    style F fill:#ff9800,color:#fff
```

### Using Confidence Thresholds

```mermaid
graph TD
    A[Confidence Score] --> B{Threshold}
    B -->|>80%| C[High Confidence<br/>Auto-label]
    B -->|50-80%| D[Medium Confidence<br/>Flag for Review]
    B -->|<50%| E[Low Confidence<br/>Human Check]
    
    style C fill:#4caf50,color:#fff
    style D fill:#ffc107
    style E fill:#f44336,color:#fff
```

## Handling Quality Issues

PhenoCAI automatically detects and handles problematic images:

```mermaid
graph TD
    A[New Image] --> B{Quality Check}
    B -->|Clean| C[Make Prediction]
    B -->|Has Issues| D[Quality Flags Added]
    
    D --> E[Weather Issues<br/>fog, rain, snow]
    D --> F[Camera Issues<br/>blur, lens_dirt]
    D --> G[Lighting Issues<br/>too_bright, shadows]
    
    E --> H[Flag & Predict]
    F --> I[Flag & Maybe Discard]
    G --> J[Flag & Adjust]
    
    style C fill:#4caf50,color:#fff
    style D fill:#ffecb3
    style I fill:#ff9800,color:#fff
```

### Automatic Discard Detection

Some ROIs are automatically marked for discard:

```mermaid
graph TD
    A[ROI Quality Check] --> B{Discard Criteria}
    B -->|Unusable| C[Discard]
    B -->|Too Dark| C
    B -->|Too Bright| C
    B -->|Severe Blur| C
    B -->|Lens Obstruction| C
    B -->|Otherwise| D[Keep]
    
    style C fill:#f44336,color:#fff
    style D fill:#4caf50,color:#fff
```

### Quality Flags Detected

PhenoCAI detects 20+ quality issues:

| Category | Flags | Impact |
|----------|-------|--------|
| **Weather** | fog, rain, snow, clouds | May affect accuracy |
| **Camera** | blur, lens_water_drops, lens_dirt, lens_snow | Often need discard |
| **Lighting** | high_brightness, low_brightness, shadows, glare | Affects visibility |
| **Other** | heterogeneous_illumination, sun_altitude_low | Special handling |

## Output Formats

### Individual Annotation Files

```yaml
filename: new_image_2024_150_20240529_080003.jpg
created: '2024-05-29T10:15:00'
model_path: 'trained_models/mobilenet/final_model.h5'
threshold: 0.5
station: 'lonnstorp'
instrument: 'LON_AGR_PL01_PHE01'
year: '2024'
day_of_year: '150'
status: 'completed'
annotations:
  - roi_name: ROI_00
    discard: false
    snow_presence: true
    flags: []
    not_needed: false
    snow_probability: 0.87
    confidence: 0.74
    model_predicted: true
  - roi_name: ROI_01
    discard: true
    snow_presence: false
    flags: ['blur', 'low_brightness']
    not_needed: false
    snow_probability: 0.08
    confidence: 0.84
    model_predicted: true
```

### Batch Results CSV

```mermaid
graph TD
    A[Predictions] --> B[CSV File]
    B --> C[Columns:<br/>• filename<br/>• roi_name<br/>• snow_prediction<br/>• confidence<br/>• timestamp]
    
    style B fill:#81c784
    style C fill:#e8f5e9
```

## Real-Time vs Batch Processing

```mermaid
graph TD
    subgraph Real-Time
        A[New Photo] --> B[Immediate<br/>Prediction]
        B --> C[Result in<br/>Seconds]
    end
    
    subgraph Batch
        D[Day's Photos] --> E[Process<br/>Together]
        E --> F[Results in<br/>Minutes]
    end
    
    style C fill:#4caf50,color:#fff
    style F fill:#81c784
```

**When to use each:**
- **Real-Time**: Urgent monitoring, alerts
- **Batch**: Daily processing, research analysis

## Quality Control in Production

### Monitoring Model Performance

```mermaid
graph TD
    A[Predictions] --> B[Random Sample<br/>10%]
    B --> C[Human Review]
    C --> D{Accurate?}
    D -->|Yes| E[Continue]
    D -->|No| F[Investigate]
    
    F --> G[Retrain if Needed]
    
    style E fill:#4caf50,color:#fff
    style F fill:#ff9800,color:#fff
```

### Drift Detection

Models can become less accurate over time:

```mermaid
graph LR
    A[Month 1<br/>90% Accurate] --> B[Month 3<br/>87% Accurate]
    B --> C[Month 6<br/>82% Accurate]
    C --> D[Time to Retrain!]
    
    style A fill:#4caf50,color:#fff
    style C fill:#ff9800,color:#fff
    style D fill:#f44336,color:#fff
```

**Causes of drift:**
- Camera gets dirty
- Seasonal changes
- Camera adjustments

## Prediction Commands (Fully Implemented)

The prediction system is now fully operational with quality-aware predictions and batch processing capabilities.

```bash
# Set up environment
source src/phenocai/config/env.sh

# Predict single image with quality detection
uv run phenocai predict apply model.h5 \
    --image path/to/new_photo.jpg

# Batch process directory with heuristics
uv run phenocai predict batch model.h5 \
    --directory /2024/ \
    --output-dir predictions/ \
    --format yaml \
    --use-heuristics

# Process specific date range
uv run phenocai predict batch model.h5 \
    --start-day 150 \
    --end-day 180 \
    --year 2024 \
    --output-dir seasonal_predictions/

# Export predictions to multiple formats
uv run phenocai predict export predictions/ \
    --format csv \
    --output predictions.csv
# Supported formats: yaml, csv, json
```

The system includes:
- Automatic quality flag detection (blur, brightness issues, etc.)
- Discard recommendations for poor quality images
- Confidence scores for all predictions
- Efficient batch processing with progress tracking
- Support for processing entire years of phenocam data

## Integration with Research Workflow

```mermaid
graph TD
    A[Raw Photos] --> B[PhenoCAI<br/>Predictions]
    B --> C[Database]
    C --> D[Research Analysis]
    D --> E[Scientific Papers]
    
    B --> F[Quality Flags]
    F --> G[Camera<br/>Maintenance]
    
    style B fill:#9c27b0,color:#fff
    style D fill:#4caf50,color:#fff
    style E fill:#ffd700
```

## Best Practices

### 1. Version Control

```mermaid
graph LR
    A[Model v1.0] --> B[Predictions<br/>Jan-Mar]
    C[Model v1.1] --> D[Predictions<br/>Apr-Jun]
    E[Model v2.0] --> F[Predictions<br/>Jul-Dec]
    
    style A fill:#e1f5fe
    style C fill:#c5e1a5
    style E fill:#ffccbc
```

Always track which model version made each prediction!

### 2. Confidence Thresholds

- **Research**: Use high threshold (>80%) for accuracy
- **Monitoring**: Use lower threshold (>50%) to catch more cases
- **Always save**: Keep confidence scores for later analysis

### 3. Regular Validation

```mermaid
graph TD
    A[Weekly] --> B[Check Random<br/>Samples]
    C[Monthly] --> D[Full Performance<br/>Review]
    E[Quarterly] --> F[Consider<br/>Retraining]
    
    style B fill:#e1f5fe
    style D fill:#fff9c4
    style F fill:#ffccbc
```

## Troubleshooting Common Issues

### Issue: Slow Predictions
- **Solution**: Increase batch size
- **Solution**: Use GPU if available
- **Solution**: Reduce image resolution

### Issue: Low Confidence Scores
- **Solution**: Check image quality
- **Solution**: Verify ROI extraction
- **Solution**: Consider retraining

### Issue: Unexpected Results
- **Solution**: Compare with training data
- **Solution**: Check for camera changes
- **Solution**: Validate preprocessing

## Success Metrics

Your prediction system is working well when:

```mermaid
graph TD
    A[Success Indicators] --> B[Processes 1000s<br/>of images daily]
    A --> C[>85% accuracy<br/>on spot checks]
    A --> D[Consistent<br/>performance]
    A --> E[Useful for<br/>research]
    
    style B fill:#4caf50,color:#fff
    style C fill:#4caf50,color:#fff
    style D fill:#4caf50,color:#fff
    style E fill:#4caf50,color:#fff
```

## Summary: From Photos to Science

```mermaid
graph LR
    A[Photos] --> B[PhenoCAI]
    B --> C[Data]
    C --> D[Research]
    D --> E[Understanding<br/>Climate Change]
    
    style B fill:#9c27b0,color:#fff
    style E fill:#4caf50,color:#fff
```

## Checklist for Production Use

- [ ] Model tested and validated
- [ ] Batch processing configured
- [ ] Output format decided
- [ ] Confidence thresholds set
- [ ] Quality control process in place
- [ ] Version tracking system
- [ ] Performance monitoring
- [ ] Backup and recovery plan

## Congratulations!

You now understand the complete PhenoCAI workflow:
1. **Prepare** → Organize and label data
2. **Train** → Teach the model
3. **Evaluate** → Test performance
4. **Predict** → Use on new data

Your trained model is now ready to help scientists monitor environmental changes automatically!