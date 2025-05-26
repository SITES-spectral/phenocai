# Snow Detection Model Improvement Guide

## Executive Summary

This guide provides specific, actionable recommendations for improving the PhenoCAI snow detection model based on current evaluation results. The model shows strong discriminative ability (AUC 0.88) but suffers from calibration issues and narrow prediction ranges.

## Current Model Analysis

### Strengths
- High AUC-ROC (0.88) indicates good feature learning
- No extreme bias (unlike unbalanced model)
- Cross-station generalization capability

### Key Issues
1. **Narrow prediction range** (0.5-0.6) making threshold selection critical
2. **Over-conservative predictions** leading to low recall
3. **Lack of seasonal variation** in predictions
4. **Limited validation data** for historical predictions

## Immediate Improvements (1-2 weeks)

### 1. Optimal Class Balance Experimentation

Instead of 50/50 balance, experiment with ratios that better reflect reality while avoiding extreme imbalance:

```python
# Recommended experiments
ratios = [
    1.5,  # 40% snow, 60% no-snow
    2.0,  # 33% snow, 67% no-snow  
    3.0,  # 25% snow, 75% no-snow
]

for ratio in ratios:
    phenocai dataset balance input.csv output_ratio_{ratio}.csv --ratio {ratio}
    phenocai train output_ratio_{ratio}.csv --output-dir models/ratio_{ratio}
```

**Expected outcome**: Better calibrated predictions with wider probability ranges

### 2. Implement Temperature Scaling

Post-training calibration to improve probability estimates:

```python
def temperature_scaling(logits, temperature=1.5):
    """Apply temperature scaling to model outputs."""
    return torch.nn.functional.softmax(logits / temperature, dim=1)

# Implementation steps:
1. Use validation set to find optimal temperature
2. Apply to all predictions
3. Re-evaluate threshold performance
```

**Expected outcome**: 15-20% improvement in calibration error

### 3. Weighted Loss Function

Replace binary crossentropy with weighted version:

```python
# In training configuration
class_weights = {0: 1.0, 1: 3.0}  # Emphasize snow class
loss = tf.keras.losses.BinaryFocalCrossentropy(
    alpha=0.25,  # Balance factor
    gamma=2.0,   # Focusing parameter
    from_logits=False
)
```

**Expected outcome**: Better recall without sacrificing too much precision

## Medium-term Improvements (2-4 weeks)

### 4. Temporal Feature Integration

Add temporal information as additional model input:

```python
def create_temporal_model():
    # Image input
    image_input = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(input_tensor=image_input, include_top=False)
    
    # Temporal input (day of year, normalized)
    temporal_input = Input(shape=(2,))  # [sin(doy), cos(doy)]
    
    # Combine features
    image_features = GlobalAveragePooling2D()(base_model.output)
    combined = Concatenate()([image_features, temporal_input])
    
    # Final layers
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([image_input, temporal_input], output)
```

**Expected outcome**: 20-30% improvement in seasonal snow detection

### 5. Multi-Scale ROI Analysis

Process multiple ROIs at different scales:

```python
roi_scales = [
    1.0,   # Original ROI
    0.8,   # 80% of ROI (center crop)
    1.2,   # 120% of ROI (context included)
]

# Average predictions across scales
final_prediction = np.mean([
    predict(extract_roi(image, scale)) 
    for scale in roi_scales
])
```

**Expected outcome**: More robust predictions, especially for edge cases

### 6. Ensemble Methods

Combine models trained with different strategies:

```python
models = {
    'balanced': load_model('balanced_model.keras'),
    'ratio_2': load_model('ratio_2_model.keras'),
    'weighted': load_model('weighted_loss_model.keras'),
}

# Weighted ensemble
weights = {'balanced': 0.3, 'ratio_2': 0.5, 'weighted': 0.2}
ensemble_pred = sum(
    weights[name] * model.predict(image) 
    for name, model in models.items()
)
```

**Expected outcome**: 10-15% improvement in overall performance

## Long-term Improvements (1-2 months)

### 7. Active Learning Pipeline

Focus annotation efforts on uncertain cases:

```python
def identify_uncertain_images(model, unlabeled_images, n=100):
    """Find images where model is most uncertain."""
    predictions = model.predict(unlabeled_images)
    uncertainty = np.abs(predictions - 0.5)  # Distance from decision boundary
    
    # Return indices of most uncertain images
    return np.argsort(uncertainty)[:n]

# Workflow:
1. Predict on unlabeled data
2. Identify 100 most uncertain images
3. Manually annotate these images
4. Retrain model with expanded dataset
5. Repeat monthly
```

**Expected outcome**: Continuous improvement with minimal annotation effort

### 8. Snow Coverage Regression

Instead of binary classification, predict snow coverage percentage:

```python
# Modify annotation to include coverage percentage
annotations = {
    'no_snow': 0.0,
    'traces': 0.1,
    'partial': 0.5,
    'mostly': 0.8,
    'complete': 1.0
}

# Use MSE loss for regression
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

# Convert to binary at inference
snow_present = coverage_prediction > 0.1
```

**Expected outcome**: More nuanced predictions and better threshold selection

### 9. Domain Adaptation

Improve cross-station performance:

```python
# Adversarial domain adaptation
def create_domain_adaptive_model():
    # Shared feature extractor
    feature_extractor = create_base_model()
    
    # Snow classifier
    snow_classifier = create_classifier_head()
    
    # Domain discriminator
    domain_discriminator = create_discriminator()
    
    # Train with gradient reversal layer
    return DomainAdaptiveModel(
        feature_extractor,
        snow_classifier,
        domain_discriminator
    )
```

**Expected outcome**: Better generalization across different camera setups

## Validation Strategy

### 1. Create Comprehensive Test Set

```python
# Stratified sampling across:
- Stations (50% each)
- Seasons (25% each)
- Years (equal representation)
- Weather conditions (if available)

# Minimum 1000 manually verified images
```

### 2. Cross-Reference with Weather Data

```python
def validate_with_weather_data(predictions_df, weather_station_data):
    """Compare predictions with temperature and precipitation data."""
    merged = pd.merge(
        predictions_df,
        weather_station_data,
        on=['date', 'station']
    )
    
    # Analyze correlation
    snow_temp_correlation = merged[
        merged['temperature'] < 2  # Below 2°C
    ]['snow_predicted'].mean()
    
    return validation_metrics
```

### 3. Human-in-the-Loop Validation

```python
# Web interface for validation
def create_validation_interface():
    """Create Streamlit app for manual validation."""
    st.title("Snow Detection Validation")
    
    # Show uncertain predictions
    uncertain = get_uncertain_predictions()
    
    image = st.image(uncertain['image_path'])
    prediction = st.text(f"Model prediction: {uncertain['probability']:.3f}")
    
    # Human feedback
    human_label = st.radio("Actual snow presence:", ['No Snow', 'Snow', 'Uncertain'])
    
    if st.button("Submit"):
        save_human_feedback(uncertain['image_id'], human_label)
```

## Performance Targets

### Short-term (1 month)
- F1 Score: 0.75 (from current 0.45)
- Calibration Error: < 0.1
- Seasonal accuracy variance: < 10%

### Medium-term (3 months)
- F1 Score: 0.85
- Cross-station accuracy variance: < 5%
- Operational threshold stability: ±0.05

### Long-term (6 months)
- F1 Score: 0.90+
- Coverage percentage MAE: < 0.15
- Fully automated operational deployment

## Implementation Priority

1. **Week 1-2**: Class ratio experiments + temperature scaling
2. **Week 3-4**: Weighted loss + temporal features
3. **Month 2**: Ensemble methods + active learning setup
4. **Month 3+**: Advanced techniques + continuous improvement

## Code Examples

### Complete Training Script with Improvements

```python
# improved_training.py
import tensorflow as tf
from phenocai.models.architectures import create_temporal_model
from phenocai.data.balanced_loader import BalancedDataLoader

# Configuration
config = {
    'class_ratio': 2.0,
    'use_focal_loss': True,
    'temperature_scaling': 1.5,
    'ensemble_size': 3,
    'temporal_features': True
}

# Data preparation
loader = BalancedDataLoader(ratio=config['class_ratio'])
train_ds, val_ds = loader.prepare_datasets()

# Model creation
if config['temporal_features']:
    model = create_temporal_model()
else:
    model = create_standard_model()

# Compilation with improvements
if config['use_focal_loss']:
    loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
else:
    loss = tf.keras.losses.BinaryFocalCrossentropy()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)

# Training with callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    TemperatureScalingCallback(validation_data=val_ds)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)
```

## Monitoring and Evaluation

### Metrics Dashboard

```python
# monitoring.py
def create_monitoring_dashboard():
    """Track model performance over time."""
    metrics = {
        'daily_predictions': count_daily_predictions(),
        'confidence_distribution': get_confidence_histogram(),
        'threshold_stability': calculate_threshold_drift(),
        'cross_station_variance': measure_station_differences(),
        'seasonal_patterns': analyze_seasonal_accuracy()
    }
    
    # Generate daily report
    generate_performance_report(metrics)
    
    # Alert on anomalies
    if metrics['threshold_stability'] > 0.1:
        send_alert("Threshold drift detected")
```

## Conclusion

The current model provides a solid foundation with good feature learning capabilities. The improvements outlined in this guide address the main limitations:

1. **Calibration issues** → Temperature scaling and better class ratios
2. **Narrow predictions** → Focal loss and ensemble methods
3. **Seasonal blindness** → Temporal feature integration
4. **Limited validation** → Active learning and weather data correlation

Following this guide should lead to a production-ready model with F1 score >0.85 and robust cross-station performance within 2-3 months.

## Resources

- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [Temperature Scaling](https://arxiv.org/abs/1706.04599)
- [Domain Adaptation](https://arxiv.org/abs/1505.07818)
- [Active Learning Strategies](https://arxiv.org/abs/1904.04088)

---

*Last updated: 2025-01-25*  
*Contact: PhenoCAI Development Team*