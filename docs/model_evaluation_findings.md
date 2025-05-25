# PhenoCAI Model Evaluation Findings and Recommendations

**Date**: May 25, 2025  
**Model**: MobileNetV2 Transfer Learning for Snow Detection  
**Dataset**: Multi-station Lönnstorp-Röbäcksdalen 2024 Dataset

## Executive Summary

The trained MobileNetV2 model demonstrates strong discriminative ability (AUC 0.90+) but exhibits extreme prediction bias towards the negative class when using the default threshold (0.5). This document presents findings from the model evaluation and provides recommendations for improving model performance and deployment.

## Key Findings

### 1. Model Performance Metrics

#### Validation Set Results
- **Accuracy**: 64.9%
- **AUC**: 0.922 (excellent discrimination)
- **Precision**: 100% (but with very few positive predictions)
- **Recall**: 5.4% (missing 94.6% of snow cases)
- **F1 Score**: 0.102 (poor due to low recall)

#### Test Set Results
- **Accuracy**: 65.9%
- **AUC**: 0.901
- **Precision**: 78.3%
- **Recall**: 5.1%
- **F1 Score**: 0.095

### 2. Prediction Distribution Problem

With default threshold (0.5):
- **Validation Set**: 98% predicted as "no snow", only 2% as "snow"
- **Test Set**: 97.7% predicted as "no snow", only 2.3% as "snow"
- **Actual Distribution**: ~35-37% of images contain snow

This extreme imbalance indicates the model outputs very low probabilities for the positive class.

### 3. Optimal Threshold Analysis

By evaluating different thresholds, we found:
- **Optimal Threshold**: 0.10
- **F1 Score at Optimal Threshold**: 
  - Validation: 0.829
  - Test: 0.840
- This represents an 8x improvement in F1 score

### 4. Root Cause Analysis

The validation metrics showed 0.0 for precision/recall during training because:
1. The model predicted all samples as negative (no positive predictions)
2. Precision calculation becomes 0/0 (undefined), which TensorFlow reports as 0
3. Recall is 0 because no true positives were predicted

## Recommendations

### 1. Immediate Actions for Current Model

#### Use Optimal Threshold
```python
# Instead of default threshold
predictions = (probabilities > 0.5)

# Use optimal threshold
predictions = (probabilities > 0.1)
```

#### Deploy with Threshold Adjustment
Use the provided `predict_with_threshold.py` script:
```bash
python scripts/predict_with_threshold.py \
    model.keras \
    /path/to/images \
    --threshold 0.1 \
    --output predictions.csv
```

### 2. Training Improvements

#### Monitor Multiple Thresholds
The training code has been updated to track metrics at multiple thresholds:
- Standard metrics (threshold 0.5)
- Low threshold metrics (thresholds 0.3 and 0.1)

This helps identify threshold issues during training rather than after.

#### Adjust Class Weights
Consider more aggressive class weighting:
```python
# Current: Inverse frequency weighting
class_weights = {0: 0.81, 1: 1.30}

# Recommended: Stronger weighting for minority class
class_weights = {0: 0.5, 1: 2.0}
```

#### Use Focal Loss
For severe class imbalance, consider focal loss:
```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed
```

### 3. Data and Preprocessing

#### Augmentation for Minority Class
Apply stronger augmentation to snow images:
```python
if has_snow:
    # Additional augmentation for snow images
    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_contrast(image, 0.7, 1.3)
```

#### Balanced Batch Sampling
Ensure each batch has balanced representation:
```python
# Use tf.data.Dataset.sample_from_datasets
snow_dataset = full_dataset.filter(lambda x, y: y == 1)
no_snow_dataset = full_dataset.filter(lambda x, y: y == 0)

balanced_dataset = tf.data.Dataset.sample_from_datasets(
    [snow_dataset, no_snow_dataset],
    weights=[0.5, 0.5]
)
```

### 4. Model Architecture Adjustments

#### Output Activation Calibration
Add temperature scaling to the output:
```python
# Add temperature parameter
temperature = 2.0  # Tune this value
outputs = layers.Dense(1, activation=None)(x)
outputs = tf.nn.sigmoid(outputs / temperature)
```

#### Ensemble Methods
Train multiple models with different random seeds and ensemble predictions:
```python
# Average predictions from multiple models
predictions = np.mean([model.predict(x) for model in models], axis=0)
```

### 5. Evaluation Best Practices

#### Always Evaluate Multiple Thresholds
```python
thresholds = np.arange(0.05, 0.95, 0.05)
for threshold in thresholds:
    metrics = calculate_metrics(y_true, y_pred > threshold)
    # Track best threshold
```

#### Monitor Prediction Distribution
Add callbacks to track prediction distribution during training:
```python
class PredictionDistributionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_predictions = self.model.predict(val_data)
        positive_rate = np.mean(val_predictions > 0.5)
        print(f"Epoch {epoch}: {positive_rate:.1%} positive predictions")
```

## Deployment Guidelines

### 1. Production Pipeline
```python
# Recommended prediction pipeline
def predict_snow(image_path, model, threshold=0.1):
    """Predict snow presence with calibrated threshold."""
    image = preprocess_image(image_path)
    probability = model.predict(image)[0][0]
    prediction = probability > threshold
    
    return {
        'prediction': 'snow' if prediction else 'no_snow',
        'probability': float(probability),
        'confidence': abs(probability - threshold)
    }
```

### 2. Monitoring in Production
- Track prediction distribution over time
- Monitor drift in probability distributions
- Collect feedback for model retraining

### 3. Threshold Tuning by Use Case
Different thresholds for different requirements:
- **High Recall** (catch all snow): Use threshold 0.05-0.10
- **Balanced**: Use threshold 0.10-0.15  
- **High Precision** (minimize false positives): Use threshold 0.20-0.30

## Conclusion

The MobileNetV2 model has successfully learned to distinguish between snow and no-snow conditions (AUC > 0.90), but requires threshold calibration for practical deployment. With the optimal threshold of 0.1, the model achieves good performance (F1 ~0.84) suitable for operational use.

Future training should incorporate the recommended improvements to produce models that are better calibrated and require less post-hoc threshold adjustment.

## Appendix: Code Updates Made

1. **Training Metrics** (`models/architectures.py`): Added multi-threshold metrics
2. **Evaluation Script** (`scripts/evaluate_model.py`): Comprehensive evaluation with threshold optimization
3. **Prediction Script** (`scripts/predict_with_threshold.py`): Production-ready prediction with configurable threshold
4. **Validation Metrics** (`training/validation_metrics.py`): Custom callbacks for proper metric calculation

All code updates are backward compatible and improve model monitoring and deployment capabilities.