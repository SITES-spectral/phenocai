"""Neural network architectures for PhenoCAI."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, List


def create_mobilenet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.2,
    freeze_base: bool = True,
    fine_tune_from: Optional[int] = None
) -> keras.Model:
    """Create transfer learning model based on MobileNetV2.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        freeze_base: Whether to freeze base model weights initially
        fine_tune_from: Layer index from which to unfreeze (for fine-tuning)
        
    Returns:
        Compiled Keras model
    """
    # Create base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model layers if requested
    base_model.trainable = not freeze_base
    
    # If fine-tuning, unfreeze from specific layer
    if fine_tune_from is not None and fine_tune_from < len(base_model.layers):
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # Preprocessing for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom top layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='snow_presence')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    filters: List[int] = [32, 64, 128, 256],
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True
) -> keras.Model:
    """Create custom CNN architecture optimized for phenocam images.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        filters: Number of filters for each conv block
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomBrightness(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # Normalize pixel values
    x = layers.Rescaling(1./255)(x)
    
    # Conv blocks
    for i, num_filters in enumerate(filters):
        # First conv in block
        x = layers.Conv2D(
            num_filters, 
            3, 
            padding='same',
            activation='relu',
            name=f'conv{i+1}_1'
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # Second conv in block
        x = layers.Conv2D(
            num_filters, 
            3, 
            padding='same',
            activation='relu',
            name=f'conv{i+1}_2'
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # Pooling and dropout
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)  # Less dropout in conv layers
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='snow_presence')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_ensemble_model(
    models: List[keras.Model],
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    ensemble_method: str = 'average'
) -> keras.Model:
    """Create ensemble model combining multiple base models.
    
    Args:
        models: List of trained base models
        input_shape: Input image shape
        num_classes: Number of output classes
        ensemble_method: How to combine predictions ('average', 'weighted', 'stacking')
        
    Returns:
        Ensemble Keras model
    """
    # Make base models non-trainable
    for model in models:
        model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    
    # Get predictions from all models
    predictions = [model(inputs) for model in models]
    
    if ensemble_method == 'average':
        # Simple averaging
        if len(predictions) > 1:
            outputs = layers.Average()(predictions)
        else:
            outputs = predictions[0]
    
    elif ensemble_method == 'weighted':
        # Weighted average (learnable weights)
        if len(predictions) > 1:
            # Create learnable weights
            weights = []
            for i in range(len(predictions)):
                w = layers.Dense(
                    1, 
                    activation='sigmoid',
                    use_bias=False,
                    name=f'weight_{i}'
                )(layers.GlobalAveragePooling1D()(layers.Reshape((1, -1))(predictions[i])))
                weights.append(w)
            
            # Normalize weights
            weight_sum = layers.Add()(weights)
            normalized_weights = [layers.Lambda(lambda x: x[0] / x[1])([w, weight_sum]) 
                                for w in weights]
            
            # Weighted sum
            weighted_preds = [layers.Multiply()([pred, w]) 
                            for pred, w in zip(predictions, normalized_weights)]
            outputs = layers.Add()(weighted_preds)
        else:
            outputs = predictions[0]
    
    elif ensemble_method == 'stacking':
        # Meta-learner on top of base predictions
        if len(predictions) > 1:
            concatenated = layers.Concatenate()(predictions)
        else:
            concatenated = predictions[0]
        
        # Meta-learner
        x = layers.Dense(32, activation='relu')(concatenated)
        x = layers.Dropout(0.2)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    model = keras.Model(inputs, outputs)
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    num_classes: int = 2,
    loss: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> keras.Model:
    """Compile model with appropriate loss and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        num_classes: Number of classes
        loss: Loss function (auto-selected if None)
        metrics: List of metrics (auto-selected if None)
        
    Returns:
        Compiled model
    """
    # Auto-select loss if not provided
    if loss is None:
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
    
    # Auto-select metrics if not provided
    if metrics is None:
        if num_classes == 2:
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        else:
            metrics = [
                'accuracy',
                'sparse_categorical_accuracy'
            ]
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model: keras.Model) -> dict:
    """Get model architecture summary.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'num_layers': len(model.layers),
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }