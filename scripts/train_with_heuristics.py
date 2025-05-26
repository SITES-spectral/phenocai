#!/usr/bin/env python
"""
Train a model that combines CNN features with heuristic features.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from phenocai.heuristics.snow_detection_improved import detect_snow_adaptive
from phenocai.heuristics.image_quality import calculate_image_statistics
from phenocai.utils import load_image
from phenocai.models.config import ModelConfig, TrainingConfig


def create_hybrid_model(input_shape=(224, 224, 3), num_heuristic_features=10):
    """Create a model that combines CNN and heuristic features."""
    
    # Image input branch (CNN)
    image_input = layers.Input(shape=input_shape, name='image_input')
    
    # Use MobileNetV2 as base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze initially
    
    # CNN feature extraction
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    cnn_features = layers.Dense(256, activation='relu', name='cnn_features')(x)
    
    # Heuristic features input branch
    heuristic_input = layers.Input(shape=(num_heuristic_features,), name='heuristic_input')
    heuristic_features = layers.Dense(64, activation='relu', name='heuristic_features')(heuristic_input)
    
    # Combine both branches
    combined = layers.Concatenate()([cnn_features, heuristic_features])
    
    # Final classification layers
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    output = layers.Dense(1, activation='sigmoid', name='snow_prediction')(x)
    
    # Create model
    model = keras.Model(
        inputs=[image_input, heuristic_input],
        outputs=output,
        name='hybrid_snow_detector'
    )
    
    return model, base_model


def extract_heuristic_features(image_path, context=None):
    """Extract heuristic features from an image."""
    image = load_image(image_path)
    if image is None:
        return None
    
    # Get adaptive heuristic results
    _, confidence, metadata = detect_snow_adaptive(image, context)
    
    # Get image statistics
    stats = calculate_image_statistics(image)
    
    # Combine features
    features = [
        metadata['snow_percentage'],
        metadata['brightness'] / 255.0,  # Normalize
        metadata['texture_uniformity'] / 100.0,
        metadata['edge_density'],
        metadata['mean_saturation'] / 255.0,
        metadata['color_variance'] / 100.0,
        metadata['high_value_percentage'],
        metadata['validation_score'],
        stats['blur_metric'] / 1000.0,  # Normalize
        confidence
    ]
    
    return np.array(features, dtype=np.float32)


class HybridDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for hybrid model."""
    
    def __init__(self, df, batch_size=32, image_size=(224, 224), 
                 shuffle=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.df))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_images = np.zeros((self.batch_size, *self.image_size, 3), dtype=np.float32)
        batch_heuristics = np.zeros((self.batch_size, 10), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size, 1), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            row = self.df.iloc[idx]
            
            # Load and preprocess image
            image = load_image(row['file_path'])
            if image is not None:
                image = tf.image.resize(image, self.image_size)
                image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
                
                if self.augment:
                    # Simple augmentation
                    if np.random.random() > 0.5:
                        image = tf.image.flip_left_right(image)
                    image = tf.image.random_brightness(image, 0.1)
                
                batch_images[i] = image
            
            # Extract heuristic features
            features = extract_heuristic_features(row['file_path'])
            if features is not None:
                batch_heuristics[i] = features
            
            # Label
            batch_labels[i] = float(row['snow_presence'])
        
        return [batch_images, batch_heuristics], batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_hybrid_model(dataset_csv, output_dir, epochs=20):
    """Train the hybrid model."""
    print("=== Training Hybrid Model with Heuristic Features ===\n")
    
    # Load dataset
    df = pd.read_csv(dataset_csv)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create data generators
    train_gen = HybridDataGenerator(train_df, batch_size=32, augment=True)
    val_gen = HybridDataGenerator(val_df, batch_size=32, augment=False)
    test_gen = HybridDataGenerator(test_df, batch_size=32, augment=False)
    
    # Create model
    model, base_model = create_hybrid_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Setup callbacks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_path / 'best_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        ),
        tf.keras.callbacks.CSVLogger(
            output_path / 'training_log.csv'
        )
    ]
    
    # Train model - Phase 1: Feature extraction layers only
    print("\nPhase 1: Training top layers...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune base model
    print("\nPhase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        initial_epoch=epochs // 2,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_gen)
    
    # Save results
    results = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'test_auc': float(test_results[4]),
        'test_f1': float(2 * test_results[2] * test_results[3] / (test_results[2] + test_results[3]))
        if (test_results[2] + test_results[3]) > 0 else 0.0
    }
    
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final model and results
    model.save(output_path / 'hybrid_model_final.keras')
    
    with open(output_path / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Combine histories
    history_combined = {
        k: history1.history.get(k, []) + history2.history.get(k, [])
        for k in history1.history.keys()
    }
    
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history_combined, f, indent=2)
    
    print(f"\nModel and results saved to: {output_path}")
    
    return model, results


def create_training_dataset():
    """Create or find a dataset for training."""
    # Try to find existing balanced dataset
    possible_paths = [
        "/home/jobelund/lu2024-12-46/SITES/Spectral/analysis/phenocams/transfer_learning/lonnstorp/experimental_multistation_combined_dataset.csv"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    print("No existing dataset found. Please create one using:")
    print("phenocai dataset create --stations lonnstorp robacksdalen")
    return None


def main():
    """Main training workflow."""
    # Find or create dataset
    dataset_path = create_training_dataset()
    
    if dataset_path is None:
        print("No dataset available for training")
        return
    
    # Output directory
    output_dir = '/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/models/hybrid_model'
    
    # Train model
    model, results = train_hybrid_model(
        dataset_csv=dataset_path,
        output_dir=output_dir,
        epochs=20
    )
    
    print("\n=== Training Complete ===")
    print(f"Final test F1 score: {results['test_f1']:.3f}")
    
    # Compare with other approaches
    print("\nComparison with other methods:")
    print("- Current heuristics: F1 ~0.48")
    print("- Improved heuristics: F1 ~0.70")
    print("- ML model (balanced): F1 ~0.45")
    print(f"- Hybrid model: F1 {results['test_f1']:.3f}")


if __name__ == "__main__":
    main()