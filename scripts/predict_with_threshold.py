#!/usr/bin/env python3
"""
Make predictions using a trained model with optimal threshold.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse


def predict_with_threshold(model_path, image_paths, threshold=0.1, batch_size=32):
    """Make predictions using specified threshold.
    
    Args:
        model_path: Path to saved model
        image_paths: List of image paths or directory
        threshold: Classification threshold (default: 0.1)
        batch_size: Batch size for prediction
        
    Returns:
        DataFrame with predictions
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Get list of images
    if isinstance(image_paths, str) and Path(image_paths).is_dir():
        # Directory provided
        image_dir = Path(image_paths)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        image_files = [str(f) for f in image_files]
    elif isinstance(image_paths, str):
        # Single file
        image_files = [image_paths]
    else:
        # List of files
        image_files = image_paths
    
    print(f"Processing {len(image_files)} images with threshold {threshold}")
    
    # Make predictions
    results = []
    
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                batch_images.append(img_array)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if batch_images:
            batch_images = np.array(batch_images)
            batch_probs = model.predict(batch_images, verbose=0)
            
            for path, prob in zip(valid_paths, batch_probs.flatten()):
                prediction = int(prob > threshold)
                results.append({
                    'image_path': path,
                    'filename': Path(path).name,
                    'snow_probability': float(prob),
                    'snow_prediction': prediction,
                    'snow_label': 'snow' if prediction else 'no_snow',
                    'threshold_used': threshold
                })
        
        print(f"Processed {min(i+batch_size, len(image_files))}/{len(image_files)} images", end='\r')
    
    print("\nPrediction complete!")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if len(df) > 0:
        print(f"\nPrediction Summary:")
        print(f"Total images: {len(df)}")
        print(f"Predicted snow: {df['snow_prediction'].sum()} ({df['snow_prediction'].mean()*100:.1f}%)")
        print(f"Predicted no snow: {(~df['snow_prediction']).sum()} ({(~df['snow_prediction']).mean()*100:.1f}%)")
        print(f"Average snow probability: {df['snow_probability'].mean():.3f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Make predictions with optimal threshold')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('images', help='Image file, directory, or CSV with file paths')
    parser.add_argument('--threshold', type=float, default=0.1, 
                        help='Classification threshold (default: 0.1)')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Handle CSV input
    if args.images.endswith('.csv'):
        df_input = pd.read_csv(args.images)
        if 'file_path' in df_input.columns:
            image_paths = df_input['file_path'].tolist()
        elif 'image_path' in df_input.columns:
            image_paths = df_input['image_path'].tolist()
        else:
            raise ValueError("CSV must contain 'file_path' or 'image_path' column")
    else:
        image_paths = args.images
    
    # Make predictions
    df_results = predict_with_threshold(
        args.model_path,
        image_paths,
        threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    # Save results
    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        # Default output name
        output_path = f"predictions_threshold_{args.threshold:.2f}.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()