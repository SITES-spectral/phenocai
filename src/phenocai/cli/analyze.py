"""Analysis commands for PhenoCAI CLI."""

import click
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import json

from ..analysis.heuristics import SnowDetector, QualityAssessment
from ..config import config


@click.group()
def analyze():
    """Analyze images using heuristic methods."""
    pass


@analyze.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--roi-mask', type=click.Path(exists=True), help='Path to ROI mask image')
@click.option('--brightness-threshold', default=180, help='Brightness threshold for snow (0-255)')
@click.option('--saturation-threshold', default=30, help='Saturation threshold for snow (0-255)')
@click.option('--min-snow-percentage', default=0.1, help='Minimum snow coverage percentage')
@click.option('--visualize', is_flag=True, help='Save visualization image')
@click.option('--output', type=click.Path(), help='Output path for visualization')
def detect_snow(
    image_path: str,
    roi_mask: Optional[str],
    brightness_threshold: int,
    saturation_threshold: int,
    min_snow_percentage: float,
    visualize: bool,
    output: Optional[str]
):
    """Detect snow in image using HSV heuristics."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        click.echo(f"Error: Could not load image from {image_path}", err=True)
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load ROI mask if provided
    mask = None
    if roi_mask:
        mask = cv2.imread(roi_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            click.echo(f"Warning: Could not load ROI mask from {roi_mask}", err=True)
    
    # Create detector
    detector = SnowDetector(
        brightness_threshold=brightness_threshold,
        saturation_threshold=saturation_threshold,
        min_snow_percentage=min_snow_percentage
    )
    
    # Detect snow
    if visualize:
        result, viz_image = detector.detect_with_visualization(image, mask)
        
        # Save visualization
        if output:
            output_path = output
        else:
            output_path = str(Path(image_path).with_suffix('.snow_viz.jpg'))
        
        # Convert back to BGR for OpenCV
        viz_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, viz_bgr)
        click.echo(f"Visualization saved to: {output_path}")
    else:
        result = detector.detect(image, mask)
    
    # Display results
    click.echo(f"\nSnow Detection Results for {Path(image_path).name}:")
    click.echo(f"  Has Snow: {'Yes' if result.has_snow else 'No'}")
    click.echo(f"  Confidence: {result.confidence:.3f}")
    click.echo(f"  Snow Coverage: {result.snow_percentage:.1%}")
    click.echo(f"  Bright Pixels: {result.bright_pixels:,} / {result.total_pixels:,}")
    click.echo(f"  Mean Brightness (V): {result.mean_brightness:.1f}")
    click.echo(f"  Mean Saturation (S): {result.mean_saturation:.1f}")


@analyze.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--roi-mask', type=click.Path(exists=True), help='Path to ROI mask image')
@click.option('--json-output', is_flag=True, help='Output results as JSON')
def assess_quality(image_path: str, roi_mask: Optional[str], json_output: bool):
    """Assess image quality issues."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        click.echo(f"Error: Could not load image from {image_path}", err=True)
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load ROI mask if provided
    mask = None
    if roi_mask:
        mask = cv2.imread(roi_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            click.echo(f"Warning: Could not load ROI mask from {roi_mask}", err=True)
    
    # Assess quality
    assessor = QualityAssessment()
    issues = assessor.assess(image, mask)
    summary = assessor.summarize_issues(issues)
    
    if json_output:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\nQuality Assessment for {Path(image_path).name}:")
        
        if not summary["has_issues"]:
            click.echo("  ✓ No quality issues detected")
        else:
            click.echo(f"  Issues Found: {summary['issue_count']}")
            click.echo(f"  Average Severity: {summary['severity_score']:.2f}")
            click.echo(f"  Max Severity: {summary['max_severity']:.2f}")
            
            click.echo("\n  Issue Breakdown:")
            for issue_type, count in summary['summary'].items():
                if count > 0:
                    click.echo(f"    - {issue_type}: {count}")
            
            click.echo("\n  Details:")
            for issue in summary['issues']:
                severity_indicator = "⚠️" if issue['severity'] < 0.5 else "❌"
                click.echo(f"    {severity_indicator} {issue['type']} (severity: {issue['severity']:.2f})")
                click.echo(f"       {issue['description']}")


@analyze.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--sample-size', default=100, help='Number of images to sample')
@click.option('--seed', default=42, help='Random seed for sampling')
def analyze_dataset(dataset_path: str, sample_size: int, seed: int):
    """Analyze a sample of images from dataset for snow and quality."""
    import pandas as pd
    from collections import Counter
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Sample images
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=seed)
        click.echo(f"Sampling {sample_size} images from {len(df)} total")
    else:
        df_sample = df
        click.echo(f"Analyzing all {len(df)} images")
    
    # Initialize analyzers
    snow_detector = SnowDetector()
    quality_assessor = QualityAssessment()
    
    # Results storage
    snow_results = []
    quality_issues = []
    
    with click.progressbar(df_sample.iterrows(), length=len(df_sample), 
                          label='Analyzing images') as bar:
        for idx, row in bar:
            # Load image
            image = cv2.imread(row['file_path'])
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Snow detection
            snow_result = snow_detector.detect(image)
            snow_results.append({
                'has_snow_heuristic': snow_result.has_snow,
                'snow_confidence': snow_result.confidence,
                'snow_percentage': snow_result.snow_percentage,
                'has_snow_label': row.get('snow_presence', None)
            })
            
            # Quality assessment
            issues = quality_assessor.assess(image)
            quality_issues.extend([issue.issue_type for issue in issues])
    
    # Analyze results
    click.echo("\n=== Snow Detection Analysis ===")
    
    # Compare with labels if available
    if 'has_snow_label' in snow_results[0] and snow_results[0]['has_snow_label'] is not None:
        correct = sum(1 for r in snow_results 
                     if r['has_snow_heuristic'] == r['has_snow_label'])
        accuracy = correct / len(snow_results)
        click.echo(f"Accuracy vs Labels: {accuracy:.1%}")
        
        # Confusion matrix
        tp = sum(1 for r in snow_results 
                if r['has_snow_heuristic'] and r['has_snow_label'])
        fp = sum(1 for r in snow_results 
                if r['has_snow_heuristic'] and not r['has_snow_label'])
        fn = sum(1 for r in snow_results 
                if not r['has_snow_heuristic'] and r['has_snow_label'])
        tn = sum(1 for r in snow_results 
                if not r['has_snow_heuristic'] and not r['has_snow_label'])
        
        click.echo(f"\nConfusion Matrix:")
        click.echo(f"              Predicted")
        click.echo(f"           Snow  No Snow")
        click.echo(f"Actual Snow   {tp:4d}   {fn:4d}")
        click.echo(f"    No Snow   {fp:4d}   {tn:4d}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            click.echo(f"\nPrecision: {precision:.1%}")
        if tp + fn > 0:
            recall = tp / (tp + fn)
            click.echo(f"Recall: {recall:.1%}")
    
    # Snow statistics
    snow_detected = sum(1 for r in snow_results if r['has_snow_heuristic'])
    click.echo(f"\nSnow Detected: {snow_detected}/{len(snow_results)} ({snow_detected/len(snow_results):.1%})")
    
    avg_confidence = sum(r['snow_confidence'] for r in snow_results) / len(snow_results)
    click.echo(f"Average Confidence: {avg_confidence:.3f}")
    
    avg_coverage = sum(r['snow_percentage'] for r in snow_results) / len(snow_results)
    click.echo(f"Average Snow Coverage: {avg_coverage:.1%}")
    
    # Quality issues
    click.echo("\n=== Quality Issues Analysis ===")
    issue_counts = Counter(quality_issues)
    
    if issue_counts:
        total_issues = sum(issue_counts.values())
        images_with_issues = sum(1 for r in snow_results if r)  # Placeholder
        
        click.echo(f"Total Issues Found: {total_issues}")
        click.echo("\nIssue Distribution:")
        for issue_type, count in issue_counts.most_common():
            percentage = count / len(df_sample) * 100
            click.echo(f"  {issue_type}: {count} ({percentage:.1f}% of images)")
    else:
        click.echo("No quality issues detected in sample")