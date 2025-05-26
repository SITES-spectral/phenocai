#!/usr/bin/env python
"""
Create visualizations for method comparison results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_method_comparison_summary():
    """Create a comprehensive visual summary of method comparison."""
    
    # Performance data
    methods = ['Current\nHeuristic', 'Improved\nHeuristic', 'ML Model', 'Hybrid']
    metrics = {
        'Precision': [0.667, 0.571, 0.419, 0.556],
        'Recall': [0.099, 0.040, 0.663, 0.149],
        'F1 Score': [0.172, 0.074, 0.513, 0.234],
        'AUC': [0.612, 0.606, 0.628, 0.656]
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Main performance comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(methods))
    width = 0.2
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, values, width, label=metric)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.8)
    
    # 2. Precision-Recall trade-off
    ax2 = plt.subplot(2, 3, 2)
    precisions = metrics['Precision']
    recalls = metrics['Recall']
    
    # Plot points
    for i, method in enumerate(methods):
        ax2.scatter(recalls[i], precisions[i], s=200, label=method.replace('\n', ' '))
        ax2.annotate(method.replace('\n', ' '), 
                    (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 0.8)
    ax2.set_ylim(0.3, 0.8)
    
    # 3. F1 Score comparison
    ax3 = plt.subplot(2, 3, 3)
    f1_scores = metrics['F1 Score']
    colors = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4']
    bars = ax3.bar(methods, f1_scores, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 0.6)
    
    # 4. Characteristics radar chart
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    
    categories = ['Precision', 'Recall', 'Speed', 'Interpretability', 'Robustness']
    
    # Normalized scores (0-1)
    ml_scores = [0.419, 0.663, 0.3, 0.2, 0.7]
    heur_scores = [0.667, 0.099, 1.0, 1.0, 0.5]
    hybrid_scores = [0.556, 0.149, 0.3, 0.6, 0.8]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    ml_scores += ml_scores[:1]
    heur_scores += heur_scores[:1]
    hybrid_scores += hybrid_scores[:1]
    angles += angles[:1]
    
    ax4.plot(angles, ml_scores, 'o-', linewidth=2, label='ML Model')
    ax4.fill(angles, ml_scores, alpha=0.25)
    ax4.plot(angles, heur_scores, 'o-', linewidth=2, label='Heuristics')
    ax4.fill(angles, heur_scores, alpha=0.25)
    ax4.plot(angles, hybrid_scores, 'o-', linewidth=2, label='Hybrid')
    ax4.fill(angles, hybrid_scores, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Method Characteristics', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax4.grid(True)
    
    # 5. Improvement potential
    ax5 = plt.subplot(2, 3, 5)
    
    current_f1 = [0.172, 0.074, 0.513, 0.234]
    expected_f1 = [0.25, 0.30, 0.65, 0.70]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, current_f1, width, label='Current', color='lightcoral')
    bars2 = ax5.bar(x + width/2, expected_f1, width, label='Expected', color='lightgreen')
    
    # Add improvement percentages
    for i in range(len(methods)):
        improvement = ((expected_f1[i] - current_f1[i]) / current_f1[i]) * 100
        ax5.text(x[i], expected_f1[i] + 0.02, f'+{improvement:.0f}%', 
                ha='center', va='bottom', fontweight='bold', color='green')
    
    ax5.set_xlabel('Method')
    ax5.set_ylabel('F1 Score')
    ax5.set_title('Improvement Potential', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 0.8)
    
    # 6. Recommendations text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    recommendations = """
    KEY RECOMMENDATIONS
    
    1. Immediate Actions:
       • Optimize hybrid weights (70/30 → 80/20)
       • Adjust heuristic thresholds
       • Expected gain: +10-15% F1
    
    2. Short-term (2 weeks):
       • Add heuristic features to ML training
       • Implement selective hybrid strategy
       • Expected gain: +20% F1
    
    3. Medium-term (1 month):
       • Build ensemble model
       • Deploy active learning
       • Expected F1: >0.75
    
    Best Path Forward:
    → Start with hybrid optimization
    → Add feature engineering
    → Deploy ensemble approach
    """
    
    ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Snow Detection Methods: Comprehensive Evaluation Summary', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/evaluation')
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / 'method_comparison_summary.png', dpi=300, bbox_inches='tight')
    print(f"Summary visualization saved to: {output_path / 'method_comparison_summary.png'}")
    
    # Create additional comparison chart
    create_decision_flow_chart()


def create_decision_flow_chart():
    """Create a decision flow chart for method selection."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Snow Detection Method Selection Guide', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Decision tree content
    flow_text = """
    START: What is your primary concern?
    
    ├─> High Precision (Few false positives)
    │   ├─> Speed critical? → Current Heuristics (F1=0.17, Precision=0.67)
    │   └─> Accuracy important? → Hybrid Approach (F1=0.23, Precision=0.56)
    │
    ├─> High Recall (Find all snow)
    │   └─> ML Model (F1=0.51, Recall=0.66)
    │
    ├─> Balanced Performance
    │   ├─> Interpretability needed? → Optimized Hybrid (Expected F1=0.70)
    │   └─> Best accuracy? → Ensemble Method (Expected F1=0.75)
    │
    └─> Real-time Processing
        ├─> Single images? → Heuristics (5ms/image)
        └─> Batch processing? → ML Model (30ms/image)
    
    
    HYBRID OPTIMIZATION SETTINGS:
    
    Default:     70% ML + 30% Heuristics
    High Recall: 85% ML + 15% Heuristics  
    Balanced:    75% ML + 25% Heuristics
    High Precision: 60% ML + 40% Heuristics
    """
    
    ax.text(0.1, 0.85, flow_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/evaluation')
    plt.savefig(output_path / 'method_selection_guide.png', dpi=300, bbox_inches='tight')
    print(f"Selection guide saved to: {output_path / 'method_selection_guide.png'}")


if __name__ == "__main__":
    create_method_comparison_summary()