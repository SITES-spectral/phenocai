#!/usr/bin/env python
"""
Create educational figures for geography students.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

def create_all_educational_figures():
    """Create all figures for the geography student document."""
    output_dir = Path('/lunarc/nobackup/projects/sitesspec/SITES/Spectral/apps/phenocai/docs/figures/educational')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figures
    create_phenocam_concept(output_dir)
    create_snow_detection_challenges(output_dir)
    create_data_imbalance_visualization(output_dir)
    create_method_comparison_simple(output_dir)
    create_seasonal_performance(output_dir)
    create_ecosystem_applications(output_dir)
    create_workflow_diagram(output_dir)
    
    print(f"Educational figures saved to: {output_dir}")


def create_phenocam_concept(output_dir):
    """Create phenocam monitoring concept illustration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Traditional vs Automated Monitoring
    ax1.text(0.5, 0.95, 'Snow Monitoring Methods', ha='center', fontsize=16, fontweight='bold', transform=ax1.transAxes)
    
    # Traditional methods
    traditional_y = 0.75
    ax1.text(0.25, traditional_y, 'Traditional Methods', ha='center', fontsize=14, fontweight='bold', transform=ax1.transAxes)
    methods = [
        '• Manual observations\n  (labor intensive)',
        '• Weather stations\n  (point measurements)',
        '• Satellite imagery\n  (low time resolution)'
    ]
    for i, method in enumerate(methods):
        ax1.text(0.05, traditional_y - 0.15*(i+1), method, fontsize=11, transform=ax1.transAxes)
    
    # Automated methods
    auto_y = 0.75
    ax1.text(0.75, auto_y, 'Phenocam Monitoring', ha='center', fontsize=14, fontweight='bold', 
             color='darkgreen', transform=ax1.transAxes)
    benefits = [
        '• Continuous monitoring\n  (every 30 minutes)',
        '• Spatial coverage\n  (whole landscape view)',
        '• Cost effective\n  (one-time setup)'
    ]
    for i, benefit in enumerate(benefits):
        ax1.text(0.55, auto_y - 0.15*(i+1), benefit, fontsize=11, color='darkgreen', transform=ax1.transAxes)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Phenocam network illustration
    ax2.text(0.5, 0.95, 'Swedish Phenocam Network', ha='center', fontsize=16, fontweight='bold', transform=ax2.transAxes)
    
    # Map-like visualization
    # Sweden outline (simplified)
    sweden_x = [0.5, 0.45, 0.4, 0.35, 0.4, 0.5, 0.55, 0.6, 0.55, 0.5]
    sweden_y = [0.1, 0.2, 0.4, 0.6, 0.8, 0.85, 0.8, 0.6, 0.3, 0.1]
    ax2.plot(sweden_x, sweden_y, 'k-', linewidth=2)
    ax2.fill(sweden_x, sweden_y, color='lightgray', alpha=0.3)
    
    # Station locations
    stations = [
        {'name': 'Röbäcksdalen', 'pos': (0.5, 0.7), 'color': 'blue'},
        {'name': 'Lönnstorp', 'pos': (0.45, 0.25), 'color': 'red'},
        {'name': 'Other stations', 'pos': (0.48, 0.5), 'color': 'gray'}
    ]
    
    for station in stations:
        ax2.plot(station['pos'][0], station['pos'][1], 'o', markersize=10, 
                color=station['color'], markeredgecolor='black')
        ax2.text(station['pos'][0] + 0.02, station['pos'][1], station['name'], 
                fontsize=10, va='center')
    
    # Add climate zones
    ax2.text(0.3, 0.75, 'Subarctic', fontsize=10, style='italic', color='blue')
    ax2.text(0.3, 0.25, 'Temperate', fontsize=10, style='italic', color='red')
    
    ax2.set_xlim(0.2, 0.8)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('Automated Environmental Monitoring with Phenocams', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'phenocam_concept.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_snow_detection_challenges(output_dir):
    """Create illustration of snow detection challenges."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Snow Detection Challenges in Phenocam Images', fontsize=16, fontweight='bold')
    
    challenges = [
        {'title': 'Clear Snow', 'difficulty': 'Easy', 'color': 'green'},
        {'title': 'Partial Snow', 'difficulty': 'Medium', 'color': 'orange'},
        {'title': 'Frost vs Snow', 'difficulty': 'Hard', 'color': 'red'},
        {'title': 'Fog/Clouds', 'difficulty': 'Hard', 'color': 'red'},
        {'title': 'Shadows', 'difficulty': 'Medium', 'color': 'orange'},
        {'title': 'Camera Issues', 'difficulty': 'Hard', 'color': 'red'}
    ]
    
    for idx, (ax, challenge) in enumerate(zip(axes.flat, challenges)):
        # Create simple illustration
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightblue', alpha=0.3))
        
        if idx == 0:  # Clear snow
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.4, color='white', alpha=0.9))
        elif idx == 1:  # Partial snow
            for i in range(3):
                ax.add_patch(plt.Rectangle((i*0.3, 0), 0.2, 0.4, color='white', alpha=0.9))
        elif idx == 2:  # Frost
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.1, color='lightgray', alpha=0.7))
        elif idx == 3:  # Fog
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.5))
        elif idx == 4:  # Shadows
            ax.add_patch(plt.Rectangle((0, 0), 0.5, 0.4, color='white', alpha=0.9))
            ax.add_patch(plt.Rectangle((0.5, 0), 0.5, 0.4, color='darkgray', alpha=0.7))
        elif idx == 5:  # Camera issues
            ax.add_patch(plt.Circle((0.5, 0.5), 0.2, color='gray', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(challenge['title'], fontsize=12, fontweight='bold')
        ax.text(0.5, 0.9, f"Difficulty: {challenge['difficulty']}", 
                ha='center', transform=ax.transAxes, 
                color=challenge['color'], fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'snow_detection_challenges.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_data_imbalance_visualization(output_dir):
    """Create data imbalance explanation."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original problem
    ax1.set_title('The Data Imbalance Problem', fontsize=14, fontweight='bold')
    days = [''] * 100
    colors = ['lightblue'] * 90 + ['white'] * 10
    y_pos = np.arange(len(days))
    bars = ax1.barh(y_pos, [1]*100, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Days in a Year (simplified)')
    ax1.set_yticks([])
    ax1.text(0.5, 0.5, '90% No Snow\n10% Snow', transform=ax1.transAxes,
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Lazy prediction
    ax2.set_title('The "Lazy" Predictor Problem', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.7, 'If model always predicts\n"NO SNOW"', 
             ha='center', transform=ax2.transAxes, fontsize=14)
    ax2.text(0.5, 0.5, 'It\'s right 90% of the time!', 
             ha='center', transform=ax2.transAxes, fontsize=16, 
             fontweight='bold', color='red')
    ax2.text(0.5, 0.3, 'But misses ALL actual snow events', 
             ha='center', transform=ax2.transAxes, fontsize=12, style='italic')
    ax2.axis('off')
    
    # Solution
    ax3.set_title('Our Solution: Balanced Training', fontsize=14, fontweight='bold')
    balanced_colors = ['lightblue'] * 50 + ['white'] * 50
    np.random.shuffle(balanced_colors)
    bars = ax3.barh(np.arange(100), [1]*100, color=balanced_colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Training Examples')
    ax3.set_yticks([])
    ax3.text(0.5, 0.5, '50% No Snow\n50% Snow', transform=ax3.transAxes,
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Understanding the Data Imbalance Challenge', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'data_imbalance_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_method_comparison_simple(output_dir):
    """Create simplified method comparison for students."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Method descriptions
    methods = [
        {
            'name': 'Color-Based Rules\n(Heuristics)',
            'pros': ['Fast (5ms)', 'Interpretable', 'No training needed'],
            'cons': ['Rigid rules', 'Misses subtle cases', 'Many false negatives'],
            'accuracy': 'Precision: 67%\nRecall: 10%',
            'analogy': 'Like a field guide:\nWhite + Bright = Snow',
            'color': 'lightblue'
        },
        {
            'name': 'Pattern Recognition\n(Machine Learning)',
            'pros': ['Finds complex patterns', 'Adapts to conditions', 'High recall'],
            'cons': ['Black box', 'Needs training data', 'Some false positives'],
            'accuracy': 'Precision: 42%\nRecall: 66%',
            'analogy': 'Like an experienced\nfield assistant',
            'color': 'lightcoral'
        },
        {
            'name': 'Combined Approach\n(Hybrid)',
            'pros': ['Balanced performance', 'More reliable', 'Flexible weights'],
            'cons': ['More complex', 'Slower (35ms)', 'Needs tuning'],
            'accuracy': 'Precision: 56%\nRecall: 15%',
            'analogy': 'Expert + Assistant\nworking together',
            'color': 'lightgreen'
        }
    ]
    
    # Create comparison boxes
    for i, method in enumerate(methods):
        x_base = i * 0.33 + 0.05
        
        # Main box
        rect = mpatches.FancyBboxPatch((x_base, 0.3), 0.25, 0.6,
                                      boxstyle="round,pad=0.02",
                                      facecolor=method['color'],
                                      edgecolor='black',
                                      linewidth=2)
        ax.add_patch(rect)
        
        # Method name
        ax.text(x_base + 0.125, 0.85, method['name'], 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Analogy
        ax.text(x_base + 0.125, 0.75, method['analogy'], 
                ha='center', va='center', fontsize=10, style='italic')
        
        # Accuracy
        ax.text(x_base + 0.125, 0.65, method['accuracy'], 
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Pros
        ax.text(x_base + 0.125, 0.5, 'Pros:', 
                ha='center', fontsize=10, fontweight='bold')
        for j, pro in enumerate(method['pros']):
            ax.text(x_base + 0.125, 0.45 - j*0.05, f'✓ {pro}', 
                    ha='center', fontsize=8, color='darkgreen')
        
        # Cons
        ax.text(x_base + 0.125, 0.25, 'Cons:', 
                ha='center', fontsize=10, fontweight='bold')
        for j, con in enumerate(method['cons']):
            ax.text(x_base + 0.125, 0.2 - j*0.05, f'✗ {con}', 
                    ha='center', fontsize=8, color='darkred')
    
    # Add performance note
    ax.text(0.5, 0.15, 'Performance Metrics Explained:', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.1, 'Precision = When it says "snow", how often is it right?', ha='center', fontsize=10)
    ax.text(0.5, 0.05, 'Recall = Of all snow events, how many does it find?', ha='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Three Approaches to Snow Detection', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison_simple.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_seasonal_performance(output_dir):
    """Create seasonal performance visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Top: Detection reliability by season
    reliability = [90, 85, 70, 50, 30, 20, 20, 20, 30, 50, 70, 85]
    colors = ['darkblue', 'darkblue', 'blue', 'lightblue', 'yellow', 'orange', 
              'orange', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue']
    
    bars = ax1.bar(months, reliability, color=colors, edgecolor='black')
    ax1.set_ylabel('Detection Reliability (%)', fontsize=12)
    ax1.set_title('How Reliable is Snow Detection Throughout the Year?', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add season labels
    ax1.text(1.5, 95, 'Winter', ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(4.5, 95, 'Spring', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(7.5, 95, 'Summer', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax1.text(10.5, 95, 'Fall', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # Bottom: Why detection varies
    ax2.set_title('Why Does Performance Vary?', fontsize=14, fontweight='bold')
    
    # Create explanation boxes
    explanations = [
        {'season': 'Winter', 'x': 1.5, 'text': 'Clear snow\nHigh contrast\nConsistent'},
        {'season': 'Spring/Fall', 'x': 7.5, 'text': 'Patchy snow\nMelting/forming\nVariable'},
        {'season': 'Summer', 'x': 6.5, 'text': 'No snow expected\nFew false alarms'}
    ]
    
    for exp in explanations:
        ax2.text(exp['x'], 0.5, exp['text'], ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=10)
    
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-0.5, 11.5)
    ax2.axis('off')
    
    plt.suptitle('Seasonal Variation in Snow Detection Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_ecosystem_applications(output_dir):
    """Create ecosystem research applications diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Central node
    center = (0.5, 0.5)
    ax.add_patch(plt.Circle(center, 0.15, color='lightblue', edgecolor='black', linewidth=2))
    ax.text(center[0], center[1], 'Automated\nSnow\nDetection', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # Application nodes
    applications = [
        {
            'name': 'Phenology\nStudies',
            'pos': (0.5, 0.85),
            'examples': ['Green-up timing', 'Growing season length', 'Flowering dates'],
            'color': 'lightgreen'
        },
        {
            'name': 'Wildlife\nMonitoring',
            'pos': (0.15, 0.65),
            'examples': ['Habitat availability', 'Migration timing', 'Foraging behavior'],
            'color': 'lightyellow'
        },
        {
            'name': 'Climate\nChange',
            'pos': (0.15, 0.35),
            'examples': ['Snow season trends', 'Extreme events', 'Model validation'],
            'color': 'lightcoral'
        },
        {
            'name': 'Hydrology',
            'pos': (0.5, 0.15),
            'examples': ['Snowmelt timing', 'Water availability', 'Flood prediction'],
            'color': 'lightblue'
        },
        {
            'name': 'Carbon\nCycling',
            'pos': (0.85, 0.35),
            'examples': ['Soil respiration', 'Growing season CO₂', 'Winter processes'],
            'color': 'lightgray'
        },
        {
            'name': 'Agriculture',
            'pos': (0.85, 0.65),
            'examples': ['Frost risk', 'Crop planning', 'Soil preparation'],
            'color': 'wheat'
        }
    ]
    
    for app in applications:
        # Draw connection
        ax.plot([center[0], app['pos'][0]], [center[1], app['pos'][1]], 
                'k-', linewidth=1, alpha=0.5)
        
        # Draw application node
        ax.add_patch(plt.Circle(app['pos'], 0.08, color=app['color'], 
                               edgecolor='black', linewidth=2))
        ax.text(app['pos'][0], app['pos'][1], app['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Add examples
        for i, example in enumerate(app['examples']):
            offset = 0.12 + i * 0.03
            ax.text(app['pos'][0], app['pos'][1] - offset, f'• {example}', 
                    ha='center', va='top', fontsize=8, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Ecosystem Research Applications of Automated Snow Detection', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ecosystem_applications.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_workflow_diagram(output_dir):
    """Create practical workflow for students."""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Workflow steps
    steps = [
        {
            'title': '1. Data Collection',
            'tasks': ['Download phenocam images', 'Check image quality', 'Note missing dates'],
            'y': 0.9
        },
        {
            'title': '2. Run Detection',
            'tasks': ['Apply automated detection', 'Check confidence scores', 'Flag uncertain cases'],
            'y': 0.75
        },
        {
            'title': '3. Validation',
            'tasks': ['Sample 10% of results', 'Manual verification', 'Check against weather data'],
            'y': 0.6
        },
        {
            'title': '4. Analysis',
            'tasks': ['Calculate snow metrics', 'Identify patterns', 'Link to ecosystem data'],
            'y': 0.45
        },
        {
            'title': '5. Interpretation',
            'tasks': ['Consider limitations', 'Compare with other years', 'Draw conclusions'],
            'y': 0.3
        }
    ]
    
    # Draw workflow
    for i, step in enumerate(steps):
        # Main box
        rect = mpatches.FancyBboxPatch((0.2, step['y'] - 0.08), 0.6, 0.12,
                                      boxstyle="round,pad=0.02",
                                      facecolor='lightblue',
                                      edgecolor='black',
                                      linewidth=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(0.5, step['y'], step['title'], ha='center', va='center',
                fontsize=14, fontweight='bold')
        
        # Tasks
        for j, task in enumerate(step['tasks']):
            ax.text(0.85, step['y'] - j*0.03, f'✓ {task}', 
                    fontsize=10, va='center')
        
        # Arrow to next step
        if i < len(steps) - 1:
            ax.arrow(0.5, step['y'] - 0.09, 0, -0.045, 
                    head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    # Add tips
    tips = [
        'TIP: Start with winter months for easier validation',
        'TIP: Compare adjacent days for consistency',
        'TIP: Document your validation criteria'
    ]
    
    for i, tip in enumerate(tips):
        ax.text(0.5, 0.15 - i*0.03, tip, ha='center', fontsize=10,
                style='italic', color='darkgreen')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Practical Workflow for Using Snow Detection in Research', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    create_all_educational_figures()