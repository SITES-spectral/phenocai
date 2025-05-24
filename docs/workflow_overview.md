# PhenoCAI Workflow Overview

## What is PhenoCAI?

Imagine you have thousands of photos taken by cameras watching nature every day. PhenoCAI is like a smart assistant that learns to look at these photos and answer questions like:
- "Is there snow in this picture?"
- "Is the image too blurry to use?"
- "Are there any problems with the camera lens?"

This document explains how PhenoCAI works, step by step, using simple language and visual diagrams.

## The Big Picture

```mermaid
graph TD
    A[Camera Takes Photos] --> B[Humans Label Some Photos]
    B --> C[Computer Learns Patterns]
    C --> D[Computer Labels New Photos]
    D --> E[Scientists Use Results]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
```

## Key Concepts

### 1. What is Transfer Learning?

Transfer learning is like teaching someone to ride a motorcycle when they already know how to ride a bicycle. We use knowledge from one task to help with another similar task.

```mermaid
graph LR
    A[Pre-trained Model<br/>Knows Basic Image Features] --> B[Add Our Knowledge<br/>About Snow and Weather]
    B --> C[PhenoCAI Model<br/>Specialized for Our Photos]
    
    style A fill:#bbdefb
    style B fill:#ffccbc
    style C fill:#c5e1a5
```

### 2. What are ROIs (Regions of Interest)?

Instead of looking at the whole photo, we often focus on specific parts:

```mermaid
graph TD
    A[Full Image] --> B[ROI_00: Full Image minus Sky]
    A --> C[ROI_01: Field Area]
    A --> D[ROI_02: Forest Edge]
    A --> E[ROI_03: Specific Crop Area]
    
    style A fill:#e3f2fd
    style B fill:#f5f5f5
    style C fill:#e1f5fe
    style D fill:#a5d6a7
    style E fill:#81c784
```

**Special Note about ROI_00**: ROI_00 is automatically calculated to exclude the sky region using advanced color detection algorithms. This makes it ideal for cross-station comparisons since it provides a consistent view of the ground/vegetation across different camera angles and locations.

### 3. What are Quality Flags?

Quality flags are like warning labels on photos:

```mermaid
graph LR
    A[Photo] --> B{Quality Check}
    B -->|Good| C[Clean Image]
    B -->|Issues| D[Has Flags]
    D --> E[Fog]
    D --> F[Too Bright]
    D --> G[Water Drops]
    D --> H[Blurry]
    
    style C fill:#c8e6c9
    style D fill:#ffecb3
    style E fill:#e1f5fe
    style F fill:#fff9c4
    style G fill:#b3e5fc
    style H fill:#d1c4e9
```

### 4. Multiple Stations and Instruments

PhenoCAI works with cameras at different research stations across Sweden:

```mermaid
graph TD
    A[Research Stations] --> B[Lönnstorp<br/>Agricultural]
    A --> C[Röbäcksdalen<br/>Agricultural]
    A --> D[Abisko<br/>Subarctic]
    
    B --> E[PHE01<br/>Main Camera]
    B --> F[PHE02<br/>Secondary Camera]
    
    C --> G[PHE01<br/>Agricultural View]
    C --> H[FOR01<br/>Forest View]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#ffecb3
    style F fill:#ffecb3
    style G fill:#c8e6c9
    style H fill:#81c784
```

Each station can have multiple cameras (instruments) looking at different areas, so PhenoCAI helps you:
- Switch between stations easily
- Select specific cameras/instruments
- Create datasets for different viewing angles
- Validate that instruments actually exist

## The Three Main Stages

### Stage 1: Prepare the Data
We organize our photos and labels so the computer can learn from them.

### Stage 2: Train the Model
The computer studies the labeled photos to learn patterns.

### Stage 3: Use the Model
The trained model can now label new photos automatically.

## Why Do We Need This?

Scientists use cameras to watch how nature changes through the seasons. But with thousands of photos every day, it's impossible for humans to look at them all. PhenoCAI helps by:

1. **Saving Time**: Automatically labeling thousands of photos
2. **Being Consistent**: Always using the same rules
3. **Finding Patterns**: Spotting changes humans might miss
4. **Working 24/7**: Processing photos day and night
5. **Cross-Station Analysis**: Compare data across different locations using ROI_00

## Cross-Station Capabilities

PhenoCAI now supports training models at one station and applying them to others. This is made possible through:

- **ROI_00 Standardization**: Automatically calculated region that excludes sky, providing consistent views across stations
- **Station Configuration**: Pre-calculated ROI_00 definitions stored in stations.yaml for efficiency
- **Universal Models**: Train once, apply everywhere using ROI_00 filtering

## Next Steps

Now that you understand the big picture, let's dive deeper into each stage:

1. [Data Preparation Workflow](workflow_data_preparation.md) - How we get photos ready for learning
2. [Training Workflow](workflow_training.md) - How the computer learns
3. [Evaluation Workflow](workflow_evaluation.md) - How we check if it learned well
4. [Prediction Workflow](workflow_prediction.md) - How we use it on new photos

## Glossary

- **Model**: A computer program that has learned to recognize patterns
- **Dataset**: A collection of photos with labels
- **Training**: Teaching the computer by showing it examples
- **ROI**: Region of Interest - a specific part of a photo
- **Flag**: A label indicating a quality issue with a photo
- **Transfer Learning**: Using knowledge from one task to help with another