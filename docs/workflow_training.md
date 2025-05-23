# Training Workflow: Teaching the Computer to See

## Overview

Training is like teaching a student by showing them many examples. The computer looks at labeled photos over and over until it learns the patterns.

```mermaid
graph TD
    A[📚 Prepared Data] --> B[🧠 Neural Network]
    B --> C[🔄 Learning Loop]
    C --> D[📈 Check Progress]
    D -->|Not Good Enough| C
    D -->|Good Enough| E[💾 Save Model]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff9c4
    style D fill:#ffecb3
    style E fill:#c8e6c9
```

## What is a Neural Network? 🧠

Think of a neural network as a series of filters that learn to recognize patterns:

```mermaid
graph LR
    A[Photo] --> B[Layer 1<br/>Find Edges]
    B --> C[Layer 2<br/>Find Shapes]
    C --> D[Layer 3<br/>Find Objects]
    D --> E[Layer 4<br/>Find Snow/Weather]
    E --> F[Answer:<br/>Snow? Yes/No]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#c5e1a5
    style F fill:#a5d6a7
```

## Transfer Learning: The Smart Shortcut 🚀

Instead of starting from scratch, we use a pre-trained model (MobileNetV2) that already knows basic image features:

```mermaid
graph TD
    A[MobileNetV2<br/>Pre-trained Model] --> B[Knows:<br/>• Edges<br/>• Textures<br/>• Shapes<br/>• Objects]
    B --> C[Freeze Early Layers<br/>Keep this knowledge]
    C --> D[Add New Layers<br/>Learn snow/weather]
    D --> E[PhenoCAI Model<br/>Specialized for our task]
    
    style A fill:#bbdefb
    style B fill:#e1f5fe
    style C fill:#fff9c4
    style D fill:#c5e1a5
    style E fill:#81c784
```

### Why Transfer Learning?

1. **Faster**: Learn in hours instead of days
2. **Better**: Start with proven knowledge
3. **Less Data**: Need fewer examples
4. **Cheaper**: Use less computing power

## The Training Process 🏃‍♂️

### Step 1: Load the Data

```mermaid
graph LR
    A[Training Data<br/>70%] --> B[Data Loader]
    B --> C[Batches of 32<br/>Images]
    C --> D[To Model]
    
    style A fill:#c8e6c9
    style B fill:#f3e5f5
    style C fill:#fff9c4
    style D fill:#e1f5fe
```

**Batch Processing**: Like studying flashcards in groups of 32 instead of all at once.

### Step 2: Forward Pass (Making Predictions)

```mermaid
graph LR
    A[Input Image] --> B[Current Model]
    B --> C[Prediction:<br/>70% Snow]
    C --> D[Compare with<br/>True Label]
    D --> E[Error:<br/>Wrong by 30%]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff9c4
    style D fill:#ffecb3
    style E fill:#ffcdd2
```

### Step 3: Backward Pass (Learning from Mistakes)

```mermaid
graph RL
    A[Error] --> B[Adjust Weights]
    B --> C[Update Model]
    C --> D[Try Again]
    
    style A fill:#ffcdd2
    style B fill:#fff9c4
    style C fill:#f3e5f5
    style D fill:#c8e6c9
```

### Step 4: Training Loop (Practice Makes Perfect)

```mermaid
graph TD
    A[Start Epoch 1] --> B[Show All Training Images]
    B --> C[Calculate Total Error]
    C --> D[Check Validation Set]
    D --> E{Better than Before?}
    E -->|Yes| F[Save Best Model]
    E -->|No| G[Continue]
    F --> G
    G --> H{More Epochs?}
    H -->|Yes| I[Next Epoch]
    H -->|No| J[Training Complete]
    I --> B
    
    style A fill:#e3f2fd
    style C fill:#ffecb3
    style D fill:#fff9c4
    style F fill:#c8e6c9
    style J fill:#81c784
```

**Epoch**: One complete pass through all training data (like reading a textbook once).

## Key Concepts Explained Simply

### Learning Rate 📊

How big steps the model takes when learning:

```mermaid
graph LR
    A[Too Small<br/>0.00001] --> B[Learns Very Slowly]
    C[Just Right<br/>0.001] --> D[Learns Well]
    E[Too Large<br/>0.1] --> F[Jumps Around<br/>Never Learns]
    
    style A fill:#e3f2fd
    style C fill:#c8e6c9
    style E fill:#ffcdd2
```

### Overfitting vs Underfitting 🎯

```mermaid
graph TD
    A[Training Data] --> B{Model Performance}
    B --> C[Underfitting<br/>😴]
    B --> D[Just Right<br/>😊]
    B --> E[Overfitting<br/>🤓]
    
    C --> F[Too Simple<br/>Can't Learn Patterns]
    D --> G[Good Balance<br/>Generalizes Well]
    E --> H[Too Complex<br/>Memorizes Instead of Learning]
    
    style C fill:#ffcdd2
    style D fill:#c8e6c9
    style E fill:#ffecb3
```

### Loss Function (How Wrong We Are) 📉

The loss function measures mistakes:

```mermaid
graph TD
    A[Prediction: 80% Snow] --> C[Loss Function]
    B[Truth: 100% Snow] --> C
    C --> D[Loss: 0.2<br/>Lower is Better]
    
    style A fill:#fff9c4
    style B fill:#c8e6c9
    style C fill:#f3e5f5
    style D fill:#ffecb3
```

## Training Parameters 🎛️

| Parameter | What it Does | Typical Value | Analogy |
|-----------|--------------|---------------|---------|
| Epochs | How many times to study | 20 | Reading a book 20 times |
| Batch Size | Images per group | 32 | Flashcards per session |
| Learning Rate | How fast to learn | 0.001 | Walking speed |
| Validation Split | Data for checking | 10% | Practice quiz |

## Monitoring Training 📊

### Loss Curves

```mermaid
graph TD
    subgraph Good Training
        A1[High Loss] --> B1[Medium Loss] --> C1[Low Loss]
    end
    
    subgraph Overfitting
        A2[High Loss] --> B2[Training: Low<br/>Validation: High]
    end
    
    style C1 fill:#c8e6c9
    style B2 fill:#ffecb3
```

### Early Stopping

Stop training when validation performance stops improving:

```mermaid
graph LR
    A[Epoch 1<br/>Loss: 0.8] --> B[Epoch 5<br/>Loss: 0.3]
    B --> C[Epoch 10<br/>Loss: 0.2]
    C --> D[Epoch 15<br/>Loss: 0.2]
    D --> E[Epoch 20<br/>Loss: 0.2]
    
    C -->|No Improvement| F[Stop Here!<br/>Save Best Model]
    
    style C fill:#c8e6c9
    style F fill:#81c784
```

## Data Augmentation 🎨

Making training data more diverse by creating variations:

```mermaid
graph TD
    A[Original Image] --> B[Rotate ±15°]
    A --> C[Flip Horizontal]
    A --> D[Adjust Brightness]
    A --> E[Add Noise]
    
    B --> F[5x More Training Data!]
    C --> F
    D --> F
    E --> F
    
    style A fill:#e3f2fd
    style F fill:#c8e6c9
```

## Fine-Tuning Strategy 🎯

Two-stage training for better results:

```mermaid
graph TD
    A[Stage 1: Train New Layers<br/>10 epochs, LR=0.001] --> B[Freeze Pre-trained Layers]
    B --> C[Stage 2: Fine-tune All<br/>10 epochs, LR=0.00005]
    C --> D[Unfreeze All Layers]
    D --> E[Final Model]
    
    style A fill:#fff9c4
    style C fill:#c5e1a5
    style E fill:#81c784
```

## Common Training Problems and Solutions

### Problem: Loss Not Decreasing
```mermaid
graph LR
    A[Stuck Loss] --> B{Check}
    B --> C[Learning Rate<br/>Too Small?]
    B --> D[Bad Data?]
    B --> E[Model Too Simple?]
```

### Problem: Validation Loss Increasing
```mermaid
graph LR
    A[Overfitting] --> B{Solutions}
    B --> C[Add Dropout]
    B --> D[More Data]
    B --> E[Simpler Model]
    B --> F[Early Stopping]
```

## Training Commands

```bash
# Basic training with MobileNetV2
uv run phenocai train model dataset.csv --epochs 20

# With custom parameters
uv run phenocai train model dataset.csv \
    --model-type mobilenet \
    --batch-size 32 \
    --learning-rate 0.001 \
    --epochs 30

# Train simple CNN (faster, lower accuracy)
uv run phenocai train model dataset.csv \
    --model-type simple-cnn \
    --epochs 30 \
    --batch-size 64

# Train with specific output directory
uv run phenocai train model dataset.csv \
    --model-type mobilenet \
    --output-dir trained_models/experiment_1/
```

The training system will:
- Automatically handle train/test/val splits from your dataset
- Apply data augmentation to training images
- Save checkpoints during training
- Track best model based on validation accuracy
- Generate training history plots

## Training Checklist ✅

- [ ] Data loaded correctly
- [ ] Model architecture chosen
- [ ] Training parameters set
- [ ] Validation data separate
- [ ] Monitoring setup
- [ ] Early stopping enabled
- [ ] Best model saving enabled

## What Happens Next?

After training completes:
1. **Best model saved** → Contains learned knowledge
2. **Training history saved** → Shows how learning progressed
3. **Ready for evaluation** → Time to test how well it learned!

## Next Step

Proceed to [Evaluation Workflow](workflow_evaluation.md) to test your trained model!