# Evaluation Workflow: Testing What the Computer Learned

## Overview

Evaluation is like giving the trained model a final exam. We test it on photos it has never seen before to check if it really learned or just memorized.

```mermaid
graph TD
    A[🧠 Trained Model] --> B[📝 Test Set]
    B --> C[🔍 Make Predictions]
    C --> D[📊 Compare with Truth]
    D --> E[📈 Calculate Scores]
    E --> F[📋 Report Card]
    
    style A fill:#f3e5f5
    style B fill:#ffecb3
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#c5e1a5
    style F fill:#81c784
```

## Why Evaluate? 🤔

Imagine a student who memorized all the answers to practice questions but can't solve new problems. We need to check if our model truly understands!

```mermaid
graph LR
    A[Training Performance] --> B{Real Understanding?}
    B -->|Just Memorized| C[❌ Bad on New Data]
    B -->|Truly Learned| D[✅ Good on New Data]
    
    style C fill:#ffcdd2
    style D fill:#c8e6c9
```

## The Test Set: Never Seen Before! 🎯

Remember our data split? The test set was kept hidden during training:

```mermaid
graph TD
    A[Original Data] --> B[Training 70%<br/>📚 For Learning]
    A --> C[Validation 10%<br/>📝 For Checking]
    A --> D[Test 20%<br/>🎓 Final Exam]
    
    D --> E[Never Used Until Now!]
    
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#ffcdd2,stroke:#f44336,stroke-width:3px
    style E fill:#ff5252,color:#fff
```

## Evaluation Metrics Explained Simply 📊

### 1. Accuracy: Overall Score

```mermaid
graph TD
    A[100 Predictions] --> B[85 Correct ✅]
    A --> C[15 Wrong ❌]
    B --> D[Accuracy = 85%]
    
    style B fill:#c8e6c9
    style C fill:#ffcdd2
    style D fill:#81c784
```

**Accuracy** = How many you got right out of total

### 2. Confusion Matrix: Detailed Breakdown

```mermaid
graph TD

subgraph Confusion Matrix

A[Reality ↓ / Prediction →] --> B[No Snow / Snow]

B --> C[No Snow: 800 ✅, 50 ❌]

B --> D[Snow: 100 ❌, 150 ✅]

end

style C fill:#e8f5e9

style D fill:#e8f5e9
```

This shows:
- **True Positives (150)**: Correctly found snow ✅
- **True Negatives (800)**: Correctly found no snow ✅
- **False Positives (50)**: Said snow when there wasn't ❌
- **False Negatives (100)**: Missed snow that was there ❌

### 3. Precision and Recall: Different Perspectives

```mermaid
graph TD

subgraph Precision

A["Said 'Snow' 200 times"] --> B["150 Correct ✅"]

A --> C["50 Wrong ❌"]

B --> D["Precision = 75%"]

end

subgraph Recall

E["Actually 250 Snow Cases"] --> F["Found 150 ✅"]

E --> G["Missed 100 ❌"]

F --> H["Recall = 60%"]

end

style D fill:#81c784

style H fill:#ffc107
```

- **Precision**: When you say "snow", how often are you right?
- **Recall**: Of all snow cases, how many did you find?

### 4. F1 Score: The Balance

```mermaid
graph LR
    A[Precision<br/>75%] --> C[F1 Score]
    B[Recall<br/>60%] --> C
    C --> D[F1 = 67%<br/>Balanced Measure]
    
    style A fill:#81c784
    style B fill:#ffc107
    style D fill:#4caf50,color:#fff
```

F1 Score balances precision and recall (like averaging them smartly).

## Evaluation by Categories 🏷️

### By ROI (Region of Interest)

```mermaid
graph TD
    A[Overall: 85%] --> B[ROI_00: 88%<br/>Full Image]
    A --> C[ROI_01: 82%<br/>Sky Area]
    A --> D[ROI_02: 79%<br/>Ground Area]
    
    B --> E[Best Performance]
    D --> F[Needs Improvement]
    
    style B fill:#4caf50,color:#fff
    style D fill:#ff9800,color:#fff
```

### By Quality Condition

```mermaid
graph TD
    A[Performance by Condition] --> B[Clean Images: 92% ✅]
    A --> C[Foggy Images: 73% ⚠️]
    A --> D[Bright Images: 68% ⚠️]
    A --> E[Multiple Issues: 51% ❌]
    
    style B fill:#4caf50,color:#fff
    style C fill:#ffc107
    style D fill:#ff9800,color:#fff
    style E fill:#f44336,color:#fff
```

## Visual Evaluation Tools 📈

### 1. ROC Curve (How Well We Separate Classes)

```mermaid
graph TD
    subgraph ROC Curve
        A[Perfect Model 📐] --> B[Area = 1.0]
        C[Good Model 📊] --> D[Area = 0.85]
        E[Random Guess 🎲] --> F[Area = 0.5]
    end
    
    style B fill:#4caf50,color:#fff
    style D fill:#81c784
    style F fill:#ffcdd2
```

Better models have curves closer to the top-left corner.

### 2. Prediction Examples

```mermaid
graph TD

subgraph Correct_Predictions

A[Clear Snow] --> B[Predicted: Snow ✓]

C[Sunny Day] --> D[Predicted: No Snow ✓]

end

subgraph Wrong_Predictions

E[Light Snow] --> F[Predicted: No Snow ✗]

G[Bright Ground] --> H[Predicted: Snow ✗]

end

style B fill:#c8e6c9

style D fill:#c8e6c9

style F fill:#ffcdd2

style H fill:#ffcdd2
```

## Error Analysis 🔍

Understanding why the model makes mistakes:

```mermaid
graph TD
    A[Common Errors] --> B[Bright Surfaces<br/>Confused with Snow]
    A --> C[Light Snow<br/>Too Subtle to Detect]
    A --> D[Foggy Conditions<br/>Can't See Clearly]
    A --> E[Wet Ground<br/>Reflects Like Snow]
    
    B --> F[Solution: More Training<br/>Examples of Bright Ground]
    C --> G[Solution: Adjust<br/>Detection Threshold]
    D --> H[Solution: Separate<br/>Fog Model]
    
    style B fill:#ffecb3
    style C fill:#ffecb3
    style D fill:#ffecb3
    style E fill:#ffecb3
    style F fill:#c5e1a5
    style G fill:#c5e1a5
    style H fill:#c5e1a5
```

## Performance by Time of Day ⏰

```mermaid
graph TD
    A[Time Analysis] --> B[Morning: 87%<br/>Good Light]
    A --> C[Noon: 83%<br/>Harsh Shadows]
    A --> D[Evening: 76%<br/>Low Light]
    A --> E[Night: 45%<br/>Too Dark]
    
    style B fill:#4caf50,color:#fff
    style C fill:#81c784
    style D fill:#ffc107
    style E fill:#f44336,color:#fff
```

## Comparing Models 🏁

When you train multiple models, compare them:

```mermaid
graph TD
    subgraph Model Comparison
        A[MobileNetV2] --> B[Accuracy: 85%<br/>Speed: Fast]
        C[Custom CNN] --> D[Accuracy: 78%<br/>Speed: Very Fast]
        E[With Augmentation] --> F[Accuracy: 88%<br/>Speed: Fast]
    end
    
    F --> G[🏆 Best Model]
    
    style F fill:#4caf50,color:#fff
    style G fill:#ffd700
```

## Evaluation Commands (When Implemented)

```bash
# Basic evaluation
uv run phenocai evaluate model saved_model.h5 test_dataset.csv

# Detailed analysis
uv run phenocai evaluate model saved_model.h5 test_dataset.csv \
    --save-predictions \
    --generate-plots \
    --output-dir results/

# Compare multiple models
uv run phenocai evaluate benchmark \
    --models-dir trained_models/ \
    --dataset test_dataset.csv
```

## Evaluation Report Card 📋

A good evaluation report includes:

```mermaid
graph TD
    A[Evaluation Report] --> B[Overall Metrics<br/>Accuracy, F1, etc.]
    A --> C[Per-Class Performance<br/>Snow vs No Snow]
    A --> D[Per-ROI Results<br/>Which regions work best]
    A --> E[Error Analysis<br/>Common mistakes]
    A --> F[Recommendations<br/>How to improve]
    
    style A fill:#e3f2fd
    style B fill:#c5e1a5
    style C fill:#c5e1a5
    style D fill:#c5e1a5
    style E fill:#ffecb3
    style F fill:#81c784
```

## Understanding Results 🎯

### Good Results Look Like:
- ✅ High accuracy (>80%)
- ✅ Balanced precision and recall
- ✅ Good performance across all ROIs
- ✅ Handles common quality issues

### Warning Signs:
- ⚠️ Big gap between training and test accuracy
- ⚠️ Very low recall (missing many cases)
- ⚠️ Poor performance on specific conditions
- ⚠️ Works only on clean images

## Making Improvements 🔧

Based on evaluation results:

```mermaid
graph TD
    A[Low Accuracy] --> B[Need More Data]
    C[Low Recall] --> D[Adjust Threshold]
    E[Condition-Specific Errors] --> F[Train Specialized Models]
    G[Overfitting] --> H[Add Regularization]
    
    style B fill:#c5e1a5
    style D fill:#c5e1a5
    style F fill:#c5e1a5
    style H fill:#c5e1a5
```

## Evaluation Checklist ✅

- [ ] Test set never used during training
- [ ] Calculate multiple metrics (not just accuracy)
- [ ] Check performance by category (ROI, condition)
- [ ] Analyze common errors
- [ ] Compare with baseline/other models
- [ ] Generate visual reports
- [ ] Document findings

## Next Step

Once you're happy with evaluation results, proceed to [Prediction Workflow](workflow_prediction.md) to use your model on new data!