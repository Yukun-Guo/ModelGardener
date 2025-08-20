# Metrics Configuration Usage Guide

The MetricsGroup has been refactored to support flexible metric selection for single and multiple output models.

## Features

### 1. Single Output Model
- **Shared Metrics**: Select one or multiple metrics for the single output
- **Example**: `"Accuracy,Top-K Categorical Accuracy,Precision"`

### 2. Multiple Output Model
- **Shared Metrics Strategy**: Same metrics applied to all outputs
- **Different Metrics Strategy**: Different metrics for each output individually

### 3. Metrics Selection
- Use comma-separated values in the `selected_metrics` field
- Available built-in metrics:
  - Accuracy
  - Categorical Accuracy
  - Sparse Categorical Accuracy
  - Top-K Categorical Accuracy (with configurable K)
  - Precision (with averaging and class_id options)
  - Recall (with averaging and class_id options)
  - F1 Score (with averaging and class_id options)
  - AUC (with curve type and multiclass strategy)
  - Mean Squared Error
  - Mean Absolute Error

### 4. Custom Metrics
- Load custom metrics from Python files using the "Load Custom Metrics" button
- Custom metrics can be functions or TensorFlow metric classes
- Once loaded, they become available for selection

## Configuration Structure

```
Metrics
├── Model Output Configuration
│   ├── num_outputs: 1 (for single) or >1 (for multiple)
│   ├── output_names: "main_output" or "output1,output2"
│   └── metrics_strategy: "shared" or "different_per_output"
├── Metrics Selection (for single/shared)
│   ├── selected_metrics: "Accuracy,Precision"
│   ├── Accuracy Config
│   │   └── name: "accuracy"
│   └── Precision Config
│       ├── name: "precision"
│       ├── average: "macro"
│       └── class_id: 0
└── Load Custom Metrics (button)
```

## Usage Examples

### Single Output with Multiple Metrics
1. Set `num_outputs` = 1
2. Set `selected_metrics` = "Accuracy,F1 Score,AUC"
3. Configure parameters for each metric

### Multiple Outputs with Shared Metrics
1. Set `num_outputs` = 2
2. Set `output_names` = "main_output,auxiliary_output"
3. Set `metrics_strategy` = "shared_metrics_all_outputs"
4. Set `selected_metrics` = "Accuracy,Precision"

### Multiple Outputs with Different Metrics
1. Set `num_outputs` = 2
2. Set `metrics_strategy` = "different_metrics_per_output"
3. For each output:
   - Output 1: `selected_metrics` = "Accuracy,Precision"
   - Output 2: `selected_metrics` = "F1 Score,AUC"

## Benefits

1. **Flexible Selection**: Choose exactly which metrics you need
2. **Output-Specific**: Different metrics for different outputs
3. **Dynamic Configuration**: Parameters update based on selection
4. **Custom Support**: Load and use custom metrics seamlessly
5. **Clean Interface**: No more enable/disable checkboxes - just select what you need
