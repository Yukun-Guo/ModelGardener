# `evaluate` Command

Comprehensive model evaluation with detailed metrics, visualizations, and performance analysis across multiple aspects of model performance.

## Synopsis

```bash
mg evaluate [OPTIONS]
```

## Description

The `evaluate` command provides comprehensive model assessment including:

- Detailed performance metrics and statistical analysis
- Confusion matrices and classification reports
- ROC curves and precision-recall analysis
- Per-class performance breakdown
- Model interpretability and visualization
- Performance benchmarking and comparison
- Export results in multiple formats

## Options

### Model and Configuration

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Path to YAML configuration file | `config.yaml` |
| `--model` | `-m` | `str` | Path to trained model file | From config |
| `--weights` | `-w` | `str` | Path to model weights file | None |

### Data Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--test-dir` | `str` | Test dataset directory | From config |
| `--batch-size` | `int` | Evaluation batch size | From config |
| `--subset` | `str` | Data subset to evaluate (train/val/test) | `test` |

### Evaluation Scope

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--metrics` | `str` | Comma-separated list of metrics to compute | `all` |
| `--per-class` | `flag` | Include per-class analysis | True |
| `--confusion-matrix` | `flag` | Generate confusion matrix | True |
| `--roc-curves` | `flag` | Generate ROC curves | True |
| `--precision-recall` | `flag` | Generate precision-recall curves | True |

### Visualization Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--plots` | `flag` | Generate visualization plots | True |
| `--interpretability` | `flag` | Include model interpretability analysis | False |
| `--grad-cam` | `flag` | Generate Grad-CAM visualizations | False |
| `--feature-maps` | `flag` | Visualize feature maps | False |

### Output Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output-dir` | `str` | Directory for evaluation outputs | `./evaluation_results` |
| `--format` | `str` | Output format (json, csv, html, pdf) | `json,html` |
| `--detailed-report` | `flag` | Generate detailed HTML report | True |
| `--save-predictions` | `flag` | Save individual predictions | False |

### Performance Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--benchmark` | `flag` | Include performance benchmarking | False |
| `--timing` | `flag` | Measure inference timing | True |
| `--memory-usage` | `flag` | Monitor memory usage | True |

## Usage Examples

### Basic Evaluation

```bash
# Evaluate with default settings
mg evaluate

# Evaluate specific model
mg evaluate --model ./logs/models/best_model.keras

# Evaluate with custom config
mg evaluate --config evaluation_config.yaml
```

### Custom Data Evaluation

```bash
# Evaluate on specific test set
mg evaluate --test-dir ./custom_test_data

# Evaluate on validation set
mg evaluate --subset val

# Evaluate with custom batch size
mg evaluate --batch-size 64
```

### Comprehensive Analysis

```bash
# Full evaluation with all features
mg evaluate \
    --per-class \
    --confusion-matrix \
    --roc-curves \
    --precision-recall \
    --interpretability \
    --grad-cam \
    --benchmark

# Quick evaluation with basic metrics
mg evaluate --metrics accuracy,precision,recall,f1
```

### Visualization and Reporting

```bash
# Generate detailed HTML report
mg evaluate --detailed-report --format html,pdf

# Save individual predictions
mg evaluate --save-predictions --format csv

# Custom output directory
mg evaluate --output-dir ./experiment_1/evaluation
```

### Performance Benchmarking

```bash
# Performance benchmarking
mg evaluate --benchmark --timing --memory-usage

# Interpretability analysis
mg evaluate --interpretability --grad-cam --feature-maps
```

## Available Metrics

### Classification Metrics

**Basic Metrics:**
- Accuracy (overall and per-class)
- Precision (macro, micro, weighted)
- Recall (macro, micro, weighted)
- F1-score (macro, micro, weighted)
- Cohen's Kappa
- Matthews Correlation Coefficient

**Advanced Metrics:**
- ROC AUC (one-vs-rest, one-vs-one)
- Average Precision Score
- Log Loss
- Balanced Accuracy
- Top-k Accuracy (configurable k)

**Statistical Metrics:**
- Confidence intervals
- Statistical significance tests
- Bootstrap confidence intervals
- Cross-validation consistency

### Detailed Analysis

**Confusion Matrix Analysis:**
- Raw confusion matrix
- Normalized confusion matrix
- Per-class error analysis
- Misclassification patterns

**ROC Analysis:**
- Individual class ROC curves
- Micro-averaged ROC curve
- Macro-averaged ROC curve
- ROC AUC scores with confidence intervals

**Precision-Recall Analysis:**
- Per-class precision-recall curves
- Average precision scores
- Precision-recall AUC
- Optimal threshold analysis

## Configuration File Structure

Evaluation-specific configuration in YAML:

```yaml
evaluation:
  model:
    model_path: "./logs/models/best_model.keras"
    weights_path: null
  
  data:
    test_dir: "./data/test"
    batch_size: 32
    subset: "test"
    preprocessing:
      resize:
        height: 224
        width: 224
      normalization:
        rescale: 0.00392156862
  
  metrics:
    include_metrics:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
      - "roc_auc"
      - "average_precision"
      - "cohen_kappa"
      - "matthews_corrcoef"
    
    per_class_analysis: true
    confidence_intervals: true
    statistical_tests: true
  
  visualization:
    confusion_matrix: true
    roc_curves: true
    precision_recall_curves: true
    class_distribution: true
    error_analysis: true
  
  interpretability:
    enable: false
    methods:
      - "grad_cam"
      - "integrated_gradients"
      - "lime"
    sample_size: 100
  
  output:
    output_dir: "./evaluation_results"
    formats:
      - "json"
      - "html"
      - "csv"
    detailed_report: true
    save_predictions: false
    save_misclassifications: true
  
  performance:
    benchmark: false
    timing_iterations: 100
    memory_monitoring: true
    resource_usage: true
```

## Evaluation Pipeline

### Data Loading and Preprocessing

1. **Data Discovery:**
   - Automatic test set detection
   - Custom directory scanning
   - Class label extraction

2. **Preprocessing Pipeline:**
   - Image resizing and normalization
   - Same preprocessing as training
   - Batch processing optimization

3. **Data Validation:**
   - Input shape verification
   - Class consistency checks
   - Data quality assessment

### Model Evaluation

1. **Model Loading:**
   - Keras model loading
   - Weight restoration
   - Architecture verification

2. **Prediction Generation:**
   - Batch prediction processing
   - Probability score extraction
   - Confidence interval calculation

3. **Metrics Calculation:**
   - Per-sample metrics
   - Aggregate statistics
   - Cross-validation consistency

### Analysis and Visualization

1. **Statistical Analysis:**
   - Significance testing
   - Confidence intervals
   - Error distribution analysis

2. **Visualization Generation:**
   - Confusion matrices
   - ROC and PR curves
   - Distribution plots

3. **Interpretability Analysis:**
   - Grad-CAM heatmaps
   - Feature importance
   - Prediction explanations

## Output Structure

Evaluation generates comprehensive outputs:

```
evaluation_results/
├── metrics/
│   ├── overall_metrics.json      # Aggregate metrics
│   ├── per_class_metrics.json    # Per-class analysis
│   ├── confusion_matrix.json     # Confusion matrix data
│   ├── roc_data.json            # ROC curve data
│   ├── precision_recall_data.json # PR curve data
│   └── statistical_tests.json   # Statistical analysis
├── visualizations/
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── confusion_matrix_normalized.png
│   ├── roc_curves.png            # ROC curves
│   ├── precision_recall_curves.png # PR curves
│   ├── class_distribution.png    # Class distribution
│   ├── error_analysis.png        # Error analysis
│   └── performance_comparison.png
├── interpretability/
│   ├── grad_cam_samples/         # Grad-CAM visualizations
│   ├── feature_maps/             # Feature map visualizations
│   ├── lime_explanations/        # LIME explanations
│   └── integrated_gradients/     # Integrated gradients
├── predictions/
│   ├── predictions.csv           # All predictions
│   ├── misclassifications.csv    # Misclassified samples
│   ├── confidence_scores.csv     # Confidence analysis
│   └── top_k_predictions.csv     # Top-k predictions
├── reports/
│   ├── evaluation_report.html    # Detailed HTML report
│   ├── evaluation_report.pdf     # PDF report
│   ├── summary_report.txt        # Text summary
│   └── executive_summary.md      # Executive summary
├── performance/
│   ├── timing_analysis.json      # Inference timing
│   ├── memory_usage.json         # Memory utilization
│   ├── benchmark_results.json    # Performance benchmark
│   └── resource_usage.json       # System resource usage
└── config/
    ├── evaluation_config.yaml    # Configuration used
    ├── model_info.json          # Model information
    └── dataset_info.json        # Dataset information
```

## Advanced Features

### Model Interpretability

**Grad-CAM Visualization:**
```bash
# Generate Grad-CAM heatmaps
mg evaluate --grad-cam --interpretability
```

**Feature Map Analysis:**
```bash
# Visualize intermediate feature maps
mg evaluate --feature-maps
```

**LIME Explanations:**
```bash
# Local interpretable model explanations
mg evaluate --interpretability
```

### Performance Benchmarking

**Inference Speed:**
- Batch processing timing
- Single image inference
- Throughput analysis
- Latency distribution

**Memory Usage:**
- Peak memory consumption
- Memory efficiency analysis
- GPU memory utilization
- Memory leak detection

**Resource Utilization:**
- CPU usage patterns
- GPU utilization
- I/O performance
- System load analysis

### Statistical Analysis

**Confidence Intervals:**
- Bootstrap confidence intervals
- Wilson score intervals
- Exact binomial intervals
- Bayesian credible intervals

**Significance Testing:**
- McNemar's test for model comparison
- Paired t-tests for metric differences
- Chi-square tests for independence
- Effect size calculations

### Error Analysis

**Misclassification Analysis:**
- Common error patterns
- Confusion pair analysis
- Error severity assessment
- Improvement recommendations

**Failure Case Analysis:**
- Edge case identification
- Systematic failure patterns
- Data quality issues
- Model limitation analysis

## Integration with Other Commands

### After Training

```bash
# Evaluate immediately after training
mg train --config config.yaml
mg evaluate --config config.yaml --model ./logs/models/best_model.keras
```

### Before Deployment

```bash
# Comprehensive evaluation before deployment
mg evaluate \
    --model production_model.keras \
    --benchmark \
    --detailed-report \
    --format html,pdf
```

### Model Comparison

```bash
# Compare multiple models
mg evaluate --model model_1.keras --output-dir ./eval_model_1
mg evaluate --model model_2.keras --output-dir ./eval_model_2
```

## Custom Metrics Integration

### Custom Metric Functions

```python
# In custom_modules/custom_metrics.py
def custom_f_beta_score(beta=2.0):
    """Custom F-beta score with configurable beta."""
    def f_beta_score(y_true, y_pred):
        # Implementation
        pass
    return f_beta_score

def domain_specific_metric(y_true, y_pred):
    """Domain-specific evaluation metric."""
    # Custom implementation
    pass
```

### Configuration Integration

```yaml
metadata:
  custom_functions:
    metrics:
      - "custom_f_beta_score"
      - "domain_specific_metric"
```

## Best Practices

### Evaluation Strategy

1. **Comprehensive Assessment:**
   - Use multiple metrics for robust evaluation
   - Include both aggregate and per-class analysis
   - Consider domain-specific requirements

2. **Statistical Rigor:**
   - Calculate confidence intervals
   - Perform significance testing
   - Use appropriate sample sizes

3. **Interpretability:**
   - Include model interpretability analysis
   - Analyze failure cases
   - Document limitations

### Performance Considerations

1. **Efficient Evaluation:**
   - Use appropriate batch sizes
   - Monitor memory usage
   - Optimize data loading

2. **Resource Management:**
   - Clean up temporary files
   - Manage disk space for outputs
   - Monitor system resources

## Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Verify model path and format
mg evaluate --model ./correct/path/model.keras
```

**Memory Issues:**
```bash
# Reduce batch size
mg evaluate --batch-size 16

# Disable memory-intensive features
mg evaluate --no-interpretability --no-grad-cam
```

**Visualization Errors:**
```bash
# Check output directory permissions
chmod 755 ./evaluation_results

# Disable problematic visualizations
mg evaluate --no-plots
```

## See Also

- [Training Command](train.md)
- [Prediction Command](predict.md)
- [Model Interpretability Tutorial](../tutorials/interpretability.md)
- [Metrics and Evaluation Guide](../tutorials/evaluation-metrics.md)
- [Statistical Analysis Guide](../tutorials/statistical-analysis.md)
