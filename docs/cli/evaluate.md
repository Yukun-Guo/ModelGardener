# `evaluate` Command

Evaluate trained machine learning models using test data with intelligent auto-discovery and comprehensive reporting.

## Synopsis

```bash
mg evaluate [OPTIONS]
```

## Description

The `evaluate` command assesses model performance on evaluation data with enhanced auto-discovery capabilities. It provides:

- **Auto-Discovery**: Automatically finds config files, models, and evaluation data
- **Comprehensive Metrics**: Model performance metrics calculation with detailed analysis
- **Multiple Output Formats**: Evaluation results in JSON and YAML formats
- **Timestamped Reports**: Organized results in `evaluation/` folder with metadata
- **Flexible Configuration**: Override auto-discovery with specific paths when needed

## Auto-Discovery Features

### 🔍 Intelligent File Discovery
- **Config Discovery**: Automatically locates `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory
  - Priority order: `final_model.keras` > `model.keras` > `best_model.keras` > latest timestamped model
- **Data Discovery**: Uses evaluation data path from configuration or specified directory

### 📊 Enhanced Reporting
- **Automatic Reports**: Creates timestamped evaluation reports in `evaluation/` folder
- **Multiple Formats**: Saves reports in both JSON and YAML formats
- **Comprehensive Metadata**: Includes model info, config details, and evaluation parameters
- **Optional Saving**: Use `--no-save` for quick evaluations without file generation

## Options

### Core Options (All Optional with Auto-Discovery)

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Configuration file path | Auto-discovered `config.yaml` |
| `--model-path` | `-m` | `str` | Path to trained model file | Auto-discovered latest model |
| `--data-path` | `-d` | `str` | Path to evaluation data | From config or auto-discovered |

### Output Control Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output-format` | `str` | Primary report format (yaml, json) | `yaml` |
| `--no-save` | `flag` | Do not save evaluation results to evaluation/ folder | False |

### Advanced Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--plots` | `flag` | Generate visualization plots | True |
| `--interpretability` | `flag` | Include model interpretability analysis | False |
| `--grad-cam` | `flag` | Generate Grad-CAM visualizations | False |
| `--feature-maps` | `flag` | Visualize feature maps | False |
## Usage Examples

### Auto-Discovery Evaluation (Recommended)

```bash
# Full auto-discovery - finds config.yaml and latest model automatically
mg evaluate

# Auto-discovery with output format preference
mg evaluate --output-format json

# Quick evaluation without saving reports
mg evaluate --no-save
```

### Selective Auto-Discovery

```bash
# Auto-discover config and model, specify custom data path
mg evaluate -d ./custom_test_data

# Auto-discover config, specify custom model
mg evaluate -m ./models/specific_model.keras

# Specify config, auto-discover model and data
mg evaluate -c ./configs/custom_config.yaml
```

### Explicit Configuration

```bash
# Fully explicit evaluation (overrides all auto-discovery)
mg evaluate -c config.yaml -m ./models/trained_model.keras -d ./test_data

# Explicit with output control
mg evaluate -c config.yaml -m ./models/best_model.keras --output-format json

# Explicit evaluation without saving results
mg evaluate -c config.yaml -m ./models/model.keras --no-save
```

### Complete Evaluation Workflow

```bash
# Train, then auto-evaluate
mg train
mg evaluate

# Train and evaluate with custom settings
mg train -c config.yaml
mg evaluate -c config.yaml --output-format json

# Multi-model evaluation
mg evaluate -m ./models/model_v1.keras -o results_v1.json
mg evaluate -m ./models/model_v2.keras -o results_v2.json
```

### Evaluation After Training

```bash
# Standard workflow
mg create project_name
cd project_name
mg train --config config.yaml
mg evaluate --config config.yaml

# Custom evaluation setup
mg evaluate \
    --config config.yaml \
    --model-path ./models/epoch_100_model.keras \
    --data-path ./custom_test_set
```

## Configuration Requirements

The evaluate command requires a configuration file that includes:

### Required Configuration Sections

- **Data Configuration**: Test/evaluation data paths and parameters
- **Model Configuration**: Model architecture and loading information
- **Evaluation Configuration**: Metrics and evaluation settings

### Example Configuration Structure

```yaml
configuration:
  data:
    test_dir: "./data/test"
    data_loader:
      name: "ImageDataGenerator"
      parameters:
        batch_size: 32
        shuffle: false
    
  model:
    model_family: "resnet"
    model_name: "resnet50"
    model_parameters:
      classes: 10
      
  evaluation:
    metrics:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
    save_results: true
    output_dir: "./evaluation_results"
```

## Evaluation Process

When you run the evaluate command, ModelGardener:

1. **Loads Configuration**: Reads evaluation configuration and data paths
2. **Loads Model**: Loads the trained model from specified path
3. **Prepares Data**: Sets up evaluation data loaders
4. **Runs Evaluation**: Executes model evaluation on test data
5. **Calculates Metrics**: Computes performance metrics
6. **Saves Results**: Outputs results in specified format

### Evaluation Metrics

ModelGardener calculates various performance metrics:

- **Accuracy**: Overall classification accuracy
- **Loss**: Model loss on evaluation data
- **Per-class Metrics**: Class-specific performance measures
- **Confusion Matrix**: Classification confusion matrix
- **Custom Metrics**: Additional metrics from configuration

## Output Files

Evaluation generates several output files:

### Results Files
- **Evaluation Results**: Main metrics and performance data
- **Detailed Logs**: Comprehensive evaluation logs
- **Model Performance**: Per-class and overall statistics

### Output Formats
- **YAML**: Human-readable structured format
- **JSON**: Machine-readable structured format
- **Console Output**: Real-time evaluation progress

### Example Output Structure
```
evaluation_results/
├── metrics.yaml         # Main evaluation metrics
├── detailed_results.yaml # Comprehensive results
├── evaluation.log       # Evaluation process logs
└── model_performance.yaml # Model-specific metrics
```

## Integration with Other Commands

The evaluate command integrates with the ModelGardener workflow:

```bash
# Complete ML pipeline
mg create project_name
mg config config.yaml --epochs 100
mg train --config config.yaml
mg evaluate --config config.yaml

# Model comparison workflow
mg train --config config_v1.yaml
mg evaluate --config config_v1.yaml --output-format json
mg train --config config_v2.yaml  
mg evaluate --config config_v2.yaml --output-format json

# Prediction after evaluation
mg evaluate --config config.yaml
mg predict --config config.yaml --input test_image.jpg
```

## Tips and Best Practices

1. **Use separate test data** that wasn't used during training
2. **Save evaluation results** for model comparison and analysis
3. **Use consistent data preprocessing** between training and evaluation
4. **Monitor multiple metrics** to get a complete performance picture
5. **Compare different models** using the same evaluation setup
6. **Document evaluation settings** for reproducibility

## Troubleshooting

### Common Issues

- **Model loading errors**: Check model path and format compatibility
- **Data loading errors**: Verify evaluation data paths and structure
- **Configuration errors**: Use `mg check config.yaml` to validate configuration
- **Memory issues**: Reduce batch size for large datasets

### Error Resolution

```bash
# Check configuration before evaluation
mg check config.yaml
mg evaluate --config config.yaml

# Fix data path issues
mg config config.yaml --val-dir /correct/path/to/test
mg evaluate --config config.yaml

# Use specific model path
mg evaluate --config config.yaml --model-path ./models/specific_model.keras
```

## Related Commands

- [`mg train`](train.md) - Train models for evaluation
- [`mg predict`](predict.md) - Make predictions with evaluated models
- [`mg config`](config.md) - Configure evaluation parameters
- [`mg check`](check.md) - Validate configuration before evaluation
- [`mg deploy`](deploy.md) - Deploy evaluated models
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
