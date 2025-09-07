# `predict` Command

Generate predictions on new data using trained models with support for single images, batch processing, and configurable output.

## Synopsis

```bash
mg predict [OPTIONS]
```

## Description

The `predict` command enables prediction on new data using trained models. It supports:

- Single image and batch prediction processing
- Top-k predictions with confidence scores
- Configurable batch sizes for performance optimization
- Multiple output formats (JSON/YAML)
- Custom model and input paths

## Options

### Required Options

| Option | Short | Type | Description | Required |
|--------|-------|------|-------------|----------|
| `--config` | `-c` | `str` | Configuration file path | Yes |
| `--input` | `-i` | `str` | Input image file or directory | Yes |

### Optional Model and Output

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--model-path` | | `str` | Path to trained model | From config |
| `--output` | `-o` | `str` | Output file for results (JSON/YAML) | Console output |

### Prediction Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--top-k` | `int` | Number of top predictions to show | 5 |
| `--batch-size` | `int` | Batch size for processing | 32 |

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--visualize` | `flag` | Generate prediction visualizations | False |
| `--grad-cam` | `flag` | Generate Grad-CAM attention maps | False |
| `--overlay` | `flag` | Overlay predictions on images | False |
| `--confidence-bars` | `flag` | Show confidence score bars | True |

## Usage Examples

### Single Image Prediction

```bash
# Predict single image
mg predict --config config.yaml --input image.jpg

# Predict with custom model path
mg predict --config config.yaml --input image.jpg --model-path ./models/my_model.keras

# Predict with top-3 results
mg predict --config config.yaml --input image.jpg --top-k 3

# Save prediction results to file
mg predict \
    --config config.yaml \
    --input image.jpg \
    --output predictions.json \
    --top-k 10
```

### Batch Directory Prediction

```bash
# Predict entire directory
mg predict --config config.yaml --input ./test_images/

# Batch prediction with custom batch size
mg predict \
    --config config.yaml \
    --input ./large_dataset/ \
    --batch-size 64

# Batch prediction with output file
mg predict \
    --config config.yaml \
    --input ./images/ \
    --output batch_predictions.yaml \
    --batch-size 16
```

### Custom Configuration and Output

```bash
# Predict with YAML output
mg predict \
    --config config.yaml \
    --input image.jpg \
    --output results.yaml

# Predict with JSON output (default)
mg predict \
    --config config.yaml \
    --input image.jpg \
    --output results.json

# Console output only (no file save)
mg predict --config config.yaml --input image.jpg --top-k 5
```

### Complete Prediction Workflow

```bash
# Train, evaluate, then predict
mg train --config config.yaml
mg evaluate --config config.yaml
mg predict --config config.yaml --input new_image.jpg

# Predict with different models
mg predict \
    --config config.yaml \
    --input test_image.jpg \
    --model-path ./models/model_v1.keras \
    --output v1_predictions.json

mg predict \
    --config config.yaml \
    --input test_image.jpg \
    --model-path ./models/model_v2.keras \
    --output v2_predictions.json
```

## Configuration Requirements

The predict command requires a configuration file that includes:

### Required Configuration Sections

- **Model Configuration**: Model architecture and loading parameters
- **Data Configuration**: Data preprocessing and input handling
- **Prediction Configuration**: Prediction-specific settings

### Example Configuration Structure

```yaml
configuration:
  model:
    model_family: "resnet"
    model_name: "resnet50"
    model_parameters:
      classes: 10
      input_shape: [224, 224, 3]
    
  data:
    preprocessing:
      resize:
        height: 224
        width: 224
      normalization:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    
  prediction:
    batch_size: 32
    top_k: 5
    model_path: "./models/trained_model.keras"
    class_names:
      - "class_0"
      - "class_1"
      - "class_2"
```

## Prediction Process

When you run the predict command, ModelGardener:

1. **Loads Configuration**: Reads prediction configuration and model settings
2. **Loads Model**: Loads the trained model from specified path
3. **Prepares Input**: Preprocesses input images according to configuration
4. **Runs Prediction**: Executes model inference on input data
5. **Processes Results**: Formats predictions with confidence scores
6. **Outputs Results**: Saves or displays prediction results

### Input Types Supported

- **Single Images**: JPG, PNG, and other common image formats
- **Image Directories**: Batch processing of multiple images
- **Preprocessed Data**: Images preprocessed according to training configuration

### Output Format

Prediction results include:

- **Top-K Predictions**: Most likely classes with confidence scores
- **Class Names**: Human-readable class labels
- **Confidence Scores**: Probability values for each prediction
- **Processing Metadata**: Input file names and processing information

### Example Output Structure

**JSON Format:**
```json
{
  "image.jpg": {
    "predictions": [
      {
        "class": "cat",
        "confidence": 0.95,
        "class_index": 1
      },
      {
        "class": "dog", 
        "confidence": 0.03,
        "class_index": 2
      }
    ],
    "top_prediction": "cat",
    "processing_time": 0.12
  }
}
```

**YAML Format:**
```yaml
image.jpg:
  predictions:
    - class: cat
      confidence: 0.95
      class_index: 1
    - class: dog
      confidence: 0.03
      class_index: 2
  top_prediction: cat
  processing_time: 0.12
```

## Performance Optimization

### Batch Size Optimization

- **Small Images**: Use larger batch sizes (64-128)
- **Large Images**: Use smaller batch sizes (8-32)
- **Memory Constraints**: Reduce batch size if out of memory errors occur

### Processing Tips

1. **Use appropriate batch sizes** for your hardware capabilities
2. **Preprocess images consistently** with training preprocessing
3. **Monitor memory usage** during batch prediction
4. **Save results regularly** for large batch jobs
5. **Use GPU acceleration** when available

## Integration with Other Commands

The predict command integrates with the ModelGardener workflow:

```bash
# Complete ML pipeline
mg create project_name
mg train --config config.yaml
mg evaluate --config config.yaml
mg predict --config config.yaml --input new_data.jpg

# Model comparison predictions
mg predict --config config.yaml --model-path ./models/model_v1.keras --input test.jpg --output v1.json
mg predict --config config.yaml --model-path ./models/model_v2.keras --input test.jpg --output v2.json

# Batch prediction workflow
mg predict --config config.yaml --input ./validation_set/ --output validation_predictions.json
mg predict --config config.yaml --input ./test_set/ --output test_predictions.json
```

## Tips and Best Practices

1. **Use consistent preprocessing** between training and prediction
2. **Save prediction results** for analysis and comparison
3. **Monitor prediction confidence** to identify uncertain predictions
4. **Use appropriate top-k values** based on your use case
5. **Batch process large datasets** for efficiency
6. **Validate input formats** match training data format

## Troubleshooting

### Common Issues

- **Model loading errors**: Check model path and format compatibility
- **Input format errors**: Verify image formats and preprocessing
- **Memory errors**: Reduce batch size for large images or datasets
- **Configuration errors**: Use `mg check config.yaml` to validate configuration

### Error Resolution

```bash
# Check configuration before prediction
mg check config.yaml
mg predict --config config.yaml --input image.jpg

# Fix model path issues
mg predict --config config.yaml --input image.jpg --model-path ./models/correct_model.keras

# Reduce batch size for memory issues
mg predict --config config.yaml --input ./large_images/ --batch-size 8
```

## Related Commands

- [`mg train`](train.md) - Train models for prediction
- [`mg evaluate`](evaluate.md) - Evaluate model performance before prediction
- [`mg config`](config.md) - Configure prediction parameters
- [`mg deploy`](deploy.md) - Deploy models for production prediction
- [`mg check`](check.md) - Validate configuration before prediction

## Input Formats

### Single Image

**Supported Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- WebP (.webp)

```bash
# Examples
mg predict --input photo.jpg
mg predict --input scan.png
mg predict --input image.tiff
```

### Directory Input

**Directory Structure:**
```
input_directory/
├── image_001.jpg
├── image_002.png
├── image_003.jpg
├── subfolder/
│   ├── image_004.jpg
│   └── image_005.png
└── image_006.tiff
```

```bash
# Recursive directory processing
mg predict --input ./input_directory/
```

### CSV Input

**CSV Format:**
```csv
image_path,metadata
./images/img1.jpg,"sample_1"
./images/img2.jpg,"sample_2"
./images/img3.png,"sample_3"
```

```bash
# CSV-based prediction
mg predict --input predictions_list.csv --input-format csv
```

## Configuration File Structure

Prediction-specific configuration:

```yaml
prediction:
  model:
    model_path: "./logs/models/best_model.keras"
    weights_path: null
  
  input:
    preprocessing:
      resize:
        height: 224
        width: 224
      normalization:
        rescale: 0.00392156862
      augmentation:
        enable: false
    
    batch_processing:
      batch_size: 32
      parallel_workers: 4
      memory_efficient: true
  
  inference:
    top_k: 5
    confidence_threshold: 0.0
    uncertainty_quantification: false
    ensemble_prediction: false
    
    optimization:
      enable: true
      mixed_precision: true
      graph_optimization: true
      memory_growth: true
  
  output:
    output_dir: "./predictions"
    formats:
      - "json"
      - "csv"
    detailed_output: false
    save_processed_images: false
    
    visualization:
      enable: false
      grad_cam: false
      attention_maps: false
      confidence_visualization: true
      overlay_predictions: false
  
  performance:
    benchmark: false
    timing_analysis: true
    memory_monitoring: true
    resource_tracking: false

metadata:
  class_names:
    - "class_0"
    - "class_1"
    - "class_2"
  
  model_info:
    architecture: "ResNet-50"
    input_shape: [224, 224, 3]
    num_classes: 10
```

## Prediction Pipeline

### Input Processing

1. **Image Loading:**
   - Multi-format image support
   - Batch loading optimization
   - Memory-efficient processing

2. **Preprocessing:**
   - Resize and normalization
   - Same preprocessing as training
   - Optional augmentation for uncertainty

3. **Validation:**
   - Input format verification
   - Shape consistency checking
   - Error handling

### Model Inference

1. **Model Preparation:**
   - Model loading and optimization
   - GPU memory allocation
   - Inference mode setup

2. **Batch Processing:**
   - Optimal batch size selection
   - Parallel processing
   - Memory management

3. **Prediction Generation:**
   - Forward pass execution
   - Probability extraction
   - Top-k selection

### Post-processing

1. **Result Analysis:**
   - Confidence scoring
   - Uncertainty quantification
   - Statistical analysis

2. **Visualization Generation:**
   - Attention map creation
   - Overlay generation
   - Chart plotting

3. **Output Formatting:**
   - Multiple format export
   - Structured data organization
   - Metadata inclusion

## Output Structure

Prediction outputs are organized as follows:

```
predictions/
├── results/
│   ├── predictions.json         # All predictions in JSON
│   ├── predictions.csv          # CSV format predictions
│   ├── summary_statistics.json  # Aggregate statistics
│   ├── confidence_analysis.json # Confidence distribution
│   └── top_k_predictions.json   # Top-k results
├── visualizations/
│   ├── prediction_charts/       # Confidence bar charts
│   ├── grad_cam_maps/          # Grad-CAM attention maps
│   ├── overlay_images/         # Predictions overlaid on images
│   ├── confidence_distribution.png
│   └── prediction_summary.png
├── processed_images/
│   ├── resized_images/         # Preprocessed images
│   ├── augmented_samples/      # Augmentation samples
│   └── failed_processing/      # Processing failures
├── performance/
│   ├── timing_analysis.json     # Inference timing
│   ├── memory_usage.json       # Memory consumption
│   ├── throughput_analysis.json # Processing throughput
│   └── benchmark_results.json   # Performance benchmark
├── uncertainty/
│   ├── uncertainty_scores.json  # Uncertainty quantification
│   ├── confidence_intervals.json
│   ├── ensemble_variance.json
│   └── epistemic_uncertainty.json
└── metadata/
    ├── prediction_config.yaml   # Configuration used
    ├── model_info.json         # Model information
    ├── input_manifest.json     # Input file listing
    └── processing_log.txt       # Processing log
```

## Prediction Output Formats

### JSON Format

```json
{
  "predictions": [
    {
      "image_path": "image_001.jpg",
      "top_predictions": [
        {
          "class_id": 2,
          "class_name": "cat",
          "confidence": 0.89,
          "probability": 0.89
        },
        {
          "class_id": 1,
          "class_name": "dog",
          "confidence": 0.08,
          "probability": 0.08
        }
      ],
      "uncertainty": {
        "epistemic": 0.12,
        "aleatoric": 0.08,
        "total": 0.15
      },
      "metadata": {
        "processing_time": 0.045,
        "image_size": [224, 224],
        "model_version": "v1.0"
      }
    }
  ],
  "summary": {
    "total_images": 100,
    "average_confidence": 0.85,
    "processing_time": 4.5,
    "high_confidence_predictions": 87
  }
}
```

### CSV Format

```csv
image_path,predicted_class,class_name,confidence,top_2_class,top_2_confidence,processing_time,uncertainty
image_001.jpg,2,cat,0.89,dog,0.08,0.045,0.15
image_002.jpg,1,dog,0.95,cat,0.03,0.041,0.08
image_003.jpg,0,bird,0.72,cat,0.18,0.048,0.22
```

### HTML Report Format

Generated HTML report includes:
- Interactive prediction gallery
- Confidence distribution charts
- Performance metrics
- Model information
- Processing statistics

## Advanced Features

### Uncertainty Quantification

**Types of Uncertainty:**
- **Epistemic Uncertainty**: Model uncertainty due to limited training data
- **Aleatoric Uncertainty**: Data uncertainty inherent in the input
- **Total Uncertainty**: Combined uncertainty measure

```bash
# Enable uncertainty quantification
mg predict --input images/ --uncertainty
```

**Uncertainty Methods:**
- Monte Carlo Dropout
- Deep Ensembles
- Bayesian Neural Networks
- Temperature Scaling

### Ensemble Prediction

**Ensemble Types:**
- Multiple model averaging
- Bootstrap aggregating
- Cross-validation ensembles
- Diverse architecture ensembles

```bash
# Ensemble prediction with multiple models
mg predict \
    --input test_set/ \
    --ensemble \
    --model model1.keras,model2.keras,model3.keras
```

### Grad-CAM Visualization

**Attention Map Generation:**
- Class-specific attention maps
- Layer-wise activation visualization
- Guided backpropagation
- Integrated gradients

```bash
# Generate Grad-CAM visualizations
mg predict \
    --input image.jpg \
    --grad-cam \
    --visualize \
    --save-images
```

### Performance Optimization

**Inference Optimization:**
- Graph optimization
- Mixed precision inference
- Memory growth limitation
- Batch size optimization

**Parallel Processing:**
- Multi-threaded image loading
- GPU parallel processing
- Asynchronous inference
- Pipeline optimization

## Performance Benchmarking

### Timing Analysis

**Metrics Measured:**
- Image loading time
- Preprocessing time
- Inference time
- Post-processing time
- Total pipeline time

**Throughput Analysis:**
- Images per second
- Batch processing efficiency
- Memory utilization
- GPU utilization

### Memory Monitoring

**Memory Tracking:**
- Peak memory usage
- Memory efficiency
- Memory leaks detection
- GPU memory utilization

```bash
# Performance benchmarking
mg predict \
    --input benchmark_set/ \
    --benchmark \
    --optimize
```

## Integration with Other Commands

### After Training

```bash
# Train and predict pipeline
mg train --config config.yaml
mg predict \
    --config config.yaml \
    --model ./logs/models/best_model.keras \
    --input ./test_data/
```

### Before Deployment

```bash
# Comprehensive prediction testing
mg predict \
    --input production_test_set/ \
    --benchmark \
    --uncertainty \
    --detailed-output
```

### Batch Processing Workflows

```bash
# Large-scale prediction processing
mg predict \
    --input ./large_dataset/ \
    --batch-size 128 \
    --parallel \
    --output-format csv \
    --optimize
```

## Custom Prediction Functions

### Custom Preprocessing

```python
# In custom_modules/custom_preprocessing.py
def custom_preprocessing(image, **kwargs):
    """Custom preprocessing for prediction."""
    # Custom implementation
    return processed_image
```

### Custom Post-processing

```python
# In custom_modules/custom_postprocessing.py
def confidence_calibration(predictions, **kwargs):
    """Apply confidence calibration."""
    # Custom implementation
    return calibrated_predictions
```

## Best Practices

### Performance Optimization

1. **Batch Size Selection:**
   - Test different batch sizes for optimal throughput
   - Consider memory constraints
   - Balance speed vs. memory usage

2. **Preprocessing Efficiency:**
   - Cache preprocessed images when possible
   - Use parallel data loading
   - Optimize image I/O operations

3. **Memory Management:**
   - Monitor GPU memory usage
   - Use memory growth for TensorFlow
   - Clean up intermediate tensors

### Accuracy Considerations

1. **Uncertainty Assessment:**
   - Use uncertainty quantification for critical decisions
   - Set appropriate confidence thresholds
   - Validate uncertainty calibration

2. **Model Robustness:**
   - Test with diverse input distributions
   - Monitor for distribution shift
   - Use ensemble methods for improved robustness

## Troubleshooting

### Common Issues

**Memory Errors:**
```bash
# Reduce batch size
mg predict --input images/ --batch-size 16

# Limit GPU memory
mg predict --input images/ --gpu-memory-limit 4096
```

**Slow Processing:**
```bash
# Enable optimization
mg predict --input images/ --optimize --parallel

# Increase batch size
mg predict --input images/ --batch-size 64
```

**Input Format Issues:**
```bash
# Specify input format explicitly
mg predict --input data.csv --input-format csv

# Check supported formats
```

### Debugging Options

```bash
# Verbose output
mg predict --input images/ --detailed-output

# Save intermediate results
mg predict --input images/ --save-images
```

## See Also

- [Training Command](train.md)
- [Evaluation Command](evaluate.md)
- [Deployment Command](deploy.md)
- [Uncertainty Quantification Tutorial](../tutorials/uncertainty.md)
- [Performance Optimization Guide](../tutorials/optimization.md)
