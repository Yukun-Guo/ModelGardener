# `predict` Command

Generate predictions on new data using trained models with intelligent auto-discovery and comprehensive output management.

## Synopsis

```bash
mg predict [OPTIONS]
```

## Description

The `predict` command enables prediction on new data using trained models with enhanced auto-discovery capabilities. It supports:

- **Auto-Discovery**: Automatically finds config files, models, and input data
- **Flexible Input**: Single image, batch processing, and directory scanning
- **Smart Output**: Automatic report generation with JSON and CSV formats
- **Performance Optimization**: Configurable batch sizes and prediction parameters
- **Comprehensive Results**: Top-k predictions with confidence scores and metadata

## Auto-Discovery Features

### 🔍 Intelligent File Discovery
- **Config Discovery**: Automatically locates `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory
- **Input Discovery**: Searches for common test directories and image files
  - Test directories: `test/`, `test_data/`, `test_images/`, `val/`, `data/test/`, `data/val/`
  - Image formats: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

### 📊 Enhanced Output Management
- **Automatic Reports**: Creates timestamped prediction reports in `predictions/` folder
- **Multiple Formats**: JSON reports with optional CSV summaries for easy analysis
- **Flexible Saving**: Control report generation with `--no-save` option
- **Custom Output**: Specify exact output paths when needed

## Options

### Core Options (All Optional with Auto-Discovery)

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Configuration file path | Auto-discovered `config.yaml` |
| `--input` | `-i` | `str` | Input image file or directory | Auto-discovered test data |
| `--model-path` | `-m` | `str` | Path to trained model | Auto-discovered latest model |

### Output Control Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--output` | `-o` | `str` | Output file for results (JSON) | Auto-generated in predictions/ |
| `--no-save` | | `flag` | Do not save prediction results | False |

### Prediction Parameters

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--top-k` | `int` | Number of top predictions to show | 5 |
| `--batch-size` | `int` | Batch size for processing | 32 |

### Visualization Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--visualize` | `flag` | Generate prediction visualizations | False |
| `--grad-cam` | `flag` | Generate Grad-CAM attention maps | False |
| `--overlay` | `flag` | Overlay predictions on images | False |
| `--confidence-bars` | `flag` | Show confidence score bars | True |

## Usage Examples

### Auto-Discovery Prediction (Recommended)

```bash
# Full auto-discovery - finds config, model, and test data automatically
mg predict

# Auto-discovery with custom top-k
mg predict --top-k 3

# Auto-discovery without saving reports
mg predict --no-save

# Auto-discovery with custom batch size
mg predict --batch-size 64
```

### Selective Auto-Discovery

```bash
# Auto-discover config and model, specify custom input
mg predict -i ./my_test_images/

# Auto-discover config, specify model and input
mg predict -m ./models/custom_model.keras -i image.jpg

# Specify config, auto-discover model and input
mg predict -c ./configs/custom_config.yaml

# Auto-discovery with custom output file
mg predict -o my_predictions.json --top-k 10
```

### Single Image Prediction

```bash
# Auto-discovery with single image
mg predict -i image.jpg

# Explicit single image prediction
mg predict -c config.yaml -i image.jpg -m ./models/my_model.keras

# Single image with custom output and top-k
mg predict -i image.jpg -o single_result.json --top-k 3
```

### Batch Directory Prediction

```bash
# Auto-discovery batch prediction
mg predict -i ./test_images/

# Batch prediction with custom settings
mg predict -i ./large_dataset/ --batch-size 64 -o batch_results.json

# Explicit batch prediction
mg predict -c config.yaml -i ./images/ -m ./models/model.keras --batch-size 16
```

### Output Control Examples

```bash
# Default auto-saving to predictions/ folder
mg predict -i ./test_images/

# Custom output file with auto-discovery
mg predict -i ./test_images/ -o my_custom_results.json

# Quick prediction without saving
mg predict -i image.jpg --no-save --top-k 5

# Console output only with custom parameters
mg predict -i ./images/ --no-save --batch-size 32
```

### Complete Prediction Workflow

```bash
# Train, evaluate, then predict with auto-discovery
mg train
mg evaluate
mg predict

# Multi-model comparison with explicit paths
mg predict -m ./models/model_v1.keras -i test_image.jpg -o v1_predictions.json
mg predict -m ./models/model_v2.keras -i test_image.jpg -o v2_predictions.json

# Batch processing pipeline
mg predict -i ./validation_set/ -o validation_predictions.json
mg predict -i ./test_set/ -o test_predictions.json
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
