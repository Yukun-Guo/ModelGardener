# `models` Command

Manage and inspect available model architectures, view model details, and perform model-related operations within the ModelGardener ecosystem.

## Synopsis

```bash
modelgardener_cli.py models [OPTIONS]
```

## Description

The `models` command provides comprehensive model management capabilities including:

- List available model architectures and families
- Inspect model details and specifications
- Download and manage pre-trained models
- Model architecture visualization and analysis
- Model performance comparison and benchmarking
- Custom model registration and management
- Model version control and tracking

## Options

### Model Discovery

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--list` | `-l` | `flag` | List all available models | False |
| `--family` | `-f` | `str` | Filter by model family | None |
| `--search` | `-s` | `str` | Search models by name or description | None |
| `--show-details` | `-d` | `str` | Show detailed information for specific model | None |

### Model Information

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--info` | `str` | Display detailed model information | None |
| `--architecture` | `flag` | Show model architecture diagram | False |
| `--parameters` | `flag` | Show parameter count and details | False |
| `--performance` | `flag` | Show performance benchmarks | False |

### Model Management

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--download` | `str` | Download pre-trained model weights | None |
| `--register` | `str` | Register custom model | None |
| `--validate` | `str` | Validate model file or definition | None |
| `--export-list` | `str` | Export model list to file | None |

### Filtering Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--task` | `str` | Filter by task type (classification, detection, etc.) | None |
| `--input-size` | `str` | Filter by input size requirements | None |
| `--memory-limit` | `str` | Filter by memory requirements | None |
| `--accuracy-threshold` | `float` | Filter by minimum accuracy | None |

## Usage Examples

### Basic Model Discovery

```bash
# List all available models
modelgardener_cli.py models --list

# List models by family
modelgardener_cli.py models --list --family resnet

# Search for specific models
modelgardener_cli.py models --search "efficient"

# Show detailed information
modelgardener_cli.py models --show-details ResNet-50
```

### Model Analysis

```bash
# Show model architecture
modelgardener_cli.py models --info ResNet-50 --architecture

# Show parameter details
modelgardener_cli.py models --info EfficientNet-B0 --parameters

# Show performance benchmarks
modelgardener_cli.py models --info MobileNet-V2 --performance
```

### Model Filtering

```bash
# Filter by task type
modelgardener_cli.py models --list --task classification

# Filter by memory requirements
modelgardener_cli.py models --list --memory-limit 1GB

# Filter by accuracy threshold
modelgardener_cli.py models --list --accuracy-threshold 0.8

# Combined filtering
modelgardener_cli.py models \
    --list \
    --family efficientnet \
    --task classification \
    --memory-limit 2GB
```

### Model Management

```bash
# Download pre-trained weights
modelgardener_cli.py models --download ResNet-50

# Register custom model
modelgardener_cli.py models --register ./custom_models/my_model.py

# Validate model definition
modelgardener_cli.py models --validate ./models/custom_architecture.py

# Export model list
modelgardener_cli.py models --export-list ./available_models.json
```

## Available Model Families

### ResNet Family

**Models Available:**
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- ResNet-50-V2, ResNet-101-V2, ResNet-152-V2
- ResNeXt-50, ResNeXt-101

**Characteristics:**
- Deep residual networks with skip connections
- Excellent for image classification tasks
- Good balance of accuracy and computational efficiency
- Pre-trained on ImageNet available

```bash
# List ResNet models
modelgardener_cli.py models --list --family resnet

# ResNet-50 details
modelgardener_cli.py models --info ResNet-50 --parameters --performance
```

### EfficientNet Family

**Models Available:**
- EfficientNet-B0 through EfficientNet-B7
- EfficientNet-V2-S, EfficientNet-V2-M, EfficientNet-V2-L

**Characteristics:**
- State-of-the-art efficiency and accuracy
- Compound scaling method
- Optimized for mobile and edge deployment
- Excellent accuracy-to-parameters ratio

```bash
# List EfficientNet models
modelgardener_cli.py models --list --family efficientnet

# Compare EfficientNet variants
modelgardener_cli.py models --info EfficientNet-B0 --performance
modelgardener_cli.py models --info EfficientNet-B7 --performance
```

### MobileNet Family

**Models Available:**
- MobileNet-V1, MobileNet-V2, MobileNet-V3-Small, MobileNet-V3-Large

**Characteristics:**
- Optimized for mobile and edge devices
- Depthwise separable convolutions
- Very low parameter count and memory usage
- Fast inference on CPU

```bash
# List MobileNet models
modelgardener_cli.py models --list --family mobilenet

# MobileNet memory analysis
modelgardener_cli.py models --info MobileNet-V2 --parameters
```

### Other Model Families

**DenseNet:**
- DenseNet-121, DenseNet-169, DenseNet-201

**Inception:**
- Inception-V3, Inception-ResNet-V2

**Xception:**
- Xception

**VGG:**
- VGG-16, VGG-19

## Model Information Display

### Basic Information

```
Model: ResNet-50
Family: ResNet
Task: Image Classification
Input Shape: (224, 224, 3)
Output Classes: 1000 (ImageNet)
Parameters: 25.6M
Model Size: 102.4 MB
FLOPs: 4.1B

Description:
50-layer residual network with skip connections, designed to address
the vanishing gradient problem in deep networks. Excellent balance
of accuracy and computational efficiency.

Pretrained Weights Available:
- ImageNet (Top-1 Accuracy: 76.1%, Top-5 Accuracy: 92.9%)
- Custom weights available for fine-tuning

Recommended Use Cases:
- General image classification
- Transfer learning base model
- Feature extraction backbone
```

### Detailed Parameters

```
Parameter Breakdown:
├── Input Layer: (224, 224, 3)
├── Conv2D Layers: 49 layers
│   ├── Total Parameters: 23.5M
│   ├── Trainable Parameters: 23.5M
│   └── Memory Usage: 94.2 MB
├── Batch Normalization: 49 layers
│   ├── Total Parameters: 131K
│   └── Memory Usage: 524 KB
├── Global Average Pooling: 1 layer
├── Dense Output: 1 layer
│   ├── Total Parameters: 2.0M
│   └── Memory Usage: 8.0 MB
└── Total: 25.6M parameters, 102.4 MB

Layer Distribution:
- Convolutional: 94.2% of parameters
- Normalization: 0.5% of parameters
- Dense: 7.8% of parameters
- Other: 0.5% of parameters
```

### Performance Benchmarks

```
Performance Metrics:
├── Accuracy Metrics:
│   ├── ImageNet Top-1: 76.1%
│   ├── ImageNet Top-5: 92.9%
│   ├── CIFAR-10: 95.3%
│   └── CIFAR-100: 78.4%
├── Inference Speed:
│   ├── CPU (Intel i7): 45ms
│   ├── GPU (V100): 3.2ms
│   ├── Mobile (Snapdragon 855): 120ms
│   └── Edge (Raspberry Pi 4): 850ms
├── Memory Usage:
│   ├── Training: 2.1 GB
│   ├── Inference: 512 MB
│   └── Mobile: 256 MB
└── Energy Consumption:
    ├── GPU: 15.3 mJ/inference
    ├── CPU: 8.7 mJ/inference
    └── Mobile: 12.1 mJ/inference
```

## Custom Model Registration

### Model Definition Format

```python
# custom_models/my_custom_model.py
import tensorflow as tf
from tensorflow.keras import layers

def create_my_custom_model(input_shape, num_classes, **kwargs):
    """
    Custom CNN architecture with attention mechanism.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        **kwargs: Additional parameters
    
    Returns:
        tf.keras.Model: Compiled model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Custom architecture implementation
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Attention mechanism
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(64, activation='relu')(attention)
    attention = layers.Dense(x.shape[-1], activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
    
    x = layers.Multiply()([x, attention])
    
    # Output layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='my_custom_model')
    return model

# Model metadata
MODEL_INFO = {
    "name": "My Custom Model",
    "family": "custom",
    "task": "classification",
    "description": "Custom CNN with attention mechanism",
    "input_shape": (224, 224, 3),
    "parameters": "estimated_2.5M",
    "pretrained_weights": False,
    "author": "Your Name",
    "version": "1.0.0",
    "tags": ["attention", "custom", "efficient"]
}
```

### Registration Process

```bash
# Register custom model
modelgardener_cli.py models \
    --register ./custom_models/my_custom_model.py \
    --validate

# Verify registration
modelgardener_cli.py models --list --family custom

# Test custom model
modelgardener_cli.py models --info "My Custom Model" --architecture
```

## Model Comparison

### Performance Comparison

```bash
# Compare multiple models
modelgardener_cli.py models \
    --compare ResNet-50,EfficientNet-B0,MobileNet-V2 \
    --output comparison_report.json
```

**Comparison Output:**
```
Model Comparison Report:
========================

Accuracy Comparison (ImageNet Top-1):
├── ResNet-50: 76.1%
├── EfficientNet-B0: 77.3%
└── MobileNet-V2: 71.8%

Parameter Count:
├── ResNet-50: 25.6M
├── EfficientNet-B0: 5.3M
└── MobileNet-V2: 3.5M

Inference Speed (GPU):
├── ResNet-50: 3.2ms
├── EfficientNet-B0: 2.8ms
└── MobileNet-V2: 1.9ms

Memory Usage (Inference):
├── ResNet-50: 512 MB
├── EfficientNet-B0: 256 MB
└── MobileNet-V2: 128 MB

Recommendation:
For high accuracy: EfficientNet-B0
For mobile deployment: MobileNet-V2
For balanced performance: ResNet-50
```

### Model Selection Guide

```bash
# Get model recommendations
modelgardener_cli.py models \
    --recommend \
    --task classification \
    --target-accuracy 0.8 \
    --memory-limit 1GB \
    --inference-speed fast
```

## Model Validation

### Validation Checks

```bash
# Validate model definition
modelgardener_cli.py models --validate ./models/custom_model.py
```

**Validation Report:**
```
Model Validation Report:
========================

✓ Syntax Check: PASSED
✓ Function Signature: PASSED
✓ Return Type: PASSED
✓ Model Compilation: PASSED
✓ Input Shape Compatibility: PASSED
✓ Output Shape Validation: PASSED
✓ Parameter Count Estimation: PASSED
✓ Memory Usage Estimation: PASSED

Warnings:
⚠ Large parameter count (>50M) may cause memory issues
⚠ No batch normalization layers detected

Suggestions:
💡 Consider adding batch normalization for training stability
💡 Add dropout layers to prevent overfitting
💡 Use depthwise separable convolutions for efficiency
```

## Integration with Training

### Model Selection in Configuration

```yaml
model:
  model_selection:
    selected_model_family: "efficientnet"
    selected_model_name: "EfficientNet-B3"
  
  model_parameters:
    classes: 10
    input_shape:
      height: 300
      width: 300
      channels: 3
    weights: "imagenet"
    include_top: false
```

### Custom Model Integration

```yaml
model:
  model_selection:
    selected_model_family: "custom"
    selected_model_name: "My Custom Model"
  
  model_parameters:
    classes: 100
    input_shape:
      height: 224
      width: 224
      channels: 3

metadata:
  custom_functions:
    models:
      - "my_custom_model"
```

## Best Practices

### Model Selection

1. **Task Alignment:**
   - Choose models designed for your specific task
   - Consider input size requirements
   - Evaluate output format compatibility

2. **Resource Constraints:**
   - Match model size to available memory
   - Consider inference speed requirements
   - Evaluate deployment platform limitations

3. **Accuracy Requirements:**
   - Balance accuracy vs. efficiency
   - Consider fine-tuning capabilities
   - Evaluate transfer learning potential

### Custom Model Development

1. **Architecture Design:**
   - Follow established design patterns
   - Include proper normalization layers
   - Implement appropriate regularization

2. **Code Quality:**
   - Use clear function signatures
   - Include comprehensive documentation
   - Follow TensorFlow best practices

3. **Testing:**
   - Validate model compilation
   - Test with sample data
   - Benchmark performance

## Troubleshooting

### Common Issues

**Model Not Found:**
```bash
# Check available models
modelgardener_cli.py models --list --search "model_name"

# Update model list
modelgardener_cli.py models --refresh
```

**Custom Model Registration Failed:**
```bash
# Validate model definition
modelgardener_cli.py models --validate ./path/to/model.py

# Check error details
modelgardener_cli.py models --register ./path/to/model.py --verbose
```

**Performance Issues:**
```bash
# Check model requirements
modelgardener_cli.py models --info ModelName --parameters

# Compare alternatives
modelgardener_cli.py models --recommend --memory-limit 1GB
```

## See Also

- [Create Command](create.md)
- [Training Command](train.md)
- [Configuration Command](config.md)
- [Model Architecture Guide](../tutorials/model-architectures.md)
- [Custom Model Development](../tutorials/custom-models.md)
- [Transfer Learning Tutorial](../tutorials/transfer-learning.md)
