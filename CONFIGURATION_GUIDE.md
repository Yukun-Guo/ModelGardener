# TensorFlow Models Configuration Guide

This guide explains the comprehensive configuration system integrated into the Model Gardener GUI, which provides access to all TensorFlow Models official parameters organized for optimal user experience.

## Configuration Structure

The configuration is organized into **Basic** and **Advanced** sections to improve usability:

### Basic Configuration

These are the most commonly used parameters that most users will need to adjust:

#### Data Section
- `train_dir` - Path to training data directory (with directory/file browser)
- `val_dir` - Path to validation data directory (with directory/file browser)
- `image_size` - Input image dimensions [width, height]
- `batch_size` - Training batch size
- `num_classes` - Number of classification classes
- `shuffle` - Whether to shuffle training data

#### Model Section
- `backbone_type` - Model architecture (resnet, efficientnet, mobilenet, vit, densenet)
- `model_id` - Specific model variant (e.g., 50 for ResNet-50)
- `dropout_rate` - Dropout rate for regularization
- `activation` - Activation function (relu, swish, gelu, leaky_relu, tanh)

#### Training Section
- `epochs` - Number of training epochs
- `learning_rate_type` - Learning rate schedule type
- `initial_learning_rate` - Starting learning rate
- `momentum` - SGD momentum parameter
- `weight_decay` - L2 regularization strength
- `label_smoothing` - Label smoothing factor

#### Runtime Section
- `model_dir` - Model checkpoint save directory (with directory browser)
- `distribution_strategy` - Training distribution strategy
- `mixed_precision` - Mixed precision training type
- `num_gpus` - Number of GPUs to use

### Advanced Configuration

Expert-level parameters for fine-tuning and specialized use cases:

#### Model Advanced Section
- Architecture-specific parameters (depth multiplier, SE ratio, etc.)
- Normalization settings (momentum, epsilon, sync batch norm)
- Initialization and output configurations

#### Data Advanced Section
- TensorFlow data pipeline parameters
- Caching, sharding, and performance settings
- Data format and field specifications

#### Augmentation Section
- Data augmentation parameters
- Crop settings, color jitter, RandAugment
- Resize methods and augmentation policies

#### Training Advanced Section
- Training loop configurations
- Checkpoint and validation intervals
- Performance and monitoring settings

#### Evaluation Section
- Evaluation metrics configuration
- Best checkpoint selection criteria

#### Runtime Advanced Section
- XLA compilation settings
- GPU thread configurations
- TPU-specific parameters

## Parameter Types

The configuration system uses specialized parameter types:

1. **Directory Parameters** - Browse for directories and files
   - `directory` type: Both directory and file browsing buttons
   - `directory_only` type: Directory browsing only

2. **Choice Parameters** - Dropdown selections for predefined options
   - Model types, optimizers, activation functions, etc.

3. **Numeric Parameters** - With appropriate ranges and step sizes
   - Integer parameters with defined limits
   - Float parameters with precision control

4. **Boolean Parameters** - Toggle switches for on/off settings

5. **Special Groups** - Composite parameters like image_size with width/height

## Integration with TensorFlow Models

The configuration maps directly to TensorFlow Models official experiment configurations:

- All parameters correspond to actual TensorFlow Models config fields
- Proper type validation and range checking
- Seamless integration with `exp_factory.get_exp_config()`

## Usage Tips

1. **Start with Basic**: Configure basic parameters first for standard use cases
2. **Advanced Tuning**: Use advanced sections for performance optimization
3. **Save/Load**: Use toolbar buttons to save configurations as JSON/YAML
4. **Validation**: Parameters are validated with appropriate ranges and types
5. **Directory Browsing**: Use the built-in directory browsers for path selection

## Configuration Persistence

Configurations can be:
- Saved as JSON or YAML files
- Loaded from existing configuration files
- Applied to TensorFlow Models experiments directly

The system maintains backward compatibility while providing comprehensive access to all TensorFlow Models configuration options.
