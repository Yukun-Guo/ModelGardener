# `config` Command

Manage and validate ModelGardener configuration files with comprehensive validation, optimization suggestions, and interactive editing capabilities.

## Synopsis

```bash
modelgardener_cli.py config [OPTIONS]
```

## Description

The `config` command provides comprehensive configuration management including:

- Configuration file validation and syntax checking
- Interactive configuration editing and generation
- Configuration optimization and best practice suggestions
- Template generation for different use cases
- Configuration migration and version management
- Batch validation for multiple configurations
- Schema documentation and help

## Options

### Basic Operations

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--file` | `-f` | `str` | Configuration file path | `config.yaml` |
| `--validate` | `-v` | `flag` | Validate configuration file | False |
| `--edit` | `-e` | `flag` | Interactive configuration editing | False |
| `--show` | `-s` | `flag` | Display current configuration | False |

### Configuration Generation

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--template` | `str` | Generate template (basic, advanced, custom) | None |
| `--use-case` | `str` | Use case template (classification, detection, etc.) | None |
| `--interactive` | `flag` | Interactive configuration builder | False |
| `--from-model` | `str` | Generate config from existing model | None |

### Validation Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--strict` | `flag` | Enable strict validation mode | False |
| `--check-paths` | `flag` | Validate file and directory paths | True |
| `--check-dependencies` | `flag` | Check for required dependencies | True |
| `--suggest-optimizations` | `flag` | Provide optimization suggestions | True |

### Output Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output` | `str` | Output file for generated config | None |
| `--format` | `str` | Output format (yaml, json) | `yaml` |
| `--backup` | `flag` | Create backup before editing | True |
| `--dry-run` | `flag` | Show changes without applying | False |

## Usage Examples

### Configuration Validation

```bash
# Validate default configuration
modelgardener_cli.py config --validate

# Validate specific configuration file
modelgardener_cli.py config --file custom_config.yaml --validate

# Strict validation with path checking
modelgardener_cli.py config \
    --file config.yaml \
    --validate \
    --strict \
    --check-paths \
    --suggest-optimizations
```

### Configuration Generation

```bash
# Generate basic configuration template
modelgardener_cli.py config --template basic --output basic_config.yaml

# Generate advanced configuration
modelgardener_cli.py config --template advanced --output advanced_config.yaml

# Interactive configuration builder
modelgardener_cli.py config --interactive --output my_config.yaml

# Use case specific template
modelgardener_cli.py config \
    --use-case image_classification \
    --output classification_config.yaml
```

### Configuration Editing

```bash
# Interactive configuration editing
modelgardener_cli.py config --file config.yaml --edit

# Show current configuration
modelgardener_cli.py config --file config.yaml --show

# Generate config from existing model
modelgardener_cli.py config \
    --from-model ./models/trained_model.keras \
    --output inferred_config.yaml
```

### Configuration Migration

```bash
# Migrate old configuration format
modelgardener_cli.py config \
    --file old_config.yaml \
    --migrate \
    --output new_config.yaml

# Batch validation
modelgardener_cli.py config \
    --validate-batch ./configs/ \
    --output validation_report.json
```

## Configuration Templates

### Basic Template

```yaml
configuration:
  data:
    train_dir: "./data/train"
    val_dir: "./data/val"
    data_loader:
      name: "ImageDataGenerator"
      parameters:
        batch_size: 32
        shuffle: true
        validation_split: 0.2

  model:
    model_selection:
      selected_model_family: "resnet"
      selected_model_name: "ResNet-50"
    model_parameters:
      classes: 10
      input_shape:
        height: 224
        width: 224
        channels: 3

  training:
    epochs: 100
    initial_learning_rate: 0.001

  runtime:
    model_dir: "./logs"
    use_gpu: true
```

### Advanced Template

```yaml
configuration:
  data:
    train_dir: "./data/train"
    val_dir: "./data/val"
    test_dir: "./data/test"
    
    data_loader:
      name: "ImageDataGenerator"
      parameters:
        batch_size: 32
        shuffle: true
        validation_split: 0.2
        preprocessing_function: null
        
    preprocessing:
      resize:
        height: 224
        width: 224
      normalization:
        rescale: 0.00392156862
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      augmentation:
        rotation_range: 20
        width_shift_range: 0.2
        height_shift_range: 0.2
        horizontal_flip: true
        vertical_flip: false
        zoom_range: 0.2
        shear_range: 0.2

  model:
    model_selection:
      selected_model_family: "efficientnet"
      selected_model_name: "EfficientNet-B3"
    
    model_parameters:
      classes: 100
      input_shape:
        height: 300
        width: 300
        channels: 3
      weights: "imagenet"
      include_top: false
      pooling: "avg"
      dropout_rate: 0.3
    
    compilation:
      optimizer:
        name: "Adam"
        parameters:
          learning_rate: 0.001
          beta_1: 0.9
          beta_2: 0.999
          epsilon: 1e-07
          amsgrad: false
      
      loss:
        name: "categorical_crossentropy"
        parameters:
          label_smoothing: 0.1
      
      metrics:
        - "accuracy"
        - "top_k_categorical_accuracy"
        - "precision"
        - "recall"

  training:
    epochs: 200
    initial_learning_rate: 0.001
    
    callbacks:
      early_stopping:
        monitor: "val_loss"
        patience: 15
        restore_best_weights: true
        min_delta: 0.001
      
      reduce_lr_on_plateau:
        monitor: "val_loss"
        factor: 0.2
        patience: 8
        min_lr: 1e-7
        cooldown: 5
      
      model_checkpoint:
        monitor: "val_accuracy"
        save_best_only: true
        save_weights_only: false
        mode: "max"
        
      csv_logger:
        filename: "training_log.csv"
        separator: ","
        append: false

  runtime:
    model_dir: "./logs"
    use_gpu: true
    mixed_precision: true
    distribution_strategy: "mirrored"
    random_seed: 42

metadata:
  project_name: "Advanced Image Classification"
  version: "1.0.0"
  description: "Advanced configuration for image classification tasks"
  author: "ModelGardener User"
  
  custom_functions:
    models: []
    data_loaders: []
    loss_functions: []
    metrics: []
    callbacks: []
    optimizers: []
    augmentations: []
    preprocessing: []
    training_loops: []
```

## Interactive Configuration Builder

When using `--interactive` mode, you'll be guided through:

### Project Setup
- Project name and description
- Use case selection
- Target platform and requirements

### Data Configuration
- Data source specification
- Preprocessing parameters
- Augmentation settings
- Validation strategy

### Model Configuration
- Architecture family selection
- Model-specific parameters
- Transfer learning settings
- Custom model integration

### Training Configuration
- Optimization settings
- Learning rate scheduling
- Callback configuration
- Resource allocation

### Advanced Settings
- Custom function integration
- Performance optimization
- Monitoring and logging
- Deployment preparation

## Configuration Validation

### Validation Levels

**Basic Validation:**
- YAML syntax checking
- Schema compliance
- Required field verification
- Data type validation

**Advanced Validation:**
- Path existence checking
- Dependency verification
- Resource requirement analysis
- Performance impact assessment

**Custom Validation:**
- Domain-specific rules
- Best practice enforcement
- Security compliance
- Platform compatibility

### Validation Output

```json
{
  "validation_status": "passed",
  "errors": [],
  "warnings": [
    {
      "type": "performance",
      "message": "Batch size 32 may be suboptimal for your GPU",
      "suggestion": "Consider increasing to 64 or 128",
      "severity": "medium"
    }
  ],
  "suggestions": [
    {
      "type": "optimization",
      "message": "Enable mixed precision for faster training",
      "path": "runtime.mixed_precision",
      "recommended_value": true
    }
  ],
  "metrics": {
    "estimated_memory_usage": "8.5 GB",
    "estimated_training_time": "2.5 hours",
    "gpu_utilization": "85%"
  }
}
```

## Configuration Schema

### Data Section Schema

```yaml
data:
  train_dir: str              # Required: Training data directory
  val_dir: str               # Required: Validation data directory
  test_dir: str              # Optional: Test data directory
  
  data_loader:
    name: str                # Required: Data loader name
    parameters:              # Required: Data loader parameters
      batch_size: int        # Required: Batch size (1-512)
      shuffle: bool          # Optional: Shuffle data (default: true)
      validation_split: float # Optional: Validation split (0.0-1.0)
      
  preprocessing:             # Optional: Preprocessing configuration
    resize:
      height: int            # Required if resize: Image height
      width: int             # Required if resize: Image width
    normalization:
      rescale: float         # Optional: Rescaling factor
      mean: [float]          # Optional: Channel means
      std: [float]           # Optional: Channel standard deviations
    augmentation:            # Optional: Data augmentation
      rotation_range: float  # Optional: Rotation range (0-180)
      width_shift_range: float # Optional: Width shift (0.0-1.0)
      height_shift_range: float # Optional: Height shift (0.0-1.0)
      horizontal_flip: bool  # Optional: Horizontal flip
      vertical_flip: bool    # Optional: Vertical flip
      zoom_range: float      # Optional: Zoom range (0.0-1.0)
      shear_range: float     # Optional: Shear range (0.0-1.0)
```

### Model Section Schema

```yaml
model:
  model_selection:
    selected_model_family: str # Required: Model family
    selected_model_name: str   # Required: Specific model name
  
  model_parameters:
    classes: int               # Required: Number of classes (1-10000)
    input_shape:               # Required: Input shape specification
      height: int              # Required: Input height
      width: int               # Required: Input width
      channels: int            # Required: Input channels (1, 3, 4)
    weights: str               # Optional: Pretrained weights
    include_top: bool          # Optional: Include top layer
    pooling: str               # Optional: Pooling type
    dropout_rate: float        # Optional: Dropout rate (0.0-0.9)
  
  compilation:
    optimizer:                 # Required: Optimizer configuration
      name: str                # Required: Optimizer name
      parameters:              # Optional: Optimizer parameters
        learning_rate: float   # Optional: Learning rate (1e-6 - 1.0)
        beta_1: float          # Optional: Adam beta_1 (0.0-1.0)
        beta_2: float          # Optional: Adam beta_2 (0.0-1.0)
    
    loss:                      # Required: Loss function
      name: str                # Required: Loss function name
      parameters:              # Optional: Loss parameters
        label_smoothing: float # Optional: Label smoothing (0.0-0.3)
    
    metrics: [str]             # Required: List of metrics
```

### Training Section Schema

```yaml
training:
  epochs: int                  # Required: Number of epochs (1-10000)
  initial_learning_rate: float # Required: Initial LR (1e-6 - 1.0)
  
  callbacks:                   # Optional: Training callbacks
    early_stopping:            # Optional: Early stopping
      monitor: str             # Required: Metric to monitor
      patience: int            # Required: Patience epochs (1-100)
      restore_best_weights: bool # Optional: Restore best weights
      min_delta: float         # Optional: Minimum change (1e-6 - 1.0)
    
    reduce_lr_on_plateau:      # Optional: Learning rate reduction
      monitor: str             # Required: Metric to monitor
      factor: float            # Required: Reduction factor (0.1-0.9)
      patience: int            # Required: Patience epochs (1-50)
      min_lr: float            # Optional: Minimum LR (1e-10 - 1e-3)
    
    model_checkpoint:          # Optional: Model checkpointing
      monitor: str             # Required: Metric to monitor
      save_best_only: bool     # Optional: Save best only
      save_weights_only: bool  # Optional: Save weights only
      mode: str                # Optional: Optimization mode
```

## Configuration Optimization

### Performance Optimization

**GPU Optimization:**
```yaml
runtime:
  use_gpu: true
  mixed_precision: true
  distribution_strategy: "mirrored"
  memory_growth: true
```

**Memory Optimization:**
```yaml
data:
  data_loader:
    parameters:
      batch_size: 32  # Optimized for memory
      
model:
  model_parameters:
    dropout_rate: 0.3  # Prevent overfitting
```

**Training Optimization:**
```yaml
training:
  callbacks:
    reduce_lr_on_plateau:
      factor: 0.2
      patience: 5
      
    early_stopping:
      patience: 10
      restore_best_weights: true
```

### Best Practice Suggestions

**Data Configuration:**
- Use appropriate batch sizes for your hardware
- Enable data augmentation for small datasets
- Set proper validation split ratios
- Use preprocessing normalization

**Model Configuration:**
- Choose appropriate architectures for your data
- Use transfer learning when possible
- Set reasonable dropout rates
- Configure proper input shapes

**Training Configuration:**
- Use early stopping to prevent overfitting
- Implement learning rate scheduling
- Save model checkpoints regularly
- Monitor multiple metrics

## Custom Configuration Extensions

### Custom Function Integration

```yaml
metadata:
  custom_functions:
    models:
      - "custom_resnet_variant"
      - "attention_enhanced_cnn"
    
    data_loaders:
      - "custom_image_loader"
      - "multi_modal_loader"
    
    loss_functions:
      - "focal_loss"
      - "weighted_categorical_crossentropy"
    
    metrics:
      - "f1_score"
      - "precision_recall_auc"
    
    callbacks:
      - "custom_lr_scheduler"
      - "model_complexity_monitor"
```

### Environment-Specific Configurations

```yaml
environments:
  development:
    training:
      epochs: 10
    runtime:
      model_dir: "./dev_logs"
  
  production:
    training:
      epochs: 200
    runtime:
      model_dir: "./prod_models"
      mixed_precision: true
```

## Configuration Migration

### Version Migration

```bash
# Migrate from v1.0 to v2.0 format
modelgardener_cli.py config \
    --file old_config.yaml \
    --migrate \
    --from-version 1.0 \
    --to-version 2.0 \
    --output migrated_config.yaml
```

### Batch Operations

```bash
# Validate multiple configurations
modelgardener_cli.py config \
    --validate-batch ./project_configs/ \
    --output batch_validation.json

# Migrate multiple configurations
modelgardener_cli.py config \
    --migrate-batch ./old_configs/ \
    --output-dir ./new_configs/
```

## Integration with Other Commands

### Configuration in Training

```bash
# Train with validated configuration
modelgardener_cli.py config --file config.yaml --validate
modelgardener_cli.py train --config config.yaml
```

### Configuration Templates for Projects

```bash
# Create project with specific configuration
modelgardener_cli.py config --template advanced --output advanced.yaml
modelgardener_cli.py create my_project --config advanced.yaml
```

## Best Practices

### Configuration Management

1. **Version Control:**
   - Store configurations in version control
   - Use meaningful commit messages
   - Tag configuration versions
   - Document configuration changes

2. **Validation:**
   - Always validate before training
   - Use strict validation for production
   - Check path existence
   - Verify dependencies

3. **Organization:**
   - Use descriptive configuration names
   - Organize by use case or project
   - Document configuration purposes
   - Maintain configuration templates

### Performance Considerations

1. **Resource Optimization:**
   - Match batch size to hardware
   - Enable appropriate acceleration
   - Use memory-efficient settings
   - Monitor resource usage

2. **Training Efficiency:**
   - Use early stopping
   - Implement learning rate scheduling
   - Enable mixed precision
   - Use appropriate callbacks

## Troubleshooting

### Common Configuration Issues

**YAML Syntax Errors:**
```bash
# Validate YAML syntax
modelgardener_cli.py config --file config.yaml --validate --strict
```

**Path Issues:**
```bash
# Check all paths
modelgardener_cli.py config --file config.yaml --check-paths
```

**Dependency Issues:**
```bash
# Verify dependencies
modelgardener_cli.py config --file config.yaml --check-dependencies
```

### Debugging Configuration

```bash
# Show detailed configuration
modelgardener_cli.py config --file config.yaml --show --verbose

# Dry run validation
modelgardener_cli.py config --file config.yaml --validate --dry-run
```

## See Also

- [Create Command](create.md)
- [Training Command](train.md)
- [Configuration Tutorial](../tutorials/configuration.md)
- [Best Practices Guide](../tutorials/best-practices.md)
- [Custom Functions Guide](../tutorials/custom-functions.md)
