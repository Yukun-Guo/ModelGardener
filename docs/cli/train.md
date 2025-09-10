# `train` Command

Train machine learning models using ModelGardener configuration files with support for resuming from checkpoints.

## Synopsis

```bash
mg train [OPTIONS]
```

## Description

The `train` command executes the training pipeline based on a configuration file. It supports:

- Training from configuration files (YAML/JSON)
- Resuming training from checkpoints
- Automatic model saving and logging
- Progress tracking and metrics reporting

## Options

### Configuration Options

| Option | Short | Type | Description | Required |
|--------|-------|------|-------------|----------|
| `--config` | `-c` | `str` | Configuration file path | No (searches for `config.yaml` in current directory if not provided) |

### Optional Training Control

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--resume` | `flag` | Resume training from checkpoint | False |
| `--checkpoint` | `str` | Specific checkpoint file to resume from | None |

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--gpus` | `int` | Number of GPUs to use | Auto-detect |
| `--strategy` | `str` | Distribution strategy (mirror, multi_worker) | `mirror` |
| `--mixed-precision` | `flag` | Enable mixed precision training | From config |

### Advanced Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--cross-validation` | `int` | K-fold cross-validation folds | None |
| `--hyperparameter-tuning` | `flag` | Enable hyperparameter optimization | False |
| `--early-stopping` | `flag` | Enable early stopping | From config |
| `--save-best-only` | `flag` | Save only best model | True |

## Usage Examples

### Basic Training

```bash
# Train with automatic config detection (looks for config.yaml in current directory)
mg train

# Train with specific configuration file
mg train --config config.yaml

# Train with different configuration file
mg train --config my_training_config.yaml
```

### Resume Training

```bash
# Resume training from checkpoint (with auto config detection)
mg train --resume

# Resume training from checkpoint with specific config
mg train --config config.yaml --resume

# Resume from specific checkpoint file
mg train --config config.yaml --resume --checkpoint path/to/checkpoint.keras
```

### Common Training Workflows

```bash
# Complete workflow: create, configure, train
mg create my_project
cd my_project
mg config config.yaml --epochs 100 --batch-size 32
mg train  # Uses config.yaml automatically

# Resume training after interruption
mg train --resume

# Train multiple configurations
mg train --config config_v1.yaml
mg train --config config_v2.yaml --resume
```

## Configuration Requirements

The train command requires a valid configuration file that includes:

### Required Configuration Sections

- **Data Configuration**: Training and validation data paths
- **Model Configuration**: Model architecture and parameters  
- **Training Configuration**: Training parameters like epochs, batch size, learning rate
- **Output Configuration**: Model save paths and logging settings

### Example Configuration Structure

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
    
  model:
    model_family: "resnet"
    model_name: "resnet50"
    model_parameters:
      classes: 10
      input_shape: [224, 224, 3]
      
  training:
    epochs: 100
    learning_rate: 0.001
    optimizer: "adam"
    loss_function: "categorical_crossentropy"
    
  output:
    model_dir: "./models"
    logs_dir: "./logs"
```

## Training Process

When you run the train command, ModelGardener:

1. **Loads Configuration**: Reads and validates the configuration file
2. **Prepares Data**: Sets up data loaders and preprocessing
3. **Builds Model**: Creates the model architecture based on configuration
4. **Configures Training**: Sets up optimizer, loss function, and callbacks
5. **Executes Training**: Runs the training loop with progress monitoring
6. **Saves Results**: Saves trained model and training logs

### Checkpoint and Resume Functionality

- **Automatic Checkpointing**: Models are automatically saved during training
- **Resume Training**: Use `--resume` to continue from the last checkpoint
- **Custom Checkpoints**: Specify `--checkpoint` to resume from a specific file
- **Progress Preservation**: Training metrics and state are preserved across resume

## Output and Logs

Training generates several outputs:

### Model Files
- Trained model files (`.keras`, `.h5`)
- Model architecture files
- Training checkpoints

### Logs and Metrics  
- Training progress logs
- Validation metrics
- Loss and accuracy curves
- TensorBoard logs (if configured)

### Directory Structure
```
project/
├── models/          # Trained model files
├── logs/            # Training logs and metrics
├── checkpoints/     # Training checkpoints
└── config.yaml      # Configuration file
```

## Integration with Other Commands

The train command works with other ModelGardener commands:

```bash
# Create project and train
mg create project_name
mg train --config project_name/config.yaml

# Configure then train
mg config config.yaml --epochs 200 --learning-rate 0.01
mg train --config config.yaml

# Train then evaluate
mg train --config config.yaml
mg evaluate --config config.yaml

# Train then predict
mg train --config config.yaml  
mg predict --config config.yaml --input test_image.jpg
```

## Tips and Best Practices

1. **Always use configuration files** for reproducible training
2. **Enable resume functionality** for long training runs
3. **Monitor training progress** through logs and metrics
4. **Use appropriate batch sizes** based on your hardware
5. **Save checkpoints frequently** to prevent data loss
6. **Validate configuration** before starting long training runs

## Troubleshooting

### Common Issues

- **Configuration errors**: Use `mg check config.yaml` to validate configuration
- **Memory issues**: Reduce batch size in configuration
- **Resume failures**: Check checkpoint file paths and compatibility
- **Data loading errors**: Verify data directory paths in configuration

### Error Resolution

```bash
# Check configuration before training
mg check config.yaml
mg train --config config.yaml

# Fix data path issues
mg config config.yaml --train-dir /correct/path/to/train
mg train --config config.yaml
```

## Related Commands

- [`mg create`](create.md) - Create new project with training configuration
- [`mg config`](config.md) - Modify training configuration parameters
- [`mg check`](check.md) - Validate configuration before training
- [`mg evaluate`](evaluate.md) - Evaluate trained models
- [`mg predict`](predict.md) - Make predictions with trained models
        rescale: 0.00392156862
      augmentation:
        rotation_range: 20
        width_shift_range: 0.2
        height_shift_range: 0.2
        horizontal_flip: true
  
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
      weights: "imagenet"
      include_top: false
      pooling: "avg"
    
    compilation:
      optimizer:
        name: "Adam"
        parameters:
          learning_rate: 0.001
          beta_1: 0.9
          beta_2: 0.999
      
      loss:
        name: "categorical_crossentropy"
        parameters: {}
      
      metrics:
        - "accuracy"
        - "top_k_categorical_accuracy"
  
  training:
    epochs: 100
    initial_learning_rate: 0.001
    callbacks:
      early_stopping:
        monitor: "val_loss"
        patience: 10
        restore_best_weights: true
      
      reduce_lr_on_plateau:
        monitor: "val_loss"
        factor: 0.2
        patience: 5
        min_lr: 0.0001
      
      model_checkpoint:
        monitor: "val_accuracy"
        save_best_only: true
        save_weights_only: false
  
  runtime:
    model_dir: "./logs"
    use_gpu: true
    mixed_precision: true
    distribution_strategy: "mirrored"

metadata:
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

## Training Pipeline Features

### Model Architecture Support

**Pre-built Models:**
- ResNet (ResNet-50, ResNet-101, ResNet-152)
- EfficientNet (B0-B7)
- MobileNet (V1, V2, V3)
- DenseNet (121, 169, 201)
- Inception (V3, ResNet-V2)
- Xception
- VGG (16, 19)

**Custom Models:**
- User-defined architectures
- Transfer learning support
- Fine-tuning capabilities
- Layer freezing/unfreezing

### Data Pipeline

**Data Loading:**
- Multiple data loader types
- Custom data generators
- Memory-efficient loading
- Parallel data processing

**Preprocessing:**
- Image resizing and normalization
- Data augmentation
- Custom preprocessing functions
- Real-time transformations

### Training Features

**Optimization:**
- Multiple optimizer support (Adam, SGD, RMSprop, AdamW)
- Learning rate scheduling
- Gradient clipping
- Weight decay

**Regularization:**
- Dropout layers
- Batch normalization
- Early stopping
- Data augmentation

**Monitoring:**
- Real-time metrics tracking
- Loss and accuracy visualization
- Training progress bars
- Resource utilization monitoring

### Distributed Training

**Multi-GPU Support:**
```bash
# Mirror strategy (recommended for single-node)
mg train --gpus 4 --strategy mirror

# Multi-worker strategy (for multi-node)
mg train --strategy multi_worker
```

**Mixed Precision Training:**
```bash
# Enable automatic mixed precision
mg train --mixed-precision
```

### Cross-Validation

```bash
# 5-fold cross-validation
mg train --cross-validation 5

# Stratified cross-validation (automatically detected)
mg train --cross-validation 10
```

**Cross-Validation Output:**
- Individual fold results
- Average performance metrics
- Standard deviation across folds
- Best performing fold model

### Hyperparameter Tuning

```bash
# Enable hyperparameter optimization
mg train --hyperparameter-tuning
```

**Tunable Parameters:**
- Learning rate
- Batch size
- Model architecture parameters
- Optimizer parameters
- Regularization parameters

### Callbacks and Monitoring

**Built-in Callbacks:**
- EarlyStopping: Prevent overfitting
- ReduceLROnPlateau: Adaptive learning rate
- ModelCheckpoint: Save best models
- CSVLogger: Log training metrics
- TensorBoard: Visualization

**Custom Callbacks:**
- User-defined callback functions
- Integration with external monitoring
- Custom logging and alerting

### Checkpointing and Resume

**Automatic Checkpointing:**
- Save model at intervals
- Best model preservation
- Training state saving

**Resume Training:**
```bash
# Resume from last checkpoint
mg train --resume ./logs/checkpoint_latest.keras

# Resume with modified config
mg train --resume checkpoint.keras --epochs 200
```

## Output Structure

Training generates the following output structure:

```
logs/
├── models/
│   ├── best_model.keras          # Best performing model
│   ├── final_model.keras         # Final epoch model
│   ├── checkpoint_epoch_*.keras  # Periodic checkpoints
│   └── model_architecture.json   # Model architecture
├── logs/
│   ├── training.log              # Detailed training logs
│   ├── metrics.csv               # Training metrics
│   ├── config_used.yaml          # Configuration used
│   └── tensorboard/              # TensorBoard logs
├── plots/
│   ├── training_history.png      # Loss/accuracy plots
│   ├── model_architecture.png    # Model visualization
│   └── learning_curves.png       # Learning rate curves
└── reports/
    ├── training_summary.txt       # Training summary
    ├── hyperparameters.json       # Final hyperparameters
    └── performance_metrics.json   # Final metrics
```

## Custom Functions Integration

### Custom Models

```python
# In custom_modules/custom_models.py
def create_advanced_cnn(input_shape, num_classes, **kwargs):
    """Custom CNN architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### Custom Data Loaders

```python
# In custom_modules/custom_data_loaders.py
def advanced_image_loader(train_dir, val_dir, **kwargs):
    """Advanced data loading with custom preprocessing."""
    # Custom implementation
    return train_generator, val_generator
```

### Custom Loss Functions

```python
# In custom_modules/custom_loss_functions.py
def focal_loss(alpha=1, gamma=2):
    """Focal loss for imbalanced datasets."""
    def focal_loss_fixed(y_true, y_pred):
        # Implementation
        pass
    return focal_loss_fixed
```

## Performance Optimization

### Memory Optimization

```bash
# Enable mixed precision for memory efficiency
mg train --mixed-precision

# Optimize batch size for available memory
mg train --batch-size 16  # For limited memory
```

### Training Speed

```bash
# Multi-GPU acceleration
mg train --gpus 4

# Parallel data loading (configured in YAML)
# Set num_workers in data loader configuration
```

### Resource Monitoring

During training, the system monitors:
- GPU utilization and memory usage
- CPU usage and system memory
- Training speed (samples/second)
- Estimated time remaining

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```bash
# Reduce batch size
mg train --batch-size 16

# Enable mixed precision
mg train --mixed-precision
```

**Slow Training:**
```bash
# Check data loading bottlenecks
# Increase num_workers in data loader config
# Use faster storage (SSD)
# Enable GPU acceleration
```

**Training Convergence Issues:**
```bash
# Adjust learning rate
mg train --learning-rate 0.0001

# Enable early stopping
# Check data quality and preprocessing
```

### Debugging Options

```bash
# Verbose logging
mg train --log-level DEBUG

# Single epoch test
mg train --epochs 1

# Small dataset test
# Modify config to use subset of data
```

## Integration with Other Commands

### After Training

```bash
# Evaluate trained model
mg evaluate --config config.yaml --model ./logs/models/best_model.keras

# Make predictions
mg predict --config config.yaml --model ./logs/models/best_model.keras --input image.jpg

# Deploy model
mg deploy --config config.yaml --model ./logs/models/best_model.keras
```

## Best Practices

### Configuration Management

1. **Use version control** for configuration files
2. **Document parameter choices** in comments
3. **Test configurations** with small datasets first
4. **Save successful configurations** for reuse

### Training Strategy

1. **Start with pre-trained models** for faster convergence
2. **Use appropriate learning rates** for your dataset size
3. **Monitor validation metrics** to prevent overfitting
4. **Save checkpoints regularly** for long training runs

### Resource Management

1. **Monitor GPU memory usage** during training
2. **Use appropriate batch sizes** for your hardware
3. **Clean up old models** to save disk space
4. **Use distributed training** for large datasets

## See Also

- [Configuration Guide](../tutorials/configuration.md)
- [Custom Functions Tutorial](../tutorials/custom-functions.md)
- [Evaluation Command](evaluate.md)
- [Distributed Training Tutorial](../tutorials/distributed-training.md)
- [Hyperparameter Tuning Guide](../tutorials/hyperparameter-tuning.md)
