# `train` Command

Train machine learning models using comprehensive configuration with support for distributed training, custom functions, and advanced monitoring.

## Synopsis

```bash
modelgardener_cli.py train [OPTIONS]
```

## Description

The `train` command executes the training pipeline with:

- Advanced model architectures with custom function support
- Distributed training across multiple GPUs
- Comprehensive logging and monitoring
- Real-time visualization and metrics tracking
- Automatic checkpointing and model saving
- Cross-validation and hyperparameter optimization

## Options

### Configuration Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--config` | `-c` | `str` | Path to YAML configuration file | `config.yaml` |
| `--resume` | `-r` | `str` | Path to checkpoint to resume from | None |
| `--override` | `-o` | `str` | Override config values (key=value) | None |

### Training Control

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--epochs` | `int` | Number of training epochs | From config |
| `--batch-size` | `int` | Training batch size | From config |
| `--learning-rate` | `float` | Initial learning rate | From config |
| `--validation-split` | `float` | Validation data split ratio | From config |

### Model Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model-family` | `str` | Model architecture family | From config |
| `--model-name` | `str` | Specific model within family | From config |
| `--pretrained` | `flag` | Use pretrained weights | From config |
| `--fine-tune` | `flag` | Enable fine-tuning mode | False |

### Distributed Training

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

### Output and Logging

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--output-dir` | `str` | Directory for model outputs | From config |
| `--log-level` | `str` | Logging level (DEBUG, INFO, WARNING) | `INFO` |
| `--tensorboard` | `flag` | Enable TensorBoard logging | From config |
| `--wandb` | `flag` | Enable Weights & Biases logging | False |

## Usage Examples

### Basic Training

```bash
# Train with default configuration
modelgardener_cli.py train

# Train with specific config file
modelgardener_cli.py train --config my_config.yaml

# Train with parameter overrides
modelgardener_cli.py train --epochs 200 --learning-rate 0.001
```

### Configuration Override

```bash
# Override specific configuration values
modelgardener_cli.py train --override "training.epochs=300"
modelgardener_cli.py train --override "model.model_parameters.classes=50"
modelgardener_cli.py train --override "data.batch_size=64,training.learning_rate=0.001"
```

### Resume Training

```bash
# Resume from checkpoint
modelgardener_cli.py train --resume ./logs/checkpoint_epoch_50.keras

# Resume with different configuration
modelgardener_cli.py train --resume ./logs/latest_checkpoint.keras --config new_config.yaml
```

### Distributed Training

```bash
# Multi-GPU training
modelgardener_cli.py train --gpus 4 --strategy mirror

# Mixed precision training
modelgardener_cli.py train --mixed-precision --gpus 2

# Multi-worker distributed training
modelgardener_cli.py train --strategy multi_worker --gpus 8
```

### Advanced Training Modes

```bash
# Cross-validation training
modelgardener_cli.py train --cross-validation 5

# Hyperparameter tuning
modelgardener_cli.py train --hyperparameter-tuning --epochs 100

# Fine-tuning pre-trained model
modelgardener_cli.py train --fine-tune --pretrained --learning-rate 0.0001
```

### Monitoring and Logging

```bash
# Enable comprehensive logging
modelgardener_cli.py train --tensorboard --wandb --log-level DEBUG

# Custom output directory
modelgardener_cli.py train --output-dir ./experiments/run_001
```

## Configuration File Structure

The training command uses a comprehensive YAML configuration:

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
        preprocessing_function: null
    
    preprocessing:
      resize:
        height: 224
        width: 224
      normalization:
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
modelgardener_cli.py train --gpus 4 --strategy mirror

# Multi-worker strategy (for multi-node)
modelgardener_cli.py train --strategy multi_worker
```

**Mixed Precision Training:**
```bash
# Enable automatic mixed precision
modelgardener_cli.py train --mixed-precision
```

### Cross-Validation

```bash
# 5-fold cross-validation
modelgardener_cli.py train --cross-validation 5

# Stratified cross-validation (automatically detected)
modelgardener_cli.py train --cross-validation 10
```

**Cross-Validation Output:**
- Individual fold results
- Average performance metrics
- Standard deviation across folds
- Best performing fold model

### Hyperparameter Tuning

```bash
# Enable hyperparameter optimization
modelgardener_cli.py train --hyperparameter-tuning
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
modelgardener_cli.py train --resume ./logs/checkpoint_latest.keras

# Resume with modified config
modelgardener_cli.py train --resume checkpoint.keras --epochs 200
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
modelgardener_cli.py train --mixed-precision

# Optimize batch size for available memory
modelgardener_cli.py train --batch-size 16  # For limited memory
```

### Training Speed

```bash
# Multi-GPU acceleration
modelgardener_cli.py train --gpus 4

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
modelgardener_cli.py train --batch-size 16

# Enable mixed precision
modelgardener_cli.py train --mixed-precision
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
modelgardener_cli.py train --learning-rate 0.0001

# Enable early stopping
# Check data quality and preprocessing
```

### Debugging Options

```bash
# Verbose logging
modelgardener_cli.py train --log-level DEBUG

# Single epoch test
modelgardener_cli.py train --epochs 1

# Small dataset test
# Modify config to use subset of data
```

## Integration with Other Commands

### After Training

```bash
# Evaluate trained model
modelgardener_cli.py evaluate --config config.yaml --model ./logs/models/best_model.keras

# Make predictions
modelgardener_cli.py predict --config config.yaml --model ./logs/models/best_model.keras --input image.jpg

# Deploy model
modelgardener_cli.py deploy --config config.yaml --model ./logs/models/best_model.keras
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
