# `create` Command

Create new ModelGardener projects with customizable templates, sample data, and ready-to-use configurations.

## Synopsis

```bash
mg create [PROJECT_NAME] [OPTIONS]
```

## Description

The `create` command initializes a new machine learning project with:

- Pre-configured YAML configuration file
- Generated Python scripts (train.py, evaluation.py, prediction.py, deploy.py)
- Custom function templates
- Sample data structure
- Documentation and requirements

## Arguments

### Positional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `PROJECT_NAME` | Name of the project to create (optional) | Current directory name |

## Options

### Directory and Project Options

| Option | Short | Type | Description | Default |
|--------|-------|------|-------------|---------|
| `--dir` | `-d` | `str` | Directory to create project in | Current directory (`.`) |
| `--interactive` | `-i` | `flag` | Enable interactive project creation mode | Disabled |
| `--auto-generate-scripts` |  | `flag` | Enable auto-generation of training scripts | Enabled |
| `--no-auto-generate-scripts` |  | `flag` | Disable auto-generation of training scripts | N/A |
| `--use-pyproject` |  | `flag` | Generate pyproject.toml instead of requirements.txt | Enabled |
| `--use-requirements` |  | `flag` | Generate requirements.txt instead of pyproject.toml | N/A |

### Model Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model-family` | `str` | Model architecture family (resnet, efficientnet, etc.) | Auto-selected |
| `--model-name` | `str` | Specific model name within family | Auto-selected |
| `--num-classes` | `int` | Number of output classes | 10 |

### Data Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--train-dir` | `str` | Training data directory path | `./data/train` |
| `--val-dir` | `str` | Validation data directory path | `./data/val` |
| `--batch-size` | `int` | Training batch size | 32 |

### Training Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--epochs` | `int` | Number of training epochs | 100 |
| `--learning-rate` | `float` | Initial learning rate | 0.001 |
| `--optimizer` | `str` | Optimizer type (Adam, SGD, RMSprop) | Adam |
| `--loss-function` | `str` | Loss function type | Categorical Crossentropy |

### Runtime Configuration

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model-dir` | `str` | Directory for saving models and logs | `./logs` |
| `--num-gpus` | `int` | Number of GPUs to use (0 for CPU-only) | Auto-detect |

## Usage Examples

### Basic Project Creation

```bash
# Create project in current directory
mg create

# Create named project in current directory
mg create my_classifier

# Create project in specific directory
mg create image_classifier --dir /path/to/workspace
```

### Interactive Mode

```bash
# Interactive project creation with guided setup
mg create my_project --interactive

# Interactive mode in specific directory
mg create --interactive --dir ./projects/new_project
```

### Batch Mode with Parameters

```bash
# Create project with specific configuration
mg create advanced_classifier \
    --model-family resnet \
    --model-name ResNet-50 \
    --num-classes 100 \
    --epochs 200 \
    --learning-rate 0.001 \
    --batch-size 64 \
    --optimizer Adam

# Create GPU-optimized project
mg create gpu_project \
    --num-gpus 2 \
    --batch-size 128 \
    --model-family efficientnet
```

### Data-Specific Configuration

```bash
# Create project with custom data paths
mg create custom_data_project \
    --train-dir /dataset/custom_train \
    --val-dir /dataset/custom_val \
    --num-classes 50

# Create project for specific dataset structure
mg create cifar_project \
    --num-classes 10 \
    --batch-size 32 \
    --model-family resnet
```

## Generated Project Structure

When you run the `create` command, it generates the following structure:

```
project_name/
├── config.yaml                # Main configuration file
├── train.py                   # Training script
├── evaluation.py              # Evaluation script
├── prediction.py              # Prediction script
├── deploy.py                  # Deployment script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/                      # Data directory
│   ├── train/                 # Training data
│   │   ├── class_0/           # Sample class directories
│   │   ├── class_1/
│   │   └── class_2/
│   └── val/                   # Validation data
│       ├── class_0/
│       ├── class_1/
│       └── class_2/
├── logs/                      # Model outputs and logs
├── custom_modules/            # Custom function templates
│   ├── __init__.py
│   ├── custom_models.py       # Custom model functions
│   ├── custom_data_loaders.py # Custom data loader functions
│   ├── custom_loss_functions.py # Custom loss functions
│   ├── custom_metrics.py      # Custom metrics
│   ├── custom_callbacks.py    # Custom callbacks
│   ├── custom_optimizers.py   # Custom optimizers
│   ├── custom_augmentations.py # Custom augmentations
│   ├── custom_preprocessing.py # Custom preprocessing
│   ├── custom_training_loops.py # Custom training loops
│   └── README.md              # Custom functions guide
└── example_data/              # Sample training data
    └── [sample images]
```

## Interactive Mode Features

When using `--interactive` mode, you'll be guided through:

### Model Selection
```
🤖 Model Selection:
1. Select model family (ResNet, EfficientNet, MobileNet, etc.)
2. Choose specific model variant
3. Configure input shape and classes
```

### Data Configuration
```
📁 Data Configuration:
1. Set data directory paths
2. Configure batch size and preprocessing
3. Set up data augmentation options
```

### Training Parameters
```
🎯 Training Configuration:
1. Set number of epochs
2. Configure learning rate and optimizer
3. Set up loss function and metrics
4. Configure callbacks and validation
```

### Custom Functions
```
🔧 Custom Functions:
1. Choose custom model architectures
2. Select custom data loaders
3. Configure custom loss functions
4. Set up custom metrics and callbacks
```

## Generated Configuration

The `config.yaml` file created includes comprehensive settings:

```yaml
configuration:
  data:
    train_dir: "./data/train"
    val_dir: "./data/val"
    data_loader:
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
    mixed_precision: true

metadata:
  custom_functions:
    models: []
    data_loaders: []
    loss_functions: []
    metrics: []
    callbacks: []
```

## Generated Scripts Features

### train.py
- Comprehensive training pipeline
- Cross-validation support
- Custom function integration
- Progress tracking and logging

### evaluation.py
- Detailed metrics calculation
- Confusion matrix generation
- Per-class performance analysis
- Visualization plots

### prediction.py
- Single image and batch prediction
- Performance optimization
- Visualization generation
- Flexible output formats

### deploy.py
- Multi-format model serving
- REST API endpoints
- Performance monitoring
- Security features

## Sample Data Generation

The command creates sample training data to help you get started:

```
example_data/
├── class_0/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── sample_003.jpg
├── class_1/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── sample_003.jpg
└── class_2/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── sample_003.jpg
```

## Custom Function Templates

Generated custom function templates include:

### Custom Models
```python
def create_simple_cnn(input_shape, num_classes):
    """Example custom CNN model."""
    # Implementation with configurable parameters
    pass
```

### Custom Data Loaders
```python
def custom_image_data_loader(train_dir, val_dir, **kwargs):
    """Example custom data loader."""
    # Implementation with flexible parameters
    pass
```

### Custom Loss Functions
```python
def dice_loss(y_true, y_pred):
    """Example custom loss function."""
    # Implementation for specialized loss
    pass
```

## Best Practices

### Project Naming
- Use descriptive project names: `image_classifier`, `object_detection`, etc.
- Avoid spaces and special characters
- Use underscores for multi-word names

### Directory Structure
- Keep data directories separate from code
- Use relative paths in configuration
- Organize custom functions by type

### Configuration Management
- Start with interactive mode for complex projects
- Use batch mode for automated project creation
- Validate configuration before training

## Common Use Cases

### Computer Vision Projects
```bash
mg create vision_project \
    --model-family efficientnet \
    --num-classes 1000 \
    --batch-size 64 \
    --epochs 300
```

### Small Dataset Projects
```bash
mg create small_dataset \
    --num-classes 3 \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.01
```

### Research Projects
```bash
mg create research_project \
    --interactive \
    --model-family custom \
    --epochs 1000
```

## Troubleshooting

### Common Issues

**Permission Errors**
```bash
# Ensure write permissions to target directory
chmod 755 /target/directory
```

**Directory Already Exists**
```bash
# The command will prompt before overwriting
# Use --force flag to overwrite automatically (if implemented)
```

**Missing Dependencies**
```bash
# Install required packages
pip install -r requirements.txt
```

## Next Steps

After creating a project:

1. **Review Configuration**: Check `config.yaml` for your specific needs
2. **Prepare Data**: Add your training data to the `data/` directories
3. **Customize Functions**: Modify custom functions in `custom_modules/`
4. **Train Model**: Run `mg train --config config.yaml`
5. **Evaluate Results**: Use `mg evaluate --config config.yaml`

## See Also

- [Configuration Guide](../tutorials/configuration.md)
- [Custom Functions Tutorial](../tutorials/custom-functions.md)
- [Training Command](train.md)
- [Complete Workflow Tutorial](../tutorials/complete-workflow.md)
