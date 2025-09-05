# ModelGardener CLI Interface Documentation

The ModelGardener CLI interface provides a command-line, allowing you to configure and run model training with enhanced user-friendly features including automatic parameter extraction from custom functions and comprehensive configuration templates.

## 🆕 What's New - Enhanced Configuration System

**Major Improvements for User-Friendly Configuration:**

- ✨ **Automatic Parameter Extraction**: Custom function parameters are automatically detected and included in config.yaml
- 📋 **Comprehensive Comments**: All available options (optimizers, loss functions, metrics, training loops) are documented in the config
- 🔧 **Standardized Wrapper Functions**: Consistent parameter handling across all custom functions
- 🎯 **Smart Custom Function Integration**: Custom preprocessing, augmentation, callbacks seamlessly integrated
- 📊 **Enhanced YAML Templates**: More readable and configurable templates with parameter visibility

## Overview

The CLI interface consists of three main components:
- `modelgardener_cli.py` - Main CLI entry point with subcommands
- `cli_config.py` - Enhanced configuration management with automatic parameter extraction
- `custom_function_wrappers.py` - Standardized wrapper classes for custom function parameter handling

## Installation and Setup

1. **Install Dependencies**
   ```bash
   # Make sure you have the virtual environment activated
   source .venv/bin/activate
   
   # Install the required CLI dependency
   pip install inquirer==3.1.3
   ```

2. **Make Scripts Executable**
   ```bash
   chmod +x modelgardener_cli.py cli_config.py
   ```

## Quick Start

### 1. Create a New Project
```bash
python modelgardener_cli.py create my_project
cd my_project
```

### 2. Configure Your Model (Interactive Mode)
```bash
python /path/to/ModelGardener/modelgardener_cli.py config --interactive --output configs/model_config.json
```

### 3. Configure Your Model (Batch Mode)
```bash
python /path/to/ModelGardener/modelgardener_cli.py config \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --model-family resnet \
    --model-name "ResNet-50" \
    --epochs 50 \
    --batch-size 32 \
    --num-classes 10 \
    --learning-rate 0.001 \
    --output configs/model_config.json
```

### 4. Train Your Model
```bash
python /path/to/ModelGardener/modelgardener_cli.py train --config configs/model_config.json
```

## Available Commands

### `config` - Configuration Management

Configure model settings and save to JSON/YAML files.

**Interactive Mode:**
```bash
python modelgardener_cli.py config --interactive --output config.json
```

**Batch Mode:**
```bash
python modelgardener_cli.py config \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --model-family resnet \
    --model-name "ResNet-50" \
    --epochs 100 \
    --batch-size 32 \
    --output config.json
```

**Template Creation:**
```bash
python modelgardener_cli.py config --template --output template.json
```

**Validation:**
```bash
python modelgardener_cli.py config --validate --config existing_config.json
```

**Configuration Options:**
- `--train-dir`: Path to training data directory
- `--val-dir`: Path to validation data directory
- `--batch-size`: Batch size for training (default: 32)
- `--model-family`: Model architecture family (resnet, efficientnet, mobilenet, etc.)
- `--model-name`: Specific model name within the family
- `--num-classes`: Number of output classes
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimizer
- `--optimizer`: Optimizer type (Adam, SGD, RMSprop, etc.)
- `--loss-function`: Loss function type
- `--model-dir`: Directory for saving models and logs
- `--num-gpus`: Number of GPUs to use (0 for CPU-only)
- `--format`: Output format (json or yaml)

### 🐍 **Automatic Python Script Generation**

**NEW FEATURE**: When you save any configuration (using `config` command or GUI), ModelGardener automatically generates ready-to-run Python scripts in the same directory as the configuration file:

- `train.py` - Standalone training script
- `evaluation.py` - Model evaluation script  
- `prediction.py` - Prediction script for new images
- `deploy.py` - REST API deployment script
- `requirements.txt` - Python dependencies
- `README.md` - Usage instructions

**Example:**
```bash
# Generate configuration and scripts
python modelgardener_cli.py config --template --format yaml --output my_project/config.yaml

# This creates:
my_project/
├── config.yaml          # Configuration file
├── train.py             # Training script
├── evaluation.py        # Evaluation script
├── prediction.py        # Prediction script
├── deploy.py           # API deployment script
├── requirements.txt     # Dependencies
└── README.md           # Instructions

# Use the generated scripts directly:
cd my_project
pip install -r requirements.txt
python train.py                                    # Train model
python evaluation.py                               # Evaluate model
python prediction.py --input path/to/image.jpg    # Make predictions
python deploy.py --port 8080                       # Deploy API
```

**Script Features:**
- ✅ **Self-contained**: Each script runs independently with your configuration
- ✅ **Customizable**: Generated code can be modified for specific needs
- ✅ **Production-ready**: Includes error handling, logging, and best practices
- ✅ **Cross-validation support**: Training script supports k-fold CV when enabled
- ✅ **API deployment**: Deploy script creates REST API with health checks
- ✅ **Batch processing**: Prediction script handles single images or directories

### `train` - Model Training

Train a model using the specified configuration.

```bash
python modelgardener_cli.py train --config config.json
```

Options:
- `--config`: Required. Path to configuration file
- `--resume`: Resume training from checkpoint (if available)
- `--checkpoint`: Specific checkpoint file to resume from

### `evaluate` - Model Evaluation

Evaluate a trained model.

```bash
python modelgardener_cli.py evaluate --config config.json --model-path ./logs/final_model.keras
```

Options:
- `--config`: Required. Path to configuration file
- `--model-path`: Path to the trained model file

### `models` - List Available Models

Display all available model architectures.

```bash
python modelgardener_cli.py models
```

### `create` - Enhanced Project Template with Custom Function Support

Create a new project template with comprehensive structure including custom function templates and sample data.

```bash
python modelgardener_cli.py create my_project --dir ./projects
```

**NEW ENHANCED FEATURES:**

✨ **Automatic Custom Function Templates**: Generated projects now include:
- `custom_modules/custom_models.py` - Custom model architectures with parameter extraction
- `custom_modules/custom_data_loaders.py` - Custom data loading functions  
- `custom_modules/custom_preprocessing.py` - Custom preprocessing functions with parameters
- `custom_modules/custom_training_loops.py` - Custom training loop implementations
- `custom_modules/README.md` - Documentation for custom function development

📊 **Sample Data Included**: Projects come with ready-to-use sample data for immediate training

🔧 **Parameter Integration**: All custom functions have their parameters automatically extracted and included in config.yaml

Options:
- `project_name`: Name of the project (required)
- `--dir`: Directory to create the project in (default: current directory)

**Example Generated Structure:**
```
my_project/
├── config.yaml                          # Enhanced config with parameter extraction
├── train.py                            # Ready-to-run training script
├── evaluation.py                       # Model evaluation script
├── prediction.py                       # Prediction script
├── deploy.py                           # API deployment script
├── requirements.txt                    # Dependencies
├── README.md                          # Project documentation
├── data/                              # Sample training data
│   ├── train/                         # Training samples (3 classes)
│   └── val/                          # Validation samples
└── custom_modules/                    # Custom function templates
    ├── custom_models.py              # Model functions with parameters
    ├── custom_data_loaders.py        # Data loader functions  
    ├── custom_preprocessing.py       # Preprocessing functions
    ├── custom_training_loops.py      # Training loop functions
    └── README.md                     # Custom function guide
```

## 🆕 Enhanced Configuration File Structure

The CLI now generates user-friendly configuration files with **automatic parameter extraction** from custom functions. The enhanced structure includes:

### YAML Format with Comments and Parameter Extraction

```yaml
# ModelGardener Configuration Template - Ready to run with custom functions and sample data

# INSTRUCTIONS:
# 1. Sample data has been copied to ./data/ directory with 3 classes
# 2. Custom functions are configured in metadata section below  
# 3. Modify parameters below to customize training behavior
# 4. Run training with: python train.py

# AVAILABLE OPTIONS REFERENCE:
# - Optimizers: [Adam, SGD, RMSprop, Adagrad, AdamW, Adadelta, Adamax, Nadam, FTRL]
# - Loss Functions: [Categorical Crossentropy, Sparse Categorical Crossentropy, Binary Crossentropy, Mean Squared Error, Mean Absolute Error, Focal Loss, Huber Loss]  
# - Metrics: [Accuracy, Categorical Accuracy, Sparse Categorical Accuracy, Top-K Categorical Accuracy, Precision, Recall, F1 Score, AUC, Mean Squared Error, Mean Absolute Error]
# - Training Loops: [Standard Training, Progressive Training, Curriculum Learning, Adversarial Training, Self-Supervised Training]

configuration:
  task_type: image_classification
  data:
    train_dir: ./data/train
    val_dir: ./data/val
    preprocessing:
      # Built-in preprocessing options
      Resizing:
        enabled: true
        target_size: {width: 224, height: 224, depth: 1}
        interpolation: bilinear
      Normalization:
        enabled: true
        method: zero-center
        min_value: 0.0
        max_value: 1.0
      # ✨ NEW: Custom preprocessing with automatic parameter extraction
      Custom Preprocessing:
        enabled: false
        function_name: adaptive_histogram_equalization
        file_path: ./custom_modules/custom_preprocessing.py
        clip_limit: 2.0        # ← Automatically extracted parameter
        tile_grid_size: 8      # ← Automatically extracted parameter
    augmentation:
      # Built-in augmentation options  
      Horizontal Flip:
        enabled: false
        probability: 0.5
      Rotation:
        enabled: false
        angle_range: 15.0
        probability: 0.5
      # ✨ NEW: Custom augmentation integration (disabled by default)
      Custom Augmentation:
        enabled: false
        function_name: custom_augmentation_function
        file_path: ./custom_modules/custom_augmentations.py
        probability: 0.5
  model:
    model_family: custom_model
    model_name: create_simple_cnn
    # ✨ NEW: Model parameters automatically extracted
    model_parameters:
      input_shape: {width: 224, height: 224, channels: 3}
      num_classes: 1000      # ← Extracted from custom function
      dropout_rate: 0.5      # ← Extracted from custom function
  callbacks:
    Early Stopping:
      enabled: false
      monitor: val_loss
      patience: 10
    Model Checkpoint:
      enabled: true
      monitor: val_loss
      save_best_only: true
    # ✨ NEW: Custom callbacks integration
    Custom Callback:
      enabled: false
      function_name: custom_callback_function  
      file_path: ./custom_modules/custom_callbacks.py

# ✨ NEW: Metadata section with extracted function parameters
metadata:
  version: 1.2
  custom_functions:
    models:
    - name: create_simple_cnn
      file_path: ./custom_modules/custom_models.py
      function_name: create_simple_cnn
      type: function
      parameters:                    # ← Automatically extracted
        input_shape: (224, 224, 3)
        num_classes: 1000
        dropout_rate: 0.5
    preprocessing:  
    - name: adaptive_histogram_equalization
      file_path: ./custom_modules/custom_preprocessing.py
      function_name: adaptive_histogram_equalization
      type: function
      parameters:                    # ← Automatically extracted
        clip_limit: 2.0
        tile_grid_size: 8
    # More custom functions with extracted parameters...
```

### Key Enhancements:

🔍 **Automatic Parameter Extraction**: Function parameters are automatically detected using Python's `inspect` module and included in both the main configuration and metadata sections.

📋 **Comprehensive Comments**: All available options are documented directly in the config file, making it self-documenting.

🎯 **Custom Function Integration**: Custom preprocessing, augmentation, callbacks, models, data loaders, and training loops are seamlessly integrated.

🔧 **Standardized Parameter Handling**: All custom functions use consistent parameter structures through wrapper classes.

📊 **Enhanced Visibility**: Users can see and modify all function parameters without needing to examine source code.

```json
{
  "configuration": {
    "task_type": "image_classification",
    "data": {
      "train_dir": "./data/train",
      "val_dir": "./data/val",
      "data_loader": {
        "selected_data_loader": "Default",
        "parameters": {
          "batch_size": 32,
          "shuffle": true,
          "buffer_size": 10000
        }
      },
      "preprocessing": {
        "Resizing": {
          "enabled": true,
          "target_size": {"width": 224, "height": 224, "depth": 1}
        },
        "Normalization": {
          "enabled": true,
          "method": "zero-center"
        }
      },
      "augmentation": {
        "Horizontal Flip": {
          "enabled": false,
          "probability": 0.5
        },
        "Rotation": {
          "enabled": false,
          "angle_range": 15.0,
          "probability": 0.5
        }
      }
    },
    "model": {
      "model_family": "resnet",
      "model_name": "ResNet-50",
      "model_parameters": {
        "input_shape": {"height": 224, "width": 224, "channels": 3},
        "include_top": true,
        "classes": 1000
      },
      "optimizer": {
        "Optimizer Selection": {
          "selected_optimizer": "Adam",
          "learning_rate": 0.001
        }
      },
      "loss_functions": {
        "Loss Selection": {
          "selected_loss": "Categorical Crossentropy"
        }
      },
      "metrics": {
        "Metrics Selection": {
          "selected_metrics": "Accuracy"
        }
      },
      "callbacks": {
        "Early Stopping": {
          "enabled": false,
          "monitor": "val_loss",
          "patience": 10
        },
        "Model Checkpoint": {
          "enabled": true,
          "monitor": "val_loss",
          "save_best_only": true
        }
      }
    },
    "training": {
      "epochs": 100,
      "initial_learning_rate": 0.1,
      "cross_validation": {
        "enabled": false,
        "k_folds": 5,
        "validation_split": 0.2,
        "stratified": true
      }
    },
    "runtime": {
      "model_dir": "./logs",
      "distribution_strategy": "mirrored",
      "num_gpus": 0
    }
  },
  "metadata": {
    "version": "1.2",
    "custom_functions": {},
    "creation_date": "2025-08-26T13:45:03.334425"
  }
}
```

## Available Model Architectures

### ResNet Family
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

### EfficientNet Family
- EfficientNetB0 through EfficientNetB7

### MobileNet Family
- MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

### Other Architectures
- VGG16, VGG19
- DenseNet121, DenseNet169, DenseNet201
- InceptionV3, InceptionResNetV2
- Xception
- NASNetMobile, NASNetLarge

### Custom Models
Support for loading custom model architectures from Python files.

## 🔧 Advanced Configuration and Custom Functions

### Enhanced Custom Function Support

The CLI now provides comprehensive support for custom functions with **automatic parameter extraction** and **standardized wrapper classes**. Custom functions are seamlessly integrated into the configuration system.

#### Supported Custom Function Types:

1. **Custom Models** (`custom_modules/custom_models.py`)
2. **Custom Data Loaders** (`custom_modules/custom_data_loaders.py`) 
3. **Custom Preprocessing** (`custom_modules/custom_preprocessing.py`)
4. **Custom Augmentations** (`custom_modules/custom_augmentations.py`)
5. **Custom Callbacks** (`custom_modules/custom_callbacks.py`)
6. **Custom Training Loops** (`custom_modules/custom_training_loops.py`)

#### Automatic Parameter Extraction

When you create a project, the system automatically:

1. **Analyzes Function Signatures**: Uses Python's `inspect` module to extract parameters
2. **Includes Parameters in Config**: Adds all parameters to both main config and metadata
3. **Preserves Default Values**: Maintains function defaults as config defaults
4. **Enables Easy Customization**: Users can modify parameters without touching code

**Example: Custom Preprocessing Function**

```python
# custom_modules/custom_preprocessing.py
def adaptive_histogram_equalization(data, clip_limit=2.0, tile_grid_size=8):
    """Custom preprocessing with automatic parameter extraction."""
    # Implementation here...
    return processed_data
```

**Automatically Generated Config:**

```yaml
preprocessing:
  Custom Preprocessing:
    enabled: false
    function_name: adaptive_histogram_equalization
    file_path: ./custom_modules/custom_preprocessing.py
    clip_limit: 2.0        # ← Extracted automatically
    tile_grid_size: 8      # ← Extracted automatically

metadata:
  custom_functions:
    preprocessing:
    - name: adaptive_histogram_equalization
      parameters:           # ← Full parameter metadata  
        clip_limit: 2.0
        tile_grid_size: 8
```

#### Standardized Wrapper Classes

The system includes wrapper classes for consistent parameter handling:

```python
# Example usage of wrapper classes
from custom_function_wrappers import PreprocessingWrapper

# Create wrapper with custom parameters
wrapper = PreprocessingWrapper(my_function, {'clip_limit': 3.0})

# Apply with config-driven parameters
result = wrapper.apply(data, config_parameters)
```

**Available Wrapper Classes:**
- `PreprocessingWrapper` - For data preprocessing functions
- `AugmentationWrapper` - For data augmentation functions  
- `CallbackWrapper` - For training callbacks
- `ModelWrapper` - For custom model architectures
- `DataLoaderWrapper` - For custom data loading functions
- `TrainingLoopWrapper` - For custom training loops

#### Integration with Generated Scripts

Custom functions are automatically integrated into generated Python scripts:

```python
# Generated train.py includes custom function loading
if config['data']['preprocessing']['Custom Preprocessing']['enabled']:
    from custom_modules.custom_preprocessing import adaptive_histogram_equalization
    from custom_function_wrappers import PreprocessingWrapper
    
    # Create wrapper with config parameters
    wrapper = PreprocessingWrapper(adaptive_histogram_equalization)
    processed_data = wrapper.apply(data, preprocessing_config)
```

### Multiple Output Formats

Save configurations in different formats:

```bash
# JSON format (default)
python modelgardener_cli.py config --interactive --format json --output config.json

# YAML format
python modelgardener_cli.py config --interactive --format yaml --output config.yaml
```

### Environment Variables

You can set default values using environment variables:

```bash
export MODELGARDENER_DATA_DIR="/path/to/data"
export MODELGARDENER_MODEL_DIR="/path/to/models"
export MODELGARDENER_BATCH_SIZE=64
```

## Error Handling and Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install inquirer==3.1.3
   ```

2. **Configuration Validation Errors**
   Use the validation command to check your configuration:
   ```bash
   python modelgardener_cli.py config --validate --config config.yaml
   ```

3. **Path Issues**
   Always use absolute paths or ensure you're running from the correct directory.

4. **GPU Configuration**
   If you encounter GPU-related errors, try setting `--num-gpus 0` for CPU-only training.

### 🆕 New Feature Troubleshooting

5. **Custom Function Parameter Extraction Issues**
   If parameters aren't being extracted properly:
   ```bash
   # Check if custom modules are generated correctly
   ls -la my_project/custom_modules/
   
   # Verify function signatures in custom modules
   python -c "import inspect; from my_project.custom_modules.custom_preprocessing import adaptive_histogram_equalization; print(inspect.signature(adaptive_histogram_equalization))"
   ```

6. **Wrapper Class Import Errors**
   Ensure the wrapper classes are available:
   ```bash
   python -c "from custom_function_wrappers import PreprocessingWrapper; print('Wrapper classes available')"
   ```

7. **Custom Function File Not Found Warnings**
   If you see "Function parameter extraction: File not found" warnings:
   - Ensure custom modules are generated before parameter extraction
   - Check that the project directory exists
   - Verify file paths in the configuration are correct

8. **Parameter Override Not Working**
   When custom parameters aren't being applied:
   - Check the wrapper class is being used correctly
   - Verify the `apply()` method is called with the right config
   - Ensure parameter names match exactly between config and function signature

### Debugging Enhanced Features

**Enable Verbose Parameter Extraction:**
```python
# Add debug prints to see parameter extraction process
python modelgardener_cli.py create debug_project
# Check the generated config for extracted parameters
cat debug_project/config.yaml | grep -A 10 "Custom Preprocessing"
```

**Test Wrapper Functionality:**
```python
# Test wrapper classes manually
python -c "
from custom_function_wrappers import PreprocessingWrapper
import sys; sys.path.append('./my_project')
from custom_modules.custom_preprocessing import adaptive_histogram_equalization
wrapper = PreprocessingWrapper(adaptive_histogram_equalization)
print('Wrapper parameters:', wrapper.parameters)
"
```

### Verbose Output

For debugging, you can increase verbosity by modifying the scripts to include more detailed logging.

## Integration with Existing Workflows

### Batch Processing

Create multiple configurations for hyperparameter sweeps:

```bash
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    python modelgardener_cli.py config \
      --learning-rate $lr \
      --batch-size $bs \
      --output "config_lr${lr}_bs${bs}.json"
  done
done
```

### CI/CD Integration

The CLI interface is perfect for continuous integration workflows:

```yaml
# Example GitHub Actions workflow
- name: Train Model
  run: |
    python modelgardener_cli.py config --template --output config.json
    # Edit configuration as needed
    python modelgardener_cli.py train --config config.json
```

## Performance Tips

1. **Use appropriate batch sizes** based on your GPU memory
2. **Enable mixed precision** for faster training on modern GPUs
3. **Use multiple GPUs** when available with the `--num-gpus` option
4. **Monitor GPU utilization** during training

## Examples

### Basic Image Classification
```bash
python modelgardener_cli.py create image_classifier
cd image_classifier

python /path/to/ModelGardener/modelgardener_cli.py config \
  --train-dir "./data/train" \
  --val-dir "./data/val" \
  --model-family efficientnet \
  --model-name "EfficientNetB0" \
  --num-classes 10 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output configs/model_config.json

python /path/to/ModelGardener/modelgardener_cli.py train --config configs/model_config.json
```

### Transfer Learning Setup
```bash
python modelgardener_cli.py config \
  --model-family resnet \
  --model-name "ResNet-50" \
  --num-classes 5 \
  --learning-rate 0.0001 \
  --epochs 20 \
  --output transfer_learning_config.json
```

## 🎉 Benefits of the Enhanced System

### For End Users

✨ **Simplified Configuration**: All custom function parameters are visible and editable in config.yaml without needing to examine source code

📋 **Self-Documenting**: Comprehensive comments show all available options for optimizers, loss functions, metrics, and training loops  

🔧 **Easy Customization**: Modify function behavior by changing config values instead of editing code

🎯 **Ready-to-Run**: Generated projects include sample data and working examples for immediate training

📊 **Parameter Visibility**: See exactly what parameters are available for each custom function

### For Developers

🔍 **Automatic Parameter Detection**: System automatically extracts function parameters using Python's inspect module

🏗️ **Standardized Architecture**: Consistent wrapper classes handle parameter management across all function types

⚡ **Reduced Manual Work**: No more manual parameter configuration - everything is extracted automatically

🔄 **Backwards Compatible**: Existing configurations continue to work while gaining new features

🧪 **Easy Testing**: Wrapper classes enable isolated testing of custom functions with different parameter sets

### System Improvements

🚀 **Enhanced Workflow**: From project creation to training, the entire workflow is streamlined

📦 **Complete Project Templates**: Generated projects are fully functional with custom modules, sample data, and documentation

🎨 **Clean Separation**: Clear separation between configuration, custom functions, and generated scripts

🔧 **Maintainable**: Wrapper classes provide consistent interfaces for all custom function types

📈 **Scalable**: System easily accommodates new custom function types and parameters

## Contributing

To extend the CLI interface with new features:

1. **Add Parameter Extraction**: Extend `_extract_function_parameters()` in `cli_config.py` for new function types
2. **Create Wrapper Classes**: Add new wrapper classes in `custom_function_wrappers.py` following existing patterns  
3. **Update Templates**: Modify template generation to include new function types
4. **Add Documentation**: Update this README and add examples for new features
5. **Test Integration**: Ensure new features work with both config generation and script generation

### New Feature Development Guidelines

When adding new custom function types:

1. **Follow Naming Conventions**: Use consistent naming (e.g., `custom_modules/custom_<type>.py`)
2. **Implement Wrapper Classes**: Create wrapper classes inheriting from `CustomFunctionWrapper`  
3. **Add Parameter Extraction**: Include parameter extraction in the config generation process
4. **Update Templates**: Modify YAML templates to include the new function type
5. **Document Parameters**: Ensure parameters are well-documented in function signatures

## Support

If you encounter issues with the CLI interface:

1. Check this documentation
2. Validate your configuration file
3. Ensure all dependencies are installed
4. Check file paths and permissions
5. Review the error messages for specific guidance
