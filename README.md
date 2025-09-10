# ModelGardener

A modular ML framework that auto-generates Python scripts from YAML configurations and supports custom function injection. ModelGardener simplifies deep learning workflows by providing a configuration-driven approach to model training, evaluation, prediction, and deployment.

## Features

- **Configuration-Driven**: Define your entire ML pipeline through YAML configuration
- **Auto-Generated Scripts**: Automatically generates `train.py`, `evaluation.py`, `prediction.py`, and `deploy.py`
- **Enhanced CLI with Auto-Discovery**: Intelligent file discovery eliminates need for repetitive parameter specification
- **Custom Function Support**: Extend functionality with custom models, loss functions, data loaders, and more
- **Multi-Format Deployment**: Deploy models in ONNX, TFLite, TF.js, and Keras formats with quantization support
- **Interactive CLI**: Guided project creation and configuration management
- **Data Visualization**: Preview and visualize your data with preprocessing and augmentation
- **Flexible Architecture**: Supports 2D/3D data, multi-input/output models, and various task types
- **Comprehensive Reporting**: Auto-generated evaluation and prediction reports with metadata

## Enhanced CLI Features

### 🔍 Intelligent Auto-Discovery
ModelGardener's CLI automatically discovers project files to minimize manual configuration:

- **Config Discovery**: Finds `config.yaml` in current directory when not specified
- **Model Discovery**: Locates the latest trained model in `logs/` directory structure
- **Data Discovery**: Identifies test datasets in common directories (`test/`, `val/`, `data/test/`, etc.)
- **Smart Fallbacks**: Graceful degradation with helpful error messages when files aren't found

### ⚡ Short Parameter Support
All major commands support intuitive short parameters:
- `-c` for `--config`
- `-m` for `--model-path`  
- `-i` for `--input`
- `-o` for `--output`
- `-f` for `--formats`
- `-q` for `--quantize`
- `-e` for `--encrypt`
- `-d` for `--data-path`

### 📊 Enhanced Reporting
Automatic generation of comprehensive reports:
- **Evaluation Reports**: JSON/YAML formats with metadata and metrics in `evaluation/` folder
- **Prediction Reports**: JSON format with CSV summaries in `predictions/` folder  
- **Timestamped Files**: All reports include timestamps for version tracking
- **Flexible Output**: Control saving behavior with `--no-save` options

## Installation

```bash
# Install from source
pip install git+https://github.com/Yukun-Guo/ModelGardener.git
# Verify installation
mg --help
```

## CLI Commands Reference

ModelGardener provides a comprehensive CLI for managing your ML projects. Use `mg --help` to see all available commands.

### Project Management

#### `mg create` - Create New Project
Create a new ModelGardener project with interactive setup or command-line arguments.

```bash
# Interactive project creation
mg create my_project --interactive
mg create --interactive  # Create in current directory

# Quick project creation
mg create my_project --dir /path/to/workspace
mg create  # Create basic template in current directory

# With specific configuration
mg create my_project --batch-size 32 --epochs 100 --model-family efficientnet
```

**Options:**
- `--dir DIR`: Directory to create project in
- `--interactive`: Interactive project creation mode
- `--script/--no-script`: Enable/disable auto-generation of training scripts
- `--use-pyproject/--use-requirements`: Choose dependency management format
- Configuration options: `--train-dir`, `--val-dir`, `--batch-size`, `--model-family`, `--model-name`, `--num-classes`, `--epochs`, `--learning-rate`, `--optimizer`, `--loss-function`, `--model-dir`, `--num-gpus`

#### `mg config` - Modify Configuration
Modify existing model configuration files interactively or via command-line arguments.

```bash
# Interactive configuration modification
mg config config.yaml --interactive
mg config --interactive  # Auto-finds config in current dir

# Direct parameter modification
mg config config.yaml --epochs 100 --learning-rate 0.01 --batch-size 64
mg config config.yaml --model-family resnet --num-classes 10
```

**Options:**
- `--interactive`: Interactive configuration modification mode
- All the same configuration options as `mg create`

#### `mg check` - Validate Configuration
Check and validate configuration files for errors and completeness.

```bash
# Basic configuration check
mg check config.yaml

# Detailed validation with verbose output
mg check config.yaml --verbose
```

**Options:**
- `--verbose`: Show detailed validation results

### Data Management

#### `mg preview` - Preview Data Samples
Preview and visualize your data samples with preprocessing and augmentation applied.

```bash
# Basic data preview
mg preview --config config.yaml

# Preview specific split with custom sample count
mg preview --config config.yaml --num-samples 16 --split val

# Save preview to file
mg preview --config config.yaml --save --output data_samples.png
mg preview --config config.yaml --split train --num-samples 12 --save
```

**Options:**
- `--num-samples`: Number of samples to preview (default: 8)
- `--split {train,val,test}`: Data split to preview (default: train)
- `--save`: Save plot to file instead of displaying
- `--output`: Output file path for saved plot

### Model Training & Evaluation

#### `mg train` - Train Model
Train a model using the specified configuration.

```bash
# Basic training (automatically uses config.yaml in current directory)
mg train

# Basic training with specific config file
mg train --config config.yaml

# Resume training from checkpoint
mg train --resume
mg train --config config.yaml --resume --checkpoint path/to/checkpoint.ckpt
```

**Options:**
- `--config`: Configuration file (optional - searches for config.yaml in current directory if not provided)
- `--resume`: Resume training from checkpoint
- `--checkpoint`: Specific checkpoint file to resume from

#### `mg evaluate` - Evaluate Model
Evaluate a trained model on test/validation data with enhanced auto-discovery and reporting.

```bash
# Auto-discovery evaluation (uses config.yaml and latest model in logs/)
mg evaluate

# Specific configuration file
mg evaluate -c config.yaml

# Specific model with auto-discovered config
mg evaluate -m logs/final_model.keras

# Specific evaluation data with auto-discovery
mg evaluate -d ./test_data

# Combination of specific parameters
mg evaluate -c config.yaml -m logs/model.keras -d ./custom_test_data

# Control output format and saving
mg evaluate -c config.yaml --output-format json
mg evaluate -c config.yaml --no-save  # Don't save to evaluation/ folder
```

**Auto-Discovery Features:**
- **Config Discovery**: Automatically finds `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory structure
- **Latest Model Priority**: Looks for `final_model.keras`, `model.keras`, or `best_model.keras`
- **Report Generation**: Creates timestamped evaluation reports in `evaluation/` folder with JSON and YAML formats

**Options:**
- `--config, -c`: Configuration file (optional - auto-discovers config.yaml)
- `--model-path, -m`: Path to trained model (optional - auto-discovers latest)
- `--data-path, -d`: Path to evaluation data directory (optional - uses config)
- `--output-format`: Report format (yaml or json, default: yaml)
- `--no-save`: Do not save evaluation results to evaluation/ folder

#### `mg predict` - Run Predictions
Run inference on new data using a trained model with intelligent auto-discovery.

```bash
# Full auto-discovery (config, model, and input data)
mg predict

# Auto-discovery with specific input
mg predict -i ./test_images/

# Auto-discovery with custom output format
mg predict --top-k 3 --no-save

# Specific configuration and model
mg predict -c config.yaml -m logs/model.keras -i image.jpg

# Batch prediction with custom settings
mg predict -i ./images/ -o results.json --top-k 5 --batch-size 16

# Custom output directory
mg predict -i ./test_data/ -o my_predictions.json
```

**Auto-Discovery Features:**
- **Config Discovery**: Automatically finds `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory
- **Input Discovery**: Searches for common test directories (`test/`, `test_data/`, `test_images/`, `val/`, `data/test/`, `data/val/`) or image files
- **Report Generation**: Creates timestamped prediction reports in `predictions/` folder with JSON and CSV formats

**Options:**
- `--config, -c`: Configuration file (optional - auto-discovers config.yaml)
- `--input, -i`: Input image file or directory (optional - auto-discovers test data)
- `--model-path, -m`: Path to trained model (optional - auto-discovers latest)
- `--output, -o`: Output file for results (JSON format)
- `--top-k`: Number of top predictions to show (default: 5)
- `--batch-size`: Batch size for processing (default: 32)
- `--no-save`: Do not save prediction results to predictions/ folder

### Model Deployment

#### `mg deploy` - Deploy Model
Deploy models in multiple formats with enhanced auto-discovery, short parameters, and optimization features.

```bash
# Full auto-discovery deployment (config, model, default formats)
mg deploy

# Auto-discovery with specific formats using short parameters
mg deploy -f keras tflite
mg deploy -f onnx -q  # With quantization

# Custom output directory
mg deploy -f keras onnx -o my_deployed_models

# Deploy with quantization and encryption
mg deploy -f tflite -q -e -k myencryptionkey

# Explicit parameters with short options
mg deploy -c config.yaml -m logs/model.keras -f onnx tflite -o production_models

# Combination of auto-discovery and custom settings
mg deploy -f keras tfjs -o web_deployment
```

**Auto-Discovery Features:**
- **Config Discovery**: Automatically finds `config.yaml` in current directory
- **Model Discovery**: Finds the latest versioned model in `logs/` directory
- **Default Formats**: Uses ONNX and TFLite formats if none specified
- **Organized Output**: Creates timestamped deployment directories

**Short Parameters:**
- `-c`: Configuration file (same as `--config`)
- `-m`: Model path (same as `--model-path`)
- `-f`: Output formats (same as `--formats`)
- `-q`: Apply quantization (same as `--quantize`)
- `-e`: Encrypt models (same as `--encrypt`)
- `-k`: Encryption key (same as `--encryption-key`)
- `-o`: Output directory (same as `--output-dir`)

**Options:**
- `--config, -c`: Configuration file (optional - auto-discovers config.yaml)
- `--model-path, -m`: Path to trained model (optional - auto-discovers latest)
- `--formats, -f {onnx,tflite,tfjs,keras}`: Output formats (can specify multiple, default: onnx tflite)
- `--quantize, -q`: Apply quantization for ONNX/TFLite formats
- `--encrypt, -e`: Encrypt model files for security
- `--encryption-key, -k`: Encryption key for model files
- `--output-dir, -o`: Output directory for deployed models (default: deployed_models)

#### `mg models` - List Available Models
List all available built-in model architectures.

```bash
mg models
```

## Quick Start Guide

### 1. Create a New Project
Start by creating a new ModelGardener project with interactive setup:

```bash
# Interactive project creation (recommended for beginners)
mg create my_image_classifier --interactive

# Or create with specific parameters
mg create my_project --model-family efficientnet --num-classes 10 --epochs 50
```

### 2. Prepare Your Data
Organize your data in the following structure:
```
my_project/
├── data/
│   ├── train/
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   └── val/
│       ├── class1/
│       ├── class2/
│       └── ...
```

### 3. Preview Your Data
Visualize your data with preprocessing and augmentation:

```bash
cd my_project
mg preview --config config.yaml --num-samples 16 --save
```

### 4. Configure Your Model (Optional)
Modify the configuration if needed:

```bash
# Interactive configuration
mg config --interactive

# Or modify specific parameters
mg config config.yaml --learning-rate 0.001 --batch-size 32
```

### 5. Validate Configuration
Check your configuration for any issues:

```bash
mg check config.yaml --verbose
```

### 6. Train Your Model
Start training with the configured parameters:

```bash
mg train --config config.yaml
```

### 7. Evaluate Your Model
Evaluate the trained model with auto-discovery:

```bash
# Auto-discovery evaluation (uses config.yaml and latest model)
mg evaluate

# Or with specific parameters and output control
mg evaluate -c config.yaml -m logs/final_model.keras --output-format json
mg evaluate --no-save  # Quick evaluation without saving reports
```

### 8. Make Predictions
Run predictions on new data with intelligent auto-discovery:

```bash
# Full auto-discovery (finds config, model, and test data automatically)
mg predict

# Auto-discovery with specific input
mg predict -i path/to/image.jpg --top-k 3

# Batch prediction with custom output
mg predict -i path/to/images/ -o my_results.json --batch-size 16
```

### 9. Deploy Your Model
Deploy in multiple formats with auto-discovery and short parameters:

```bash
# Auto-discovery deployment with default formats
mg deploy

# Custom formats with short parameters
mg deploy -f onnx tflite -q -o production_models

# Full deployment with encryption
mg deploy -f keras onnx tflite -q -e -k mykey -o secure_deployment
```

## Project Structure

When you create a new project, ModelGardener generates the following structure:

```
my_project/
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and saved models
├── evaluation/         # Auto-generated evaluation reports (JSON/YAML)
├── predictions/        # Auto-generated prediction results (JSON/CSV)
├── deployed_models/    # Auto-generated model deployments
├── custom_modules/     # Custom functions (optional)
│   ├── custom_models.py
│   ├── custom_data_loaders.py
│   ├── custom_loss_functions.py
│   ├── custom_metrics.py
│   ├── custom_callbacks.py
│   ├── custom_preprocessing.py
│   └── custom_augmentations.py
├── config.yaml         # Model configuration
├── train.py           # Training script (auto-generated)
├── evaluation.py      # Evaluation script (auto-generated)
├── prediction.py      # Prediction script (auto-generated)
├── deploy.py          # Deployment script (auto-generated)
├── pyproject.toml     # Python dependencies (auto-generated)
└── README.md          # Project-specific README
```

**Generated Output Directories:**
- `evaluation/`: Contains timestamped evaluation reports with metrics and metadata
- `predictions/`: Contains prediction results in JSON format with optional CSV summaries
- `deployed_models/`: Contains models converted to various formats (ONNX, TFLite, TF.js, Keras)

## Configuration Options

The `config.yaml` file includes comprehensive settings for:

### Model Architecture
- Model family selection (ResNet, EfficientNet, Custom, etc.)
- Model-specific parameters
- Input shape configuration
- Number of classes

### Training Parameters
- Number of epochs
- Learning rate and scheduling
- Batch size
- Optimizer selection (Adam, SGD, etc.)
- Loss function configuration

### Data Pipeline
- Data preprocessing options
- Augmentation strategies
- Dataset splitting
- Data loading parameters

### Runtime Settings
- GPU usage configuration
- Model output directory
- Checkpoint saving
- Logging preferences

### Custom Function Integration
- Custom model architectures
- Custom loss functions
- Custom data loaders
- Custom metrics and callbacks

## Custom Functions

ModelGardener's power lies in its ability to integrate custom functions seamlessly. You can extend any aspect of the training pipeline:

### Available Custom Function Types

1. **Custom Models** (`custom_modules/custom_models.py`)
   - Define your own neural network architectures
   - Support for multi-input/output models
   - 2D and 3D data support

2. **Custom Data Loaders** (`custom_modules/custom_data_loaders.py`)
   - Custom data loading logic
   - Support for various data formats
   - Optimized tf.data pipelines

3. **Custom Loss Functions** (`custom_modules/custom_loss_functions.py`)
   - Implement domain-specific loss functions
   - Multi-task learning support

4. **Custom Metrics** (`custom_modules/custom_metrics.py`)
   - Custom evaluation metrics
   - Real-time monitoring during training

5. **Custom Callbacks** (`custom_modules/custom_callbacks.py`)
   - Custom training callbacks
   - Advanced learning rate scheduling
   - Custom model checkpointing

6. **Custom Preprocessing** (`custom_modules/custom_preprocessing.py`)
   - Domain-specific preprocessing
   - Advanced normalization techniques

7. **Custom Augmentations** (`custom_modules/custom_augmentations.py`)
   - Custom data augmentation strategies
   - Task-specific augmentations

### Function Loading Hierarchy

ModelGardener loads custom functions in the following order:
1. `./custom_modules/` (project-specific functions)
2. `example_funcs/` package (built-in examples)
3. Relative `example_funcs/` path (fallback)

### Example Custom Function

```python
# custom_modules/custom_models.py
import tensorflow as tf

def my_custom_cnn(input_shape, num_classes, **kwargs):
    """Custom CNN architecture."""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```

Then reference it in your `config.yaml`:
```yaml
model:
  custom_model_function: "my_custom_cnn"
  custom_model_file: "custom_modules/custom_models.py"
```

## Generated Scripts

ModelGardener automatically generates Python scripts for your project:

### `train.py`
- Complete training script based on your configuration
- Handles data loading, model building, and training loop
- Supports resuming from checkpoints
- Includes logging and model saving

### `evaluation.py`
- Model evaluation script
- Supports multiple evaluation metrics
- Generates evaluation reports
- Handles test data loading

### `prediction.py`
- Inference script for new data
- Supports single image and batch prediction
- Configurable output formats
- Preprocessing pipeline integration

### `deploy.py`
- Model deployment utilities
- Multi-format export (ONNX, TFLite, TF.js)
- Model optimization and quantization
- API deployment helpers

## Advanced Usage

### Multi-GPU Training
```bash
mg create gpu_project --num-gpus 4 --batch-size 128
mg train --config config.yaml
```

### Custom Training Loops
```python
# custom_modules/custom_training_loops.py
def custom_training_loop(model, train_ds, val_ds, epochs, **kwargs):
    # Your custom training logic here
    pass
```

### Cross-Validation
Configure cross-validation in your `config.yaml`:
```yaml
training:
  use_cross_validation: true
  cv_folds: 5
```

### Model Ensembles
```yaml
model:
  ensemble:
    models: ["efficientnet_b0", "resnet50", "densenet121"]
    weights: [0.4, 0.3, 0.3]
```

## Examples

### Image Classification Project
```bash
# Create a new image classification project
mg create image_classifier --interactive

# Set up your data in data/train/ and data/val/
# Preview your data
mg preview --config config.yaml --num-samples 12 --save

# Train the model
mg train --config config.yaml

# Evaluate and deploy
mg evaluate --config config.yaml
mg deploy --config config.yaml --formats onnx tflite --quantize
```

### Custom Model Training
```bash
# Create project with custom model
mg create custom_project --model-family custom

# Implement your model in custom_modules/custom_models.py
# Update config.yaml to reference your custom function
mg config --interactive

# Train with custom architecture
mg train --config config.yaml
```

### Batch Prediction Pipeline
```bash
# Set up prediction pipeline
mg predict --config config.yaml --input ./test_images/ --batch-size 32 --output predictions.yaml

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   ```bash
   mg check config.yaml --verbose
   ```

2. **Data Loading Issues**
   ```bash
   mg preview --config config.yaml --split train
   ```

3. **GPU Memory Issues**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable mixed precision training

4. **Custom Function Import Errors**
   - Check function signatures match expected format
   - Verify file paths in configuration
   - Use verbose mode for detailed error messages

### Performance Optimization

1. **Data Pipeline Optimization**
   - Enable data prefetching
   - Use tf.data optimizations
   - Consider data caching for repeated training

2. **Model Optimization**
   - Use mixed precision training
   - Enable XLA compilation
   - Optimize model architecture

3. **Training Optimization**
   - Use learning rate scheduling
   - Implement early stopping
   - Use gradient clipping

## Contributing

We welcome contributions to ModelGardener! Please see our contribution guidelines for more information.

### Development Setup
```bash
git clone https://github.com/Yukun-Guo/ModelGardener.git
cd ModelGardener
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Tests
```bash
python test_simple_validation.py
python quick_validation.py
```

## Support & Documentation

- **CLI Help**: Run any command with `--help` for detailed usage information
- **Interactive Mode**: Use `--interactive` flag for guided setup
- **Example Functions**: Check `src/modelgardener/example_funcs/` for reference implementations
- **Configuration Examples**: Use `mg create --interactive` to see all available options
- **Verbose Logging**: Use `--verbose` flags for detailed operation logs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ModelGardener in your research, please cite:

```bibtex
@software{modelgardener,
  title={ModelGardener: A Modular ML Framework for Configuration-Driven Deep Learning},
  author={Yukun Guo},
  year={2025},
  url={https://github.com/Yukun-Guo/ModelGardener}
}
```
