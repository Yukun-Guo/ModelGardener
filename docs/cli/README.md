# CLI Reference

ModelGardener provides a comprehensive command-line interface with intelligent auto-discovery features for all machine learning operations. This section documents each CLI command with detailed usage examples and options.

## 🌟 Key Features

- **Auto-Discovery**: Intelligent file detection eliminates repetitive parameter specification
- **Short Parameters**: Intuitive short forms for all major options (`-c`, `-m`, `-i`, `-o`, etc.)
- **Enhanced Reporting**: Automatic generation of timestamped reports with comprehensive metadata
- **Flexible Workflows**: Mix auto-discovery with explicit parameters as needed

## Available Commands

| Command | Purpose | Auto-Discovery | Status |
|---------|---------|----------------|--------|
| [`config`](config.md) | Modify existing configuration files | ❌ | ✅ Available |
| [`create`](create.md) | Create new project templates | ❌ | ✅ Available |
| [`train`](train.md) | Train machine learning models | ⚡ Config | ✅ Available |
| [`evaluate`](evaluate.md) | Evaluate trained models | ⚡ Config + Model + Data | ✅ Enhanced |
| [`predict`](predict.md) | Run predictions on new data | ⚡ Config + Model + Input | ✅ Enhanced |
| [`deploy`](deploy.md) | Deploy models in multiple formats | ⚡ Config + Model | ✅ Enhanced |
| [`models`](models.md) | List available model architectures | ❌ | ✅ Available |
| [`check`](check.md) | Validate configuration files | ⚡ Config | ✅ Available |
| [`preview`](preview.md) | Preview data with preprocessing/augmentation | ⚡ Config | ✅ Available |

## Auto-Discovery Guide

For comprehensive information about ModelGardener's intelligent auto-discovery features, see the [Auto-Discovery Guide](auto-discovery.md).

### Quick Examples

```bash
# Full auto-discovery workflows (recommended)
mg evaluate                    # Finds config.yaml and latest model
mg predict                     # Finds config, model, and test data
mg deploy                      # Finds config and model, uses default formats

# Mix auto-discovery with custom parameters
mg evaluate -d ./custom_test_data     # Auto-discover config+model, specify data
mg predict -i ./my_images/ -o results.json  # Auto-discover config+model, specify input+output
mg deploy -f onnx tflite -o production/     # Auto-discover config+model, specify formats+output
```

## Quick Start

```bash
# Get help for any command
mg --help
mg <command> --help

# Enhanced workflow with auto-discovery
mg create my_project --interactive
cd my_project
mg preview                     # Auto-discovers config.yaml
mg train                       # Auto-discovers config.yaml
mg evaluate --config config.yaml
mg predict --config config.yaml --input test_image.jpg
mg deploy --config config.yaml --formats onnx tflite
```

## Global Options

All commands support these global options:

- `--help` / `-h`: Show help message
- `--version`: Show version information

## Command Categories

### Project Management
- **[`create`](create.md)**: Initialize new ML projects with templates
- **[`config`](config.md)**: Modify and validate configuration files
- **[`check`](check.md)**: Validate configuration file syntax and structure

### Data Operations
- **[`preview`](preview.md)**: Preview data samples with preprocessing and augmentation visualization

### Model Operations
- **[`train`](train.md)**: Train models with comprehensive pipeline
- **[`evaluate`](evaluate.md)**: Comprehensive model evaluation with metrics
- **[`predict`](predict.md)**: Run predictions on single images or batches
- **[`deploy`](deploy.md)**: Deploy models in production formats

### Information
- **[`models`](models.md)**: Browse available model architectures

## Configuration File Integration

All commands work with YAML configuration files that define:

- Model architecture and parameters
- Training settings and hyperparameters
- Data loading and preprocessing
- Custom functions and callbacks
- Runtime and deployment settings

See [Configuration Guide](../tutorials/configuration.md) for detailed information.

## Error Handling

The CLI provides comprehensive error handling with:

- Clear error messages with suggested solutions
- Graceful fallbacks for missing dependencies
- Validation of inputs and configurations
- Progress tracking and status updates

## Examples

### Basic Project Creation and Training

```bash
# Create a new project
mg create image_classifier --interactive

# Navigate to project directory
cd image_classifier

# Train the model
mg train --config config.yaml

# Evaluate the model
mg evaluate --config config.yaml --output-format json

# Make predictions
mg predict --config config.yaml --input ./test_images/
```

### Advanced Usage

```bash
# Create project with specific parameters
mg create advanced_project \
    --model-family resnet \
    --num-classes 100 \
    --epochs 200 \
    --learning-rate 0.001

# Evaluate with custom data and format
mg evaluate \
    --config config.yaml \
    --data-path ./custom_test_data \
    --output-format json \
    --model-path ./custom_models/

# Deploy with multiple formats and security
mg deploy \
    --config config.yaml \
    --formats onnx tflite tfjs \
    --quantize \
    --encrypt \
    --encryption-key mySecretKey
```

## Integration with Generated Scripts

Each command can also be executed using the generated Python scripts:

```bash
# Equivalent operations using generated scripts
python train.py                    # Same as: mg train
python evaluation.py               # Same as: mg evaluate
python prediction.py --input data/ # Same as: mg predict
python deploy.py --port 8080       # Same as: mg deploy
```

## Next Steps

- Start with the [Quick Start Tutorial](../tutorials/quickstart.md)
- Learn about [Custom Functions](../tutorials/custom-functions.md)
- Explore [Advanced Configuration](../tutorials/advanced-configuration.md)
- See [Production Deployment](../tutorials/production-deployment.md) examples

---

For detailed documentation of each command, click on the command links above.
