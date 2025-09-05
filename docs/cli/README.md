# CLI Reference

ModelGardener provides a comprehensive command-line interface for all machine learning operations. This section documents each CLI command with detailed usage examples and options.

## Available Commands

| Command | Purpose | Status |
|---------|---------|--------|
| [`config`](config.md) | Modify existing configuration files | ✅ Available |
| [`create`](create.md) | Create new project templates | ✅ Available |
| [`train`](train.md) | Train machine learning models | ✅ Available |
| [`evaluate`](evaluate.md) | Evaluate trained models | ✅ Available |
| [`predict`](predict.md) | Run predictions on new data | ✅ Available |
| [`deploy`](deploy.md) | Deploy models in multiple formats | ✅ Available |
| [`models`](models.md) | List available model architectures | ✅ Available |
| [`check`](check.md) | Validate configuration files | ✅ Available |

## Quick Start

```bash
# Get help for any command
modelgardener_cli.py --help
modelgardener_cli.py <command> --help

# Basic workflow
modelgardener_cli.py create my_project --interactive
cd my_project
modelgardener_cli.py train --config config.yaml
modelgardener_cli.py evaluate --config config.yaml
modelgardener_cli.py predict --config config.yaml --input test_image.jpg
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
modelgardener_cli.py create image_classifier --interactive

# Navigate to project directory
cd image_classifier

# Train the model
modelgardener_cli.py train --config config.yaml

# Evaluate the model
modelgardener_cli.py evaluate --config config.yaml --output-format json

# Make predictions
modelgardener_cli.py predict --config config.yaml --input ./test_images/
```

### Advanced Usage

```bash
# Create project with specific parameters
modelgardener_cli.py create advanced_project \
    --model-family resnet \
    --num-classes 100 \
    --epochs 200 \
    --learning-rate 0.001

# Evaluate with custom data and format
modelgardener_cli.py evaluate \
    --config config.yaml \
    --data-path ./custom_test_data \
    --output-format json \
    --model-path ./custom_models/

# Deploy with multiple formats and security
modelgardener_cli.py deploy \
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
python train.py                    # Same as: modelgardener_cli.py train
python evaluation.py               # Same as: modelgardener_cli.py evaluate
python prediction.py --input data/ # Same as: modelgardener_cli.py predict
python deploy.py --port 8080       # Same as: modelgardener_cli.py deploy
```

## Next Steps

- Start with the [Quick Start Tutorial](../tutorials/quickstart.md)
- Learn about [Custom Functions](../tutorials/custom-functions.md)
- Explore [Advanced Configuration](../tutorials/advanced-configuration.md)
- See [Production Deployment](../tutorials/production-deployment.md) examples

---

For detailed documentation of each command, click on the command links above.
