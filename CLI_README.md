# ModelGardener CLI Interface Documentation

The ModelGardener CLI interface provides a command-line alternative to the PySide6 GUI, allowing you to configure and run model training without the risk of GUI crashes.

## Overview

The CLI interface consists of two main components:
- `modelgardener_cli.py` - Main CLI entry point with subcommands
- `cli_config.py` - Configuration management tool (can be used standalone)

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

### `create` - Project Template

Create a new project template with proper directory structure.

```bash
python modelgardener_cli.py create my_project --dir ./projects
```

Options:
- `project_name`: Name of the project (required)
- `--dir`: Directory to create the project in (default: current directory)

## Configuration File Structure

The CLI generates configuration files with the following structure:

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
      }
    },
    "training": {
      "epochs": 100,
      "initial_learning_rate": 0.1
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

## Advanced Configuration

### Custom Functions

The CLI supports custom functions (models, loss functions, metrics, etc.) through the configuration file. Custom functions can be specified in the `metadata.custom_functions` section:

```json
{
  "metadata": {
    "custom_functions": {
      "models": [
        {
          "name": "CustomModel",
          "file_path": "./custom_functions/my_model.py",
          "function_name": "create_custom_model"
        }
      ],
      "loss_functions": [
        {
          "name": "CustomLoss",
          "file_path": "./custom_functions/my_losses.py",
          "function_name": "custom_loss"
        }
      ]
    }
  }
}
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
   python modelgardener_cli.py config --validate --config config.json
   ```

3. **Path Issues**
   Always use absolute paths or ensure you're running from the correct directory.

4. **GPU Configuration**
   If you encounter GPU-related errors, try setting `--num-gpus 0` for CPU-only training.

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

## Contributing

To extend the CLI interface:

1. Add new command-line options in the argument parser
2. Implement the corresponding logic in the CLI classes
3. Update the configuration structure as needed
4. Add tests and documentation

## Support

If you encounter issues with the CLI interface:

1. Check this documentation
2. Validate your configuration file
3. Ensure all dependencies are installed
4. Check file paths and permissions
5. Review the error messages for specific guidance
