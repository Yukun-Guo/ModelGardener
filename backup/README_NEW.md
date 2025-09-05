# ModelGardener - CLI-Only Deep Learning Training Tool

**A command-line interface for deep learning model training with TensorFlow/Keras**

## Overview

ModelGardener is a CLI-only tool that provides a streamlined interface for configuring and training deep learning models. It has been refactored to remove all GUI dependencies, making it more stable and suitable for server environments.

## Features

✨ **CLI-Only Interface**: No GUI dependencies - perfect for servers and automated workflows
🔧 **Configuration Management**: YAML-based configuration with automatic parameter extraction
📊 **Multiple Architectures**: Support for ResNet, VGG, DenseNet, EfficientNet, and custom models
🎯 **Custom Functions**: Easy integration of custom models, data loaders, preprocessing, and training loops
🔄 **Project Generation**: Automated generation of complete training projects with sample data
⚡ **TensorFlow Integration**: Built on TensorFlow 2.x with Keras high-level API

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ModelGardener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Create a new project**:
```bash
python main.py create my_classifier
cd my_classifier
```

2. **Configure training** (interactive mode):
```bash
python /path/to/ModelGardener/main.py config --interactive
```

3. **Train the model**:
```bash
python train.py
```

## Key Commands

- `python main.py create <project_name>` - Create a new project
- `python main.py config` - Configure training parameters
- `python main.py train` - Start training
- `python main.py models` - List available model architectures

## Project Structure

When you create a new project, you'll get:

```
my_project/
├── config.yaml                    # Training configuration
├── train.py                      # Training script
├── evaluation.py                 # Model evaluation
├── prediction.py                 # Inference script
├── requirements.txt              # Dependencies
├── data/                         # Sample training data
│   ├── train/
│   └── val/
└── custom_modules/               # Custom function templates
    ├── custom_models.py
    ├── custom_data_loaders.py
    └── README.md
```

## Supported Model Architectures

- **ResNet**: ResNet-50, ResNet-101, ResNet-152
- **VGG**: VGG-16, VGG-19
- **DenseNet**: DenseNet-121, DenseNet-169
- **EfficientNet**: EfficientNetB0, EfficientNetB1
- **Custom Models**: Load your own model architectures

## Configuration

Training configuration is managed through YAML files with automatic parameter extraction:

```yaml
configuration:
  task_type: image_classification
  data:
    train_dir: ./data/train
    val_dir: ./data/val
  model:
    model_family: resnet
    model_name: ResNet-50
    model_parameters:
      input_shape: {height: 224, width: 224, channels: 3}
      classes: 10
  training:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
```

## Custom Functions

Easily integrate custom functionality:

1. **Custom Models**: Define model architectures in `custom_modules/custom_models.py`
2. **Custom Data Loaders**: Custom data loading logic
3. **Custom Preprocessing**: Data preprocessing functions
4. **Custom Training Loops**: Advanced training strategies

## CLI Documentation

For detailed CLI usage, see [CLI_README.md](CLI_README.md)

## Changes in CLI-Only Version

This version has been refactored to remove all PySide6/GUI dependencies:

- ❌ Removed main GUI window
- ❌ Removed interactive parameter trees
- ❌ Removed file dialog boxes
- ❌ Removed progress widgets
- ✅ Retained all core training functionality
- ✅ Maintained custom function support
- ✅ Kept configuration management
- ✅ Enhanced CLI interface

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License.
