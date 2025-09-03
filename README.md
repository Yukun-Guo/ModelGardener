# . - ModelGardener Project

## Project Structure
```
./
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and models
├── custom_modules/     # Custom function templates (auto-generated)
├── config.yaml         # Model configuration
├── train.py           # Training script (auto-generated)
├── evaluation.py      # Evaluation script (auto-generated)
├── prediction.py      # Prediction script (auto-generated)
├── deploy.py          # Deployment script (auto-generated)
├── requirements.txt   # Python dependencies (auto-generated)
└── README.md          # This file
```

## Quick Start

### 1. Prepare Your Data
Place your training images in `data/train/` and validation images in `data/val/`

### 2. Configure Your Model
Edit the `config.yaml` file to customize your model settings, or use the interactive configuration:
```bash
# Interactive configuration (overwrites config.yaml)
python /path/to/ModelGardener/modelgardener_cli.py config --interactive --output config.yaml

# Or directly edit config.yaml
```

### 3. Train Your Model
```bash
# Use the generated training script
python train.py

# Or use the CLI
python /path/to/ModelGardener/modelgardener_cli.py train --config config.yaml
```

### 4. Evaluate Your Model
```bash
# Use the generated evaluation script  
python evaluation.py

# Or use the CLI
python /path/to/ModelGardener/modelgardener_cli.py evaluate --config config.yaml --model-path logs/final_model.keras
```

## Generated Files

This project includes auto-generated files to help you get started:

- **config.yaml** - Complete model configuration with examples and documentation
- **train.py** - Ready-to-use training script
- **evaluation.py** - Model evaluation script
- **prediction.py** - Inference script for new data
- **deploy.py** - Deployment utilities
- **custom_modules/** - Template files for custom functions:
  - `custom_models.py` - Custom model architectures
  - `custom_data_loaders.py` - Custom data loading functions
  - `custom_loss_functions.py` - Custom loss functions
  - `custom_optimizers.py` - Custom optimizers
  - `custom_metrics.py` - Custom metrics
  - `custom_callbacks.py` - Custom training callbacks
  - `custom_augmentations.py` - Custom data augmentation
  - `custom_preprocessing.py` - Custom preprocessing functions
  - `custom_training_loops.py` - Custom training strategies

## Configuration Options

The `config.yaml` file includes comprehensive settings for:
- Model architecture selection (ResNet, EfficientNet, Custom, etc.)
- Training parameters (epochs, learning rate, batch size, etc.)
- Data preprocessing and augmentation options
- Runtime settings (GPU usage, model directory, etc.)
- Custom function integration

## Custom Functions

You can customize any aspect of the training pipeline by editing the files in `custom_modules/`:
1. Edit the template functions to implement your custom logic
2. Update the `config.yaml` to reference your custom functions
3. The training scripts will automatically load and use your custom functions

## Need Help?

- Run ModelGardener CLI with `--help` to see all available options
- Use interactive mode for guided configuration: `modelgardener_cli.py config --interactive`
- Check the custom_modules/README.md for detailed examples
- See the ModelGardener documentation for advanced usage
