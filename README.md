# test_ - ModelGardener Project

## Project Structure
```
test_/
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and models
├── custom_modules/     # Custom functions (models, losses, etc.)
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

## Configuration Options

The `config.yaml` file includes comprehensive settings for:
- Model architecture selection (ResNet, EfficientNet, Custom, etc.)  
- Training parameters (epochs, learning rate, batch size, etc.)
- Data preprocessing and augmentation options
- Runtime settings (GPU usage, model directory, etc.)
- Custom function integration

## Custom Functions

You can customize any aspect of the training pipeline by creating your own Python files:
1. Create Python files with your custom functions (models, loss functions, etc.)
2. Update the `config.yaml` to reference your custom function files
3. The training scripts will automatically load and use your custom functions

## Need Help?

- Run ModelGardener CLI with `--help` to see all available options
- Use interactive mode for guided configuration: `modelgardener_cli.py config --interactive`
- Check the custom_modules/README.md for detailed examples
- See the ModelGardener documentation for advanced usage
