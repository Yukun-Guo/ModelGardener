#!/usr/bin/env python3
"""
ModelGardener CLI Entry Point
Provides command-line access to ModelGardener functionality without the GUI.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from cli_config import ModelConfigCLI, create_argument_parser
    from config_manager import ConfigManager
    from enhanced_trainer import EnhancedTrainer
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("💡 Make sure you're running from the ModelGardener directory")
    sys.exit(1)


class ModelGardenerCLI:
    """Main CLI interface for ModelGardener."""
    
    def __init__(self):
        self.config_cli = ModelConfigCLI()
        self.config_manager = ConfigManager()

    def find_config_file(self, directory: str = ".") -> Optional[str]:
        """Find existing config file in the specified directory."""
        config_patterns = ["config.yaml", "config.yml", "model_config.yaml", "model_config.yml", 
                          "config.json", "model_config.json"]
        
        for pattern in config_patterns:
            config_path = Path(directory) / pattern
            if config_path.exists():
                return str(config_path)
        
        return None

    def modify_existing_config(self, config_file: str, interactive: bool = False, **kwargs) -> bool:
        """Modify an existing configuration file."""
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        print(f"🔧 Modifying existing configuration: {config_file}")
        
        # Load existing configuration
        config = self.config_cli.load_config(config_file)
        if not config:
            print("❌ Failed to load existing configuration")
            return False
        
        print("✅ Existing configuration loaded")
        self.config_cli.display_config_summary(config)
        
        if interactive:
            print("\n🔄 Interactive configuration modification")
            print("=" * 50)
            
            # Run interactive configuration with existing config as base
            modified_config = self.config_cli.interactive_configuration_with_existing(config)
        else:
            print("\n⚡ Batch configuration modification")
            
            # Apply batch modifications to existing config
            modified_config = self.config_cli.batch_configuration_with_existing(config, kwargs)
        
        # Validate and save the modified configuration
        if self.config_cli.validate_config(modified_config):
            print("\n📋 Modified Configuration Summary:")
            self.config_cli.display_config_summary(modified_config)
            
            # Determine output format from file extension
            format_type = 'yaml' if config_file.endswith(('.yaml', '.yml')) else 'json'
            
            if self.config_cli.save_config(modified_config, config_file, format_type):
                print(f"✅ Configuration updated successfully: {config_file}")
                return True
            else:
                print("❌ Failed to save configuration")
                return False
        else:
            print("❌ Modified configuration validation failed")
            return False

    def run_training(self, config_file: str, **kwargs):
        """Run training using CLI configuration."""
        print(f"🚀 Starting ModelGardener training from CLI")
        print(f"📄 Configuration: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            # Validate configuration
            if not self.config_cli.validate_config(config):
                print("❌ Configuration validation failed")
                return False
            
            # Extract the main configuration
            main_config = config.get('configuration', {})
            
            print("✅ Configuration loaded and validated")
            self.config_cli.display_config_summary(config)
            
            # Initialize trainer
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Run training
            print("\n🏃 Starting training...")
            success = trainer.train()
            
            if success:
                print("✅ Training completed successfully!")
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def run_evaluation(self, config_file: str, model_path: str = None):
        """Run model evaluation using CLI."""
        print(f"📊 Starting ModelGardener evaluation from CLI")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            main_config = config.get('configuration', {})
            
            # Use provided model path or default from config
            if model_path:
                main_config['runtime']['model_dir'] = model_path
            
            print("✅ Configuration loaded for evaluation")
            
            # Initialize trainer for evaluation
            trainer = EnhancedTrainer(
                config=main_config,
                custom_functions=config.get('metadata', {}).get('custom_functions', {})
            )
            
            # Run evaluation
            print("\n📈 Starting evaluation...")
            results = trainer.evaluate()
            
            if results:
                print("✅ Evaluation completed successfully!")
                print("\n📊 Results:")
                for metric, value in results.items():
                    print(f"  {metric}: {value}")
                return True
            else:
                print("❌ Evaluation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def list_available_models(self):
        """List all available models."""
        print("🤖 Available Models in ModelGardener")
        print("=" * 50)
        
        for family, models in self.config_cli.available_models.items():
            print(f"\n📂 {family.upper()}")
            for model in models:
                print(f"  • {model}")

    def create_project_template(self, project_name: str = None, project_dir: str = ".", interactive: bool = False, **kwargs):
        """Create a new project template with CLI configuration."""
        
        # Handle case where no project name is provided - use current directory
        if not project_name:
            project_path = Path(project_dir).resolve()
            project_name = project_path.name
            print(f"🌱 Creating ModelGardener project in current directory: {project_name}")
            print(f"📁 Project directory: {project_path}")
        else:
            project_path = Path(project_dir) / project_name
            project_path.mkdir(parents=True, exist_ok=True)
            print(f"🌱 Creating ModelGardener project: {project_name}")
            print(f"📁 Project directory: {project_path}")

        # Create project structure
        (project_path / "data" / "train").mkdir(parents=True, exist_ok=True)
        (project_path / "data" / "val").mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)

        # Create configuration based on mode
        config_file = project_path / "config.yaml"

        if interactive:
            print("\n🔧 Interactive project configuration")
            print("=" * 50)

            # Update paths to be relative to the project directory
            old_cwd = os.getcwd()
            os.chdir(project_path)

            try:
                config = self.config_cli.interactive_configuration()

                # Validate and save configuration
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, str(config_file), 'yaml'):
                        print(f"✅ Configuration saved to {config_file}")
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(str(config_file), 'yaml')
            finally:
                os.chdir(old_cwd)
        else:
            # Batch mode: create config using provided arguments or default template
            if kwargs:
                # Create a default config first, then apply batch modifications
                default_config = self.config_cli.create_default_config()
                config = self.config_cli.batch_configuration_with_existing(default_config, kwargs)
                if self.config_cli.validate_config(config):
                    self.config_cli.display_config_summary(config)
                    if self.config_cli.save_config(config, str(config_file), 'yaml'):
                        print(f"✅ Configuration saved to {config_file}")
                    else:
                        print("❌ Failed to save configuration, using default template")
                        self.config_cli.create_template(str(config_file), 'yaml')
                else:
                    print("❌ Configuration validation failed, using default template")
                    self.config_cli.create_template(str(config_file), 'yaml')
            else:
                # No batch args, create default template
                self.config_cli.create_template(str(config_file), 'yaml')
        
        # Create README
        readme_content = f"""# {project_name} - ModelGardener Project

## Project Structure
```
{project_name}/
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
"""
        
        readme_file = project_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Project template created successfully!")
        print(f"✅ Project template created successfully!")
        print(f"📖 See {readme_file} for instructions")

    def check_configuration(self, config_file: str, verbose: bool = False) -> bool:
        """Check and validate a configuration file."""
        print(f"🔍 Checking configuration file: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration
            config = self.config_cli.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration file")
                return False
            
            if verbose:
                print(f"📄 Configuration file format: {'YAML' if config_file.endswith(('.yaml', '.yml')) else 'JSON'}")
                print(f"📊 Configuration size: {len(str(config))} characters")
            
            # Validate configuration
            is_valid = self.config_cli.validate_config(config)
            
            if verbose and is_valid:
                # Display configuration summary
                print("\n📋 Configuration Summary:")
                self.config_cli.display_config_summary(config)
            
            if is_valid:
                print("✅ Configuration file is valid!")
            else:
                print("❌ Configuration file validation failed")
            
            return is_valid
            
        except Exception as e:
            print(f"❌ Error validating configuration: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False


def create_main_argument_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="ModelGardener CLI - Train and manage deep learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  config      Modify existing model configuration files
  train       Train a model
  evaluate    Evaluate a trained model
  models      List available models
  create      Create a new project template
  check       Check configuration files

Examples:
  # Create a new project with interactive setup
  modelgardener_cli.py create my_project --interactive
  modelgardener_cli.py create my_project --dir /path/to/workspace
  modelgardener_cli.py create --interactive  # Create in current directory
  modelgardener_cli.py create  # Create basic template in current directory
  
  # Modify existing configuration files
  modelgardener_cli.py config config.yaml --interactive
  modelgardener_cli.py config --interactive  # Auto-finds config in current dir
  modelgardener_cli.py config config.yaml --epochs 100 --learning-rate 0.01
  
  # Check configuration files
  modelgardener_cli.py check config.yaml
  modelgardener_cli.py check config.json --verbose
  
  # Train a model
  modelgardener_cli.py train --config config.yaml
  
  # Evaluate the trained model
  modelgardener_cli.py evaluate --config config.yaml
  
  # List available models
  modelgardener_cli.py models
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Modify existing model configuration')
    config_parser.add_argument('config_file', nargs='?', help='Existing configuration file to modify (optional - will search current directory)')
    config_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive configuration modification mode')
    config_parser.add_argument('--format', '-f', choices=['json', 'yaml'], help='Output format (inferred from file extension if not specified)')
    
    # Add configuration modification arguments for batch mode
    config_parser.add_argument('--train-dir', type=str, help='Training data directory')
    config_parser.add_argument('--val-dir', type=str, help='Validation data directory')
    config_parser.add_argument('--batch-size', type=int, help='Batch size')
    config_parser.add_argument('--model-family', help='Model family')
    config_parser.add_argument('--model-name', type=str, help='Model name')
    config_parser.add_argument('--num-classes', type=int, help='Number of classes')
    config_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    config_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    config_parser.add_argument('--optimizer', help='Optimizer')
    config_parser.add_argument('--loss-function', help='Loss function')
    config_parser.add_argument('--model-dir', type=str, help='Model output directory')
    config_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    train_parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume from')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', '-c', type=str, required=True, help='Configuration file')
    eval_parser.add_argument('--model-path', type=str, help='Path to trained model')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new project template')
    create_parser.add_argument('project_name', nargs='?', default=None, help='Name of the project (optional - uses current directory name if not provided)')
    create_parser.add_argument('--dir', '-d', default='.', help='Directory to create project in (ignored if no project_name provided)')
    create_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive project creation mode')
    # Add configuration arguments for batch mode (same as config)
    create_parser.add_argument('--train-dir', type=str, help='Training data directory')
    create_parser.add_argument('--val-dir', type=str, help='Validation data directory')
    create_parser.add_argument('--batch-size', type=int, help='Batch size')
    create_parser.add_argument('--model-family', help='Model family')
    create_parser.add_argument('--model-name', type=str, help='Model name')
    create_parser.add_argument('--num-classes', type=int, help='Number of classes')
    create_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    create_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    create_parser.add_argument('--optimizer', help='Optimizer')
    create_parser.add_argument('--loss-function', help='Loss function')
    create_parser.add_argument('--model-dir', type=str, help='Model output directory')
    create_parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check configuration file')
    check_parser.add_argument('config_file', help='Configuration file to check')
    check_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed validation results')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_main_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ModelGardenerCLI()
    
    try:
        if args.command == 'config':
            # Handle configuration command - only modify existing configs
            config_file = args.config_file
            
            # If no config file specified, try to find one in current directory
            if not config_file:
                config_file = cli.find_config_file(".")
                if not config_file:
                    print("❌ No configuration file found in current directory.")
                    print("💡 Expected files: config.yaml, config.yml, model_config.yaml, model_config.yml, config.json, or model_config.json")
                    print("💡 Use 'create' command to create a new project with configuration.")
                    return
                else:
                    print(f"🔍 Found configuration file: {config_file}")
            
            # Prepare kwargs for batch mode
            kwargs = {}
            if hasattr(args, 'train_dir') and args.train_dir:
                kwargs['train_dir'] = args.train_dir
            if hasattr(args, 'val_dir') and args.val_dir:
                kwargs['val_dir'] = args.val_dir
            if hasattr(args, 'batch_size') and args.batch_size:
                kwargs['batch_size'] = args.batch_size
            if hasattr(args, 'model_family') and args.model_family:
                kwargs['model_family'] = args.model_family
            if hasattr(args, 'model_name') and args.model_name:
                kwargs['model_name'] = args.model_name
            if hasattr(args, 'num_classes') and args.num_classes:
                kwargs['num_classes'] = args.num_classes
            if hasattr(args, 'epochs') and args.epochs:
                kwargs['epochs'] = args.epochs
            if hasattr(args, 'learning_rate') and args.learning_rate:
                kwargs['learning_rate'] = args.learning_rate
            if hasattr(args, 'optimizer') and args.optimizer:
                kwargs['optimizer'] = args.optimizer
            if hasattr(args, 'loss_function') and args.loss_function:
                kwargs['loss_function'] = args.loss_function
            if hasattr(args, 'model_dir') and args.model_dir:
                kwargs['model_dir'] = args.model_dir
            if hasattr(args, 'num_gpus') and args.num_gpus is not None:
                kwargs['num_gpus'] = args.num_gpus
            
            # Modify the existing configuration
            success = cli.modify_existing_config(config_file, args.interactive, **kwargs)
            if not success:
                sys.exit(1)
        
        elif args.command == 'train':
            success = cli.run_training(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'evaluate':
            success = cli.run_evaluation(args.config, args.model_path)
            sys.exit(0 if success else 1)
        
        elif args.command == 'models':
            cli.list_available_models()
        
        elif args.command == 'create':
            # Prepare kwargs for batch mode
            kwargs = {}
            if hasattr(args, 'train_dir') and args.train_dir:
                kwargs['train_dir'] = args.train_dir
            if hasattr(args, 'val_dir') and args.val_dir:
                kwargs['val_dir'] = args.val_dir
            if hasattr(args, 'batch_size') and args.batch_size:
                kwargs['batch_size'] = args.batch_size
            if hasattr(args, 'model_family') and args.model_family:
                kwargs['model_family'] = args.model_family
            if hasattr(args, 'model_name') and args.model_name:
                kwargs['model_name'] = args.model_name
            if hasattr(args, 'num_classes') and args.num_classes:
                kwargs['num_classes'] = args.num_classes
            if hasattr(args, 'epochs') and args.epochs:
                kwargs['epochs'] = args.epochs
            if hasattr(args, 'learning_rate') and args.learning_rate:
                kwargs['learning_rate'] = args.learning_rate
            if hasattr(args, 'optimizer') and args.optimizer:
                kwargs['optimizer'] = args.optimizer
            if hasattr(args, 'loss_function') and args.loss_function:
                kwargs['loss_function'] = args.loss_function
            if hasattr(args, 'model_dir') and args.model_dir:
                kwargs['model_dir'] = args.model_dir
            if hasattr(args, 'num_gpus') and args.num_gpus is not None:
                kwargs['num_gpus'] = args.num_gpus
            cli.create_project_template(args.project_name, args.dir, args.interactive, **kwargs)
        
        elif args.command == 'check':
            success = cli.check_configuration(args.config_file, args.verbose)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚡ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
