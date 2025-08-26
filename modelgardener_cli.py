#!/usr/bin/env python3
"""
ModelGardener CLI Entry Point
Provides command-line access to ModelGardener functionality without the GUI.
"""

import argparse
import sys
import os
from pathlib import Path

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

    def create_project_template(self, project_name: str, project_dir: str = "."):
        """Create a new project template with CLI configuration."""
        project_path = Path(project_dir) / project_name
        project_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🌱 Creating ModelGardener project: {project_name}")
        print(f"📁 Project directory: {project_path}")
        
        # Create project structure
        (project_path / "data" / "train").mkdir(parents=True, exist_ok=True)
        (project_path / "data" / "val").mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        (project_path / "configs").mkdir(exist_ok=True)
        (project_path / "custom_functions").mkdir(exist_ok=True)
        
        # Create default configuration
        config_file = project_path / "configs" / "model_config.json"
        self.config_cli.create_template(str(config_file))
        
        # Create README
        readme_content = f"""# {project_name} - ModelGardener Project

## Project Structure
```
{project_name}/
├── data/
│   ├── train/          # Training data
│   └── val/            # Validation data
├── logs/               # Training logs and models
├── configs/            # Configuration files
│   └── model_config.json
├── custom_functions/   # Custom functions (models, losses, etc.)
└── README.md          # This file
```

## Quick Start

### 1. Prepare Your Data
Place your training images in `data/train/` and validation images in `data/val/`

### 2. Configure Your Model
```bash
# Interactive configuration
python /path/to/ModelGardener/modelgardener_cli.py config --interactive --output configs/model_config.json

# Or edit the template configuration file
# configs/model_config.json
```

### 3. Train Your Model
```bash
python /path/to/ModelGardener/modelgardener_cli.py train --config configs/model_config.json
```

### 4. Evaluate Your Model
```bash
python /path/to/ModelGardener/modelgardener_cli.py evaluate --config configs/model_config.json --model-path logs/final_model.keras
```

## Configuration Options

Use the CLI configuration tool to set up your model:
- Task type (image classification, object detection, etc.)
- Model architecture (ResNet, EfficientNet, etc.)
- Training parameters (epochs, learning rate, etc.)
- Data preprocessing options
- Runtime settings (GPU usage, model directory, etc.)

## Custom Functions

You can add custom models, loss functions, metrics, etc. in the `custom_functions/` directory.
See the ModelGardener documentation for examples.

## Need Help?

- Run with `--help` to see all available options
- Check the ModelGardener documentation
- Use interactive mode for guided configuration
"""
        
        readme_file = project_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Project template created successfully!")
        print(f"📖 See {readme_file} for instructions")


def create_main_argument_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="ModelGardener CLI - Train and manage deep learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  config      Configure model settings
  train       Train a model
  evaluate    Evaluate a trained model
  models      List available models
  create      Create a new project template

Examples:
  # Create and configure a new project
  modelgardener_cli.py create my_project
  modelgardener_cli.py config --interactive --output my_project/configs/model_config.json
  
  # Train a model
  modelgardener_cli.py train --config my_project/configs/model_config.json
  
  # Evaluate the trained model
  modelgardener_cli.py evaluate --config my_project/configs/model_config.json
  
  # Quick configuration and training
  modelgardener_cli.py config --train-dir ./data/train --val-dir ./data/val --epochs 50 --output config.json
  modelgardener_cli.py train --config config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configure model settings')
    config_parser.add_argument('--config', '-c', type=str, help='Load existing configuration file')
    config_parser.add_argument('--output', '-o', type=str, default='model_config.json', help='Output configuration file')
    config_parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    config_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive configuration mode')
    config_parser.add_argument('--template', '-t', action='store_true', help='Create configuration template')
    config_parser.add_argument('--validate', '-v', action='store_true', help='Validate configuration file')
    
    # Add all the configuration arguments
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
    create_parser.add_argument('project_name', help='Name of the project')
    create_parser.add_argument('--dir', '-d', default='.', help='Directory to create project in')
    
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
            # Handle configuration command
            config_cli = ModelConfigCLI()
            
            if args.template:
                config_cli.create_template(args.output)
                return
            
            if args.validate:
                if not args.config:
                    print("❌ --validate requires --config to specify the file to validate")
                    return
                config = config_cli.load_config(args.config)
                if config:
                    config_cli.validate_config(config)
                return
            
            # Load existing configuration if specified
            config = None
            if args.config:
                config = config_cli.load_config(args.config)
                if not config:
                    print("❌ Failed to load configuration, creating new one")
            
            # Interactive or batch mode
            if args.interactive:
                if config:
                    print("🔄 Loaded existing configuration")
                    config_cli.display_config_summary(config)
                config = config_cli.interactive_configuration()
            else:
                config = config_cli.batch_configuration(args)
            
            # Validate and save
            if config_cli.validate_config(config):
                config_cli.display_config_summary(config)
                if config_cli.save_config(config, args.output, args.format):
                    print(f"\n🎉 Configuration saved to {args.output}")
        
        elif args.command == 'train':
            success = cli.run_training(args.config)
            sys.exit(0 if success else 1)
        
        elif args.command == 'evaluate':
            success = cli.run_evaluation(args.config, args.model_path)
            sys.exit(0 if success else 1)
        
        elif args.command == 'models':
            cli.list_available_models()
        
        elif args.command == 'create':
            cli.create_project_template(args.project_name, args.dir)
    
    except KeyboardInterrupt:
        print("\n\n⚡ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
