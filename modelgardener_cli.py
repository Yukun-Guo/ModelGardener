#!/usr/bin/env python3
"""
ModelGardener CLI - Main entry point using modular configuration.
Provides complete CLI access to ModelGardener functionality.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import CLIInterface
    from script_generator import ScriptGenerator
    from enhanced_trainer import EnhancedTrainer
    from config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("💡 Make sure you're running from the ModelGardener directory")
    sys.exit(1)


class ModelGardenerCLI:
    """Main CLI interface for ModelGardener using modular configuration."""
    
    def __init__(self):
        self.cli_interface = CLIInterface()
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

    def run_training(self, config_file: str, resume: bool = False, checkpoint: str = None) -> bool:
        """Run training using CLI configuration."""
        print(f"🚀 Starting ModelGardener training from CLI")
        print(f"📄 Configuration: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration using modular interface
            config = self.cli_interface.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            # Validate configuration
            try:
                self.cli_interface._validate_configuration(config)
                print("✅ Configuration loaded and validated")
            except Exception as e:
                print(f"❌ Configuration validation failed: {e}")
                return False
            print("\n📋 Configuration Summary:")
            print("=" * 50)
            self.cli_interface.print_configuration_summary(config)
            
            # Initialize trainer with resume options
            trainer = EnhancedTrainer(config=config)
            
            if resume and checkpoint:
                print(f"🔄 Resuming training from checkpoint: {checkpoint}")
                # TODO: Implement checkpoint loading
            elif resume:
                print("🔄 Resuming training from latest checkpoint")
                # TODO: Implement auto-checkpoint detection
            
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

    def run_evaluation(self, config_file: str, model_path: str = None) -> bool:
        """Run model evaluation using CLI."""
        print(f"📊 Starting ModelGardener evaluation from CLI")
        print(f"📄 Configuration: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration using modular interface
            config = self.cli_interface.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration")
                return False
            
            # Override model path if provided
            if model_path:
                if 'runtime' not in config:
                    config['runtime'] = {}
                config['runtime']['model_dir'] = model_path
                print(f"🔄 Using model path: {model_path}")
            
            print("✅ Configuration loaded for evaluation")
            
            # Initialize trainer for evaluation
            trainer = EnhancedTrainer(config=config)
            
            # Run evaluation
            print("\n📈 Starting evaluation...")
            results = trainer.evaluate()
            
            if results:
                print("✅ Evaluation completed successfully!")
                print("\n📊 Results:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
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
        
        # Get available models from the modular configuration
        from config.model_config import ModelConfig
        model_config = ModelConfig()
        
        # Access the available models (this might need to be adjusted based on actual implementation)
        available_models = {
            'resnet': ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'],
            'efficientnet': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4'],
            'vgg': ['VGG16', 'VGG19'],
            'mobilenet': ['MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large'],
            'densenet': ['DenseNet121', 'DenseNet169', 'DenseNet201'],
            'inception': ['InceptionV3', 'InceptionResNetV2'],
            'custom': ['Custom models from your files']
        }
        
        for family, models in available_models.items():
            print(f"\n📂 {family.upper()}")
            for model in models:
                print(f"  • {model}")

    def check_configuration(self, config_file: str, verbose: bool = False) -> bool:
        """Check and validate a configuration file."""
        print(f"🔍 Checking configuration file: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        try:
            # Load configuration using modular interface
            config = self.cli_interface.load_config(config_file)
            if not config:
                print("❌ Failed to load configuration file")
                return False
            
            if verbose:
                print(f"📄 Configuration file format: {'YAML' if config_file.endswith(('.yaml', '.yml')) else 'JSON'}")
                print(f"📊 Configuration size: {len(str(config))} characters")
            
            # Validate configuration
            try:
                self.cli_interface._validate_configuration(config)
                is_valid = True
            except Exception as e:
                is_valid = False
                print(f"Validation error: {e}")
            
            if verbose and is_valid:
                # Display configuration summary
                print("\n📋 Configuration Summary:")
                print("=" * 50)
                self.cli_interface.print_configuration_summary(config)
            
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

    def modify_existing_config(self, config_file: str, interactive: bool = False, **kwargs) -> bool:
        """Modify an existing configuration file."""
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        print(f"🔧 Modifying existing configuration: {config_file}")
        
        # Load existing configuration
        config = self.cli_interface.load_config(config_file)
        if not config:
            print("❌ Failed to load existing configuration")
            return False
        
        print("✅ Existing configuration loaded")
        print("\n📋 Current Configuration Summary:")
        print("=" * 50)
        print(self.cli_interface.generate_config_summary(config))
        
        if interactive:
            print("\n🔄 Interactive configuration modification")
            print("=" * 50)
            
            # Use the existing interactive configuration but pre-populate with current config
            # For now, just run normal interactive configuration
            modified_config = self.cli_interface.interactive_configuration()
        else:
            print("\n⚡ Batch configuration modification")
            
            # Apply batch modifications to existing config
            modified_config = self.apply_batch_modifications(config, kwargs)
        
        # Validate and save the modified configuration
        try:
            self.cli_interface._validate_configuration(modified_config)
            print("\n📋 Modified Configuration Summary:")
            print("=" * 50)
            print(self.cli_interface.generate_config_summary(modified_config))
            
            # Determine output format from file extension
            format_type = 'yaml' if config_file.endswith(('.yaml', '.yml')) else 'json'
            
            if self.cli_interface.save_config(modified_config, config_file, format_type):
                print(f"✅ Configuration updated successfully: {config_file}")
                return True
            else:
                print("❌ Failed to save configuration")
                return False
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False


    def create_project_template(self, project_name: str = None, project_dir: str = ".", 
                               interactive: bool = False, use_existing_config: str = None, 
                               format_type: str = 'yaml', **kwargs):
        """Create a new project template with CLI configuration."""
        
        # Create project path - if project_name is provided, create a subfolder
        if project_name:
            project_path = Path(project_dir) / project_name
            project_path.mkdir(parents=True, exist_ok=True)
        else:
            # If no project name provided, use current directory
            project_path = Path(project_dir).resolve()
            project_name = project_path.name
        
        print(f"🌱 Creating ModelGardener project: {project_name}")
        print(f"📁 Project directory: {project_path}")

        # Create project structure
        (project_path / "data" / "train").mkdir(parents=True, exist_ok=True)
        (project_path / "data" / "val").mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        (project_path / "custom_modules").mkdir(exist_ok=True)

        # Create configuration based on mode
        config_file = project_path / f"config.{format_type}"

        if use_existing_config:
            print(f"📋 Using existing configuration: {use_existing_config}")
            # Load existing configuration
            config = self.cli_interface.load_config(use_existing_config)
            if config:
                # Save to project directory
                if self.cli_interface.save_config(config, str(config_file), format_type):
                    print(f"✅ Configuration copied to {config_file}")
                else:
                    print("❌ Failed to copy configuration, creating default")
                    config = self.cli_interface.create_default_config()
                    self.cli_interface.save_config(config, str(config_file), format_type)
            else:
                print("❌ Failed to load existing configuration, creating default")
                config = self.cli_interface.create_default_config()
                self.cli_interface.save_config(config, str(config_file), format_type)
        elif interactive:
            print("\n🔧 Interactive project configuration")
            print("=" * 50)

            # Update paths to be relative to the project directory
            old_cwd = os.getcwd()
            os.chdir(project_path)

            try:
                config = self.cli_interface.interactive_configuration()

                # Validate and save configuration
                self.cli_interface._validate_configuration(config)
                print("\n📋 Configuration Summary:")
                print("=" * 50)
                print(self.cli_interface.generate_config_summary(config))
                
                if self.cli_interface.save_config(config, f"config.{format_type}", format_type):
                    print(f"✅ Configuration saved to {config_file}")
                else:
                    print("❌ Failed to save configuration, using default")
                    config = self.cli_interface.create_default_config()
                    self.cli_interface.save_config(config, f"config.{format_type}", format_type)
            finally:
                os.chdir(old_cwd)
        else:
            # Batch mode or default
            print("📋 Using default configuration")
            config = self.cli_interface.create_default_config()
            
            # Apply any provided batch modifications
            if kwargs:
                config = self.apply_batch_modifications(config, kwargs)
            
            # Display and save configuration
            print("\n📋 Configuration Summary:")
            print("=" * 50)
            print(self.cli_interface.generate_config_summary(config))
            
            if self.cli_interface.save_config(config, str(config_file), format_type):
                print(f"✅ Configuration saved to {config_file}")
        
        # Generate Python scripts
        print("\n🐍 Generating Python scripts...")
        try:
            script_gen = ScriptGenerator()
            success = script_gen.generate_scripts(config, str(project_path), str(config_file))
            if success:
                print("✅ Python scripts generated successfully!")
            else:
                print("⚠️  Some scripts may not have been generated correctly")
            
            print(f"📁 Location: {project_path.absolute()}")
            print("📄 Generated files:")
            
            generated_files = [
                ("train.py", "Training script"),
                ("evaluation.py", "Evaluation script"), 
                ("prediction.py", "Prediction script"),
                ("deploy.py", "Deployment script"),
                ("requirements.txt", "Python dependencies"),
                ("README.md", "Usage instructions"),
                ("custom_modules/", "Custom function templates")
            ]
            
            for filename, description in generated_files:
                file_path = project_path / filename
                if file_path.exists():
                    print(f"   • {filename} - {description}")
            
        except (ImportError, AttributeError, OSError) as e:
            print(f"❌ Failed to generate scripts: {e}")
        
        print(f"✅ Project template created successfully!")
        print(f"📖 See {project_path / 'README.md'} for instructions")

    def apply_batch_modifications(self, config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply batch modifications to a configuration."""
        # This is a simple implementation - you might want to make it more sophisticated
        cfg = config.get('configuration', {})
        
        if 'train_dir' in kwargs:
            cfg.setdefault('data', {})['train_dir'] = kwargs['train_dir']
        if 'val_dir' in kwargs:
            cfg.setdefault('data', {})['val_dir'] = kwargs['val_dir']
        if 'batch_size' in kwargs:
            cfg.setdefault('data', {}).setdefault('data_loader', {}).setdefault('parameters', {})['batch_size'] = kwargs['batch_size']
        if 'model_family' in kwargs:
            cfg.setdefault('model', {})['model_family'] = kwargs['model_family']
        if 'model_name' in kwargs:
            cfg.setdefault('model', {})['model_name'] = kwargs['model_name']
        if 'num_classes' in kwargs:
            cfg.setdefault('model', {}).setdefault('model_parameters', {})['classes'] = kwargs['num_classes']
        if 'epochs' in kwargs:
            cfg.setdefault('training', {})['epochs'] = kwargs['epochs']
        if 'learning_rate' in kwargs:
            cfg.setdefault('training', {}).setdefault('optimizer', {})['learning_rate'] = kwargs['learning_rate']
        if 'optimizer' in kwargs:
            cfg.setdefault('training', {}).setdefault('optimizer', {})['name'] = kwargs['optimizer']
        if 'loss_function' in kwargs:
            cfg.setdefault('training', {})['loss_function'] = kwargs['loss_function']
        if 'model_dir' in kwargs:
            cfg.setdefault('runtime', {})['model_dir'] = kwargs['model_dir']
        if 'num_gpus' in kwargs:
            cfg.setdefault('runtime', {})['num_gpus'] = kwargs['num_gpus']
            
        return config


def create_argument_parser():
    """Create command line argument parser with all original commands."""
    parser = argparse.ArgumentParser(
        description='ModelGardener CLI - ML model configuration and training setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create my_project                    # Create project with default config
  %(prog)s create my_project -i                # Interactive configuration
  %(prog)s create my_project --config my.yaml  # Use existing config file
  %(prog)s train --config config.yaml          # Train a model
  %(prog)s evaluate --config config.yaml       # Evaluate a model
  %(prog)s models                               # List available models
  %(prog)s config                               # Modify existing config
  %(prog)s check config.yaml                   # Validate config file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new ML project')
    create_parser.add_argument('project_name', nargs='?', default=None, 
                              help='Name of the project (optional - uses current directory name if not provided)')
    create_parser.add_argument('--dir', '-d', default='.', 
                              help='Directory to create project in (ignored if no project_name provided)')
    create_parser.add_argument('-i', '--interactive', action='store_true',
                              help='Use interactive configuration mode')
    create_parser.add_argument('-c', '--config', type=str,
                              help='Path to existing configuration file to copy')
    create_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Configuration file format (default: yaml)')
    
    # Add batch configuration arguments
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
    
    # Config command - modify existing configuration
    config_parser = subparsers.add_parser('config', help='Modify existing model configuration')
    config_parser.add_argument('config_file', nargs='?', 
                              help='Existing configuration file to modify (optional - will search current directory)')
    config_parser.add_argument('--interactive', '-i', action='store_true', 
                              help='Interactive configuration modification mode')
    config_parser.add_argument('--format', '-f', choices=['json', 'yaml'], 
                              help='Output format (inferred from file extension if not specified)')
    
    # Add batch modification arguments for config command
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
    train_parser.add_argument('--config', '-c', type=str, required=True, 
                             help='Configuration file')
    train_parser.add_argument('--resume', action='store_true', 
                             help='Resume training from checkpoint')
    train_parser.add_argument('--checkpoint', type=str, 
                             help='Checkpoint file to resume from')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', '-c', type=str, required=True, 
                            help='Configuration file')
    eval_parser.add_argument('--model-path', type=str, 
                            help='Path to trained model')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check configuration file')
    check_parser.add_argument('config_file', help='Configuration file to check')
    check_parser.add_argument('--verbose', '-v', action='store_true', 
                             help='Show detailed validation results')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ModelGardenerCLI()
    
    try:
        if args.command == 'create':
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
            
            cli.create_project_template(
                project_name=args.project_name, 
                project_dir=getattr(args, 'dir', '.'),
                interactive=args.interactive,
                use_existing_config=getattr(args, 'config', None),
                format_type=args.format,
                **kwargs
            )
        
        elif args.command == 'config':
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
            success = cli.run_training(args.config, args.resume, getattr(args, 'checkpoint', None))
            sys.exit(0 if success else 1)
        
        elif args.command == 'evaluate':
            success = cli.run_evaluation(args.config, getattr(args, 'model_path', None))
            sys.exit(0 if success else 1)
        
        elif args.command == 'models':
            cli.list_available_models()
        
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
