#!/usr/bin/env python3
"""
CLI Configuration Tool for ModelGardener
Provides a command-line interface to configure model_config.json without the GUI.
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import inquirer
from dataclasses import dataclass
from config_manager import ConfigManager

# Import script generator
try:
    from script_generator import ScriptGenerator
except ImportError:
    print("Warning: ScriptGenerator not available")
    ScriptGenerator = None


@dataclass
class CLIConfig:
    """Configuration class for CLI settings."""
    config_file: str = "model_config.json"
    output_format: str = "json"
    interactive: bool = True
    template_mode: bool = False


class ModelConfigCLI:
    """CLI interface for ModelGardener configuration."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.current_config = {}
        self.available_models = {
            'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
            'efficientnet': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 
                           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'],
            'mobilenet': ['MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large'],
            'vgg': ['VGG16', 'VGG19'],
            'densenet': ['DenseNet121', 'DenseNet169', 'DenseNet201'],
            'inception': ['InceptionV3', 'InceptionResNetV2'],
            'xception': ['Xception'],
            'nasnet': ['NASNetMobile', 'NASNetLarge'],
            'custom': ['Custom Model']
        }
        self.available_optimizers = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
        self.available_losses = [
            'Categorical Crossentropy', 'Sparse Categorical Crossentropy', 'Binary Crossentropy',
            'Mean Squared Error', 'Mean Absolute Error', 'Huber Loss', 'Focal Loss'
        ]
        self.available_metrics = [
            'Accuracy', 'Categorical Accuracy', 'Sparse Categorical Accuracy', 'Top K Categorical Accuracy',
            'Precision', 'Recall', 'F1 Score', 'AUC', 'Mean Squared Error', 'Mean Absolute Error'
        ]

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration structure."""
        return {
            "configuration": {
                "task_type": "image_classification",
                "data": {
                    "train_dir": "",
                    "val_dir": "",
                    "data_loader": {
                        "selected_data_loader": "Default",
                        "use_for_train": True,
                        "use_for_val": True,
                        "parameters": {
                            "batch_size": 32,
                            "shuffle": True,
                            "buffer_size": 10000
                        }
                    },
                    "preprocessing": {
                        "Resizing": {
                            "enabled": True,
                            "target_size": {
                                "width": 224,
                                "height": 224,
                                "depth": 1
                            },
                            "interpolation": "bilinear",
                            "preserve_aspect_ratio": True,
                            "data_format": "2D"
                        },
                        "Normalization": {
                            "enabled": True,
                            "method": "zero-center",
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "mean": {"r": 0.485, "g": 0.456, "b": 0.406},
                            "std": {"r": 0.229, "g": 0.224, "b": 0.225},
                            "axis": -1,
                            "epsilon": 1e-07
                        }
                    },
                    "augmentation": {
                        "Horizontal Flip": {
                            "enabled": False,
                            "probability": 0.5
                        },
                        "Vertical Flip": {
                            "enabled": False,
                            "probability": 0.5
                        },
                        "Rotation": {
                            "enabled": False,
                            "angle_range": 15.0,
                            "probability": 0.5
                        },
                        "Gaussian Noise": {
                            "enabled": False,
                            "std_dev": 0.1,
                            "probability": 0.5
                        },
                        "Brightness": {
                            "enabled": False,
                            "delta_range": 0.2,
                            "probability": 0.5
                        },
                        "Contrast": {
                            "enabled": False,
                            "factor_range": [0.8, 1.2],
                            "probability": 0.5
                        }
                    }
                },
                "model": {
                    "model_family": "resnet",
                    "model_name": "ResNet-50",
                    "model_parameters": {
                        "input_shape": {"height": 224, "width": 224, "channels": 3},
                        "include_top": True,
                        "weights": "",
                        "pooling": "",
                        "classes": 1000,
                        "classifier_activation": "",
                        "kwargs": {}
                    },
                    "optimizer": {
                        "Optimizer Selection": {
                            "selected_optimizer": "Adam",
                            "learning_rate": 0.001,
                            "beta_1": 0.9,
                            "beta_2": 0.999,
                            "epsilon": 1e-07,
                            "amsgrad": False
                        }
                    },
                    "loss_functions": {
                        "Model Output Configuration": {
                            "num_outputs": 1,
                            "output_names": "main_output",
                            "loss_strategy": "single_loss_all_outputs"
                        },
                        "Loss Selection": {
                            "selected_loss": "Categorical Crossentropy",
                            "loss_weight": 1.0,
                            "from_logits": False,
                            "label_smoothing": 0.0,
                            "reduction": "sum_over_batch_size"
                        }
                    },
                    "metrics": {
                        "Model Output Configuration": {
                            "num_outputs": 1,
                            "output_names": "main_output",
                            "metrics_strategy": "shared_metrics_all_outputs"
                        },
                        "Metrics Selection": {
                            "selected_metrics": "Accuracy"
                        }
                    },
                    "callbacks": {
                        "Early Stopping": {
                            "enabled": False,
                            "monitor": "val_loss",
                            "patience": 10,
                            "min_delta": 0.001,
                            "mode": "min",
                            "restore_best_weights": True
                        },
                        "Learning Rate Scheduler": {
                            "enabled": False,
                            "scheduler_type": "ReduceLROnPlateau",
                            "monitor": "val_loss",
                            "factor": 0.5,
                            "patience": 5,
                            "min_lr": 1e-7
                        },
                        "Model Checkpoint": {
                            "enabled": True,
                            "monitor": "val_loss",
                            "save_best_only": True,
                            "save_weights_only": False,
                            "mode": "min",
                            "save_freq": "epoch"
                        }
                    }
                },
                "training": {
                    "epochs": 100,
                    "learning_rate_type": "exponential",
                    "initial_learning_rate": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "label_smoothing": 0.0,
                    "cross_validation": {
                        "enabled": False,
                        "k_folds": 5,
                        "validation_split": 0.2,
                        "stratified": True,
                        "shuffle": True,
                        "random_seed": 42,
                        "save_fold_models": False,
                        "fold_models_dir": "./logs/fold_models",
                        "aggregate_metrics": True,
                        "fold_selection_metric": "val_accuracy"
                    },
                    "training_loop": {
                        "selected_strategy": "Default Training Loop"
                    }
                },
                "runtime": {
                    "model_dir": "./logs",
                    "distribution_strategy": "mirrored",
                    "mixed_precision": None,
                    "num_gpus": 0
                }
            },
            "metadata": {
                "version": "1.2",
                "custom_functions": {},
                "sharing_strategy": "file_paths_with_content",
                "creation_date": "",
                "model_gardener_version": "1.0"
            }
        }

    def interactive_configuration(self) -> Dict[str, Any]:
        """Interactive configuration using inquirer."""
        print("\n🌱 ModelGardener CLI Configuration Tool")
        print("=" * 50)
        
        config = self.create_default_config()
        
        # Task Type Selection
        task_types = ['image_classification', 'object_detection', 'semantic_segmentation']
        task_type = inquirer.list_input(
            "Select task type",
            choices=task_types,
            default='image_classification'
        )
        config['configuration']['task_type'] = task_type
        
        # Data Configuration
        print("\n📁 Data Configuration")
        train_dir = inquirer.text("Enter training data directory", default="./example_data/train")
        val_dir = inquirer.text("Enter validation data directory", default="./example_data/val")
        
        config['configuration']['data']['train_dir'] = train_dir
        config['configuration']['data']['val_dir'] = val_dir
        
        # Batch size
        batch_size = inquirer.text("Enter batch size", default="32")
        try:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = int(batch_size)
        except ValueError:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = 32
        
        # Model Configuration
        print("\n🤖 Model Configuration")
        
        # Model family selection
        model_families = list(self.available_models.keys())
        model_family = inquirer.list_input(
            "Select model family",
            choices=model_families,
            default='resnet'
        )
        config['configuration']['model']['model_family'] = model_family
        
        # Model name selection
        model_names = self.available_models[model_family]
        model_name = inquirer.list_input(
            f"Select {model_family} model",
            choices=model_names,
            default=model_names[0] if model_names else 'ResNet-50'
        )
        config['configuration']['model']['model_name'] = model_name
        
        # Input shape configuration
        print("\n📐 Input Shape Configuration")
        height = inquirer.text("Enter image height", default="224")
        width = inquirer.text("Enter image width", default="224")
        channels = inquirer.text("Enter image channels", default="3")
        
        try:
            config['configuration']['model']['model_parameters']['input_shape'] = {
                'height': int(height),
                'width': int(width),
                'channels': int(channels)
            }
            # Update preprocessing size to match input shape
            config['configuration']['data']['preprocessing']['Resizing']['target_size']['height'] = int(height)
            config['configuration']['data']['preprocessing']['Resizing']['target_size']['width'] = int(width)
        except ValueError:
            print("⚠️  Invalid input shape values, using defaults")
        
        # Number of classes
        num_classes = inquirer.text("Enter number of classes", default="1000")
        try:
            config['configuration']['model']['model_parameters']['classes'] = int(num_classes)
        except ValueError:
            config['configuration']['model']['model_parameters']['classes'] = 1000
        
        # Optimizer Configuration
        print("\n⚡ Optimizer Configuration")
        optimizer = inquirer.list_input(
            "Select optimizer",
            choices=self.available_optimizers,
            default='Adam'
        )
        config['configuration']['model']['optimizer']['Optimizer Selection']['selected_optimizer'] = optimizer
        
        learning_rate = inquirer.text("Enter learning rate", default="0.001")
        try:
            config['configuration']['model']['optimizer']['Optimizer Selection']['learning_rate'] = float(learning_rate)
        except ValueError:
            config['configuration']['model']['optimizer']['Optimizer Selection']['learning_rate'] = 0.001
        
        # Loss Function Configuration
        print("\n📊 Loss Function Configuration")
        loss_function = inquirer.list_input(
            "Select loss function",
            choices=self.available_losses,
            default='Categorical Crossentropy'
        )
        config['configuration']['model']['loss_functions']['Loss Selection']['selected_loss'] = loss_function
        
        # Metrics Configuration
        print("\n📈 Metrics Configuration")
        metrics = inquirer.checkbox(
            "Select metrics (use space to select, enter to confirm)",
            choices=self.available_metrics,
            default=['Accuracy']
        )
        config['configuration']['model']['metrics']['Metrics Selection']['selected_metrics'] = ','.join(metrics)
        
        # Training Configuration
        print("\n🏃 Training Configuration")
        epochs = inquirer.text("Enter number of epochs", default="100")
        try:
            config['configuration']['training']['epochs'] = int(epochs)
        except ValueError:
            config['configuration']['training']['epochs'] = 100
        
        # Runtime Configuration
        print("\n⚙️  Runtime Configuration")
        model_dir = inquirer.text("Enter model output directory", default="./logs")
        config['configuration']['runtime']['model_dir'] = model_dir
        
        # GPU Configuration
        use_gpu = inquirer.confirm("Use GPU training?", default=True)
        if use_gpu:
            num_gpus = inquirer.text("Enter number of GPUs", default="1")
            try:
                config['configuration']['runtime']['num_gpus'] = int(num_gpus)
            except ValueError:
                config['configuration']['runtime']['num_gpus'] = 1
        else:
            config['configuration']['runtime']['num_gpus'] = 0
        
        # Set creation timestamp
        from datetime import datetime
        config['metadata']['creation_date'] = datetime.now().isoformat()
        
        return config

    def load_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(file_path):
            print(f"❌ Configuration file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            print(f"✅ Configuration loaded from: {file_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading configuration: {str(e)}")
            return {}

    def save_config(self, config: Dict[str, Any], file_path: str, format_type: str = 'json') -> bool:
        """Save configuration to file and generate Python scripts."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    # Use proper YAML format with better styling
                    yaml.dump(config, f, 
                             default_flow_style=False,  # Use block style
                             allow_unicode=True, 
                             indent=2,
                             sort_keys=False,  # Keep original order
                             width=1000)  # Avoid line wrapping
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to: {file_path}")
            
            # Generate Python scripts
            self._generate_python_scripts(config, file_path)
            
            return True
        except Exception as e:
            print(f"❌ Error saving configuration: {str(e)}")
            return False

    def _generate_python_scripts(self, config: Dict[str, Any], config_file_path: str):
        """
        Generate Python scripts (train.py, evaluation.py, prediction.py, deploy.py) 
        and custom modules templates in the same directory as the config file.
        
        Args:
            config: The configuration dictionary
            config_file_path: Path to the saved configuration file
        """
        if ScriptGenerator is None:
            print("⚠️  ScriptGenerator not available, skipping script generation")
            return
        
        try:
            # Get the directory where the config file is saved
            config_dir = os.path.dirname(config_file_path)
            if not config_dir:
                config_dir = '.'
            config_filename = os.path.basename(config_file_path)
            
            # Create script generator
            generator = ScriptGenerator()
            
            # Generate scripts
            print("\n🐍 Generating Python scripts...")
            success = generator.generate_scripts(config, config_dir, config_filename)
            
            # Generate custom modules templates
            print("📁 Generating custom modules templates...")
            custom_modules_success = generator.generate_custom_modules_templates(config_dir)
            
            if success:
                print("✅ Python scripts generated successfully!")
                print(f"📁 Location: {os.path.abspath(config_dir)}")
                print("📄 Generated files:")
                print("   • train.py - Training script")
                print("   • evaluation.py - Evaluation script") 
                print("   • prediction.py - Prediction script")
                print("   • deploy.py - Deployment script")
                print("   • requirements.txt - Python dependencies")
                print("   • README.md - Usage instructions")
                
                if custom_modules_success:
                    print("   • custom_modules/ - Custom function templates")
            else:
                print("❌ Failed to generate some Python scripts")
                
        except Exception as e:
            print(f"❌ Error generating Python scripts: {str(e)}")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        required_sections = ['configuration', 'metadata']
        required_config_sections = ['task_type', 'data', 'model', 'training', 'runtime']
        
        # Check top-level structure
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False
        
        # Check configuration sections
        config_section = config.get('configuration', {})
        for section in required_config_sections:
            if section not in config_section:
                print(f"❌ Missing required configuration section: {section}")
                return False
        
        # Validate data paths
        data_config = config_section.get('data', {})
        train_dir = data_config.get('train_dir', '')
        val_dir = data_config.get('val_dir', '')
        
        if train_dir and not os.path.exists(train_dir):
            print(f"⚠️  Warning: Training directory does not exist: {train_dir}")
        
        if val_dir and not os.path.exists(val_dir):
            print(f"⚠️  Warning: Validation directory does not exist: {val_dir}")
        
        print("✅ Configuration validation passed")
        return True

    def display_config_summary(self, config: Dict[str, Any]):
        """Display a summary of the configuration."""
        print("\n📋 Configuration Summary")
        print("=" * 50)
        
        config_section = config.get('configuration', {})
        
        print(f"Task Type: {config_section.get('task_type', 'N/A')}")
        
        # Data info
        data = config_section.get('data', {})
        print(f"Training Data: {data.get('train_dir', 'N/A')}")
        print(f"Validation Data: {data.get('val_dir', 'N/A')}")
        print(f"Batch Size: {data.get('data_loader', {}).get('parameters', {}).get('batch_size', 'N/A')}")
        
        # Model info
        model = config_section.get('model', {})
        print(f"Model: {model.get('model_name', 'N/A')} ({model.get('model_family', 'N/A')})")
        
        model_params = model.get('model_parameters', {})
        input_shape = model_params.get('input_shape', {})
        print(f"Input Shape: {input_shape.get('height', 'N/A')}x{input_shape.get('width', 'N/A')}x{input_shape.get('channels', 'N/A')}")
        print(f"Classes: {model_params.get('classes', 'N/A')}")
        
        # Optimizer info
        optimizer = model.get('optimizer', {}).get('Optimizer Selection', {})
        print(f"Optimizer: {optimizer.get('selected_optimizer', 'N/A')}")
        print(f"Learning Rate: {optimizer.get('learning_rate', 'N/A')}")
        
        # Loss function info
        loss = model.get('loss_functions', {}).get('Loss Selection', {})
        print(f"Loss Function: {loss.get('selected_loss', 'N/A')}")
        
        # Metrics info
        metrics = model.get('metrics', {}).get('Metrics Selection', {})
        print(f"Metrics: {metrics.get('selected_metrics', 'N/A')}")
        
        # Training info
        training = config_section.get('training', {})
        print(f"Epochs: {training.get('epochs', 'N/A')}")
        
        # Runtime info
        runtime = config_section.get('runtime', {})
        print(f"Model Directory: {runtime.get('model_dir', 'N/A')}")
        print(f"GPUs: {runtime.get('num_gpus', 'N/A')}")
        
        print("=" * 50)

    def batch_configuration(self, args: argparse.Namespace):
        """Configure using command line arguments."""
        config = self.create_default_config()
        
        # Update configuration based on command line arguments
        if hasattr(args, 'train_dir') and args.train_dir:
            config['configuration']['data']['train_dir'] = args.train_dir
        if hasattr(args, 'val_dir') and args.val_dir:
            config['configuration']['data']['val_dir'] = args.val_dir
        if hasattr(args, 'batch_size') and args.batch_size:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = args.batch_size
        if hasattr(args, 'model_family') and args.model_family:
            config['configuration']['model']['model_family'] = args.model_family
        if hasattr(args, 'model_name') and args.model_name:
            config['configuration']['model']['model_name'] = args.model_name
        if hasattr(args, 'epochs') and args.epochs:
            config['configuration']['training']['epochs'] = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config['configuration']['model']['optimizer']['Optimizer Selection']['learning_rate'] = args.learning_rate
        if hasattr(args, 'optimizer') and args.optimizer:
            config['configuration']['model']['optimizer']['Optimizer Selection']['selected_optimizer'] = args.optimizer
        if hasattr(args, 'loss_function') and args.loss_function:
            config['configuration']['model']['loss_functions']['Loss Selection']['selected_loss'] = args.loss_function
        if hasattr(args, 'num_classes') and args.num_classes:
            config['configuration']['model']['model_parameters']['classes'] = args.num_classes
        if hasattr(args, 'input_height') and args.input_height:
            config['configuration']['model']['model_parameters']['input_shape']['height'] = args.input_height
            config['configuration']['data']['preprocessing']['Resizing']['target_size']['height'] = args.input_height
        if hasattr(args, 'input_width') and args.input_width:
            config['configuration']['model']['model_parameters']['input_shape']['width'] = args.input_width
            config['configuration']['data']['preprocessing']['Resizing']['target_size']['width'] = args.input_width
        if hasattr(args, 'input_channels') and args.input_channels:
            config['configuration']['model']['model_parameters']['input_shape']['channels'] = args.input_channels
        if hasattr(args, 'model_dir') and args.model_dir:
            config['configuration']['runtime']['model_dir'] = args.model_dir
        if hasattr(args, 'num_gpus') and args.num_gpus is not None:
            config['configuration']['runtime']['num_gpus'] = args.num_gpus
        
        # Set creation timestamp
        from datetime import datetime
        config['metadata']['creation_date'] = datetime.now().isoformat()
        
        return config

    def create_template(self, template_path: str, format_type: str = 'yaml'):
        """Create a configuration template."""
        config = self.create_default_config()
        
        # Add comments for template
        template_config = {
            "_comment": "ModelGardener Configuration Template - Fill in the values below",
            "_instructions": {
                "train_dir": "Path to training data directory",
                "val_dir": "Path to validation data directory", 
                "model_family": f"Available: {list(self.available_models.keys())}",
                "model_name": "Specific model name within the family",
                "batch_size": "Batch size for training",
                "epochs": "Number of training epochs",
                "learning_rate": "Learning rate for optimizer",
                "num_classes": "Number of output classes"
            },
            **config
        }
        
        if self.save_config(template_config, template_path, format_type):
            print(f"✅ Template created at: {template_path}")
            print("💡 Edit the template file and run with --config to use it")


def create_argument_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="ModelGardener CLI Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive configuration
  python cli_config.py --interactive
  
  # Quick batch configuration
  python cli_config.py --train-dir ./data/train --val-dir ./data/val --model-family resnet --model-name ResNet-50 --epochs 50
  
  # Load and modify existing config
  python cli_config.py --config existing_config.json --interactive
  
  # Create a template
  python cli_config.py --template --output template.json
  
  # Export to YAML
  python cli_config.py --interactive --format yaml --output config.yaml
        """
    )
    
    # Input/Output options
    parser.add_argument('--config', '-c', type=str, help='Load existing configuration file')
    parser.add_argument('--output', '-o', type=str, default='model_config.json', help='Output configuration file')
    parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json', help='Output format')
    
    # Mode options
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive configuration mode')
    parser.add_argument('--template', '-t', action='store_true', help='Create configuration template')
    parser.add_argument('--validate', '-v', action='store_true', help='Validate configuration file')
    
    # Data configuration
    parser.add_argument('--train-dir', type=str, help='Training data directory')
    parser.add_argument('--val-dir', type=str, help='Validation data directory')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    
    # Model configuration
    parser.add_argument('--model-family', choices=list(ModelConfigCLI().available_models.keys()), 
                       help='Model family')
    parser.add_argument('--model-name', type=str, help='Model name')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--input-height', type=int, help='Input image height')
    parser.add_argument('--input-width', type=int, help='Input image width')
    parser.add_argument('--input-channels', type=int, help='Input image channels')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', choices=ModelConfigCLI().available_optimizers, help='Optimizer')
    parser.add_argument('--loss-function', choices=ModelConfigCLI().available_losses, help='Loss function')
    
    # Runtime configuration
    parser.add_argument('--model-dir', type=str, help='Model output directory')
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use')
    
    return parser


def main():
    """Main CLI function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    cli = ModelConfigCLI()
    
    # Template mode
    if args.template:
        cli.create_template(args.output)
        return
    
    # Validation mode
    if args.validate:
        if not args.config:
            print("❌ --validate requires --config to specify the file to validate")
            return
        config = cli.load_config(args.config)
        if config:
            cli.validate_config(config)
        return
    
    # Load existing configuration if specified
    if args.config:
        config = cli.load_config(args.config)
        if not config:
            print("❌ Failed to load configuration, creating new one")
            config = None
    else:
        config = None
    
    # Interactive mode
    if args.interactive:
        if config:
            print("🔄 Loaded existing configuration, you can modify it interactively")
            cli.display_config_summary(config)
            modify = inquirer.confirm("Do you want to modify this configuration?", default=True)
            if not modify:
                # Just save the existing config with new timestamp
                from datetime import datetime
                config['metadata']['creation_date'] = datetime.now().isoformat()
                cli.save_config(config, args.output, args.format)
                return
        
        config = cli.interactive_configuration()
    else:
        # Batch mode - use command line arguments
        config = cli.batch_configuration(args)
    
    # Validate configuration
    if not cli.validate_config(config):
        print("❌ Configuration validation failed")
        return
    
    # Display summary
    cli.display_config_summary(config)
    
    # Save configuration
    if cli.save_config(config, args.output, args.format):
        print(f"\n🎉 Configuration successfully created!")
        print(f"📄 File: {args.output}")
        print(f"📝 Format: {args.format.upper()}")
        print(f"\n💡 You can now use this configuration with ModelGardener")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚡ Configuration cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
