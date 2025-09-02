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
                "sharing_strategy": "file_paths_only",
                "creation_date": "",
                "model_gardener_version": "1.0"
            }
        }

    def _add_custom_functions_to_config(self, config: Dict[str, Any], project_dir: str) -> Dict[str, Any]:
        """
        Add custom function references to the configuration.
        
        Args:
            config: The base configuration
            project_dir: Path to the project directory
            
        Returns:
            Updated configuration with custom functions
        """
        custom_modules_dir = os.path.join(project_dir, 'custom_modules')
        
        # Define the custom functions to add based on generated files
        custom_functions = {
            'models': [{
                'name': 'create_simple_cnn',
                'file_path': './custom_modules/custom_models.py',
                'function_name': 'create_simple_cnn',
                'type': 'function'
            }],
            'data_loaders': [{
                'name': 'custom_image_data_loader',
                'file_path': './custom_modules/custom_data_loaders.py',
                'function_name': 'custom_image_data_loader',
                'type': 'function'
            }],
            'loss_functions': [{
                'name': 'custom_focal_loss',
                'file_path': './custom_modules/custom_loss_functions.py',
                'function_name': 'custom_focal_loss',
                'type': 'function'
            }],
            'optimizers': [{
                'name': 'custom_sgd_with_warmup',
                'file_path': './custom_modules/custom_optimizers.py',
                'function_name': 'custom_sgd_with_warmup',
                'type': 'function'
            }],
            'metrics': [{
                'name': 'custom_f1_score',
                'file_path': './custom_modules/custom_metrics.py',
                'function_name': 'custom_f1_score',
                'type': 'function'
            }],
            'callbacks': [{
                'name': 'LossThresholdStopping',
                'file_path': './custom_modules/custom_callbacks.py',
                'function_name': 'LossThresholdStopping',
                'type': 'class'
            }],
            'augmentations': [{
                'name': 'random_pixelate',
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': 'random_pixelate',
                'type': 'function'
            }],
            'preprocessing': [{
                'name': 'adaptive_histogram_equalization',
                'file_path': './custom_modules/custom_preprocessing.py',
                'function_name': 'adaptive_histogram_equalization',
                'type': 'function'
            }],
            'training_loops': [{
                'name': 'progressive_training_loop',
                'file_path': './custom_modules/custom_training_loops.py',
                'function_name': 'progressive_training_loop',
                'type': 'function'
            }]
        }
        
        # Update metadata with custom functions
        config['metadata']['custom_functions'] = custom_functions
        
        # Update specific configuration sections to use some of the custom functions
        # Example: Use custom model in model configuration
        config['configuration']['model']['model_family'] = 'custom_model'
        config['configuration']['model']['model_name'] = 'create_simple_cnn'
        config['configuration']['model']['model_parameters'] = {
            'input_shape': {'width': 224, 'height': 224, 'channels': 3},
            'num_classes': 3,  # Matching our example data
            'dropout_rate': 0.5,
            'custom_model_file_path': './custom_modules/custom_models.py',
            'custom_info': {
                'file_path': './custom_modules/custom_models.py',
                'type': 'function'
            }
        }
        
        # Update data paths to use local data
        config['configuration']['data']['train_dir'] = './data/train'
        config['configuration']['data']['val_dir'] = './data/val'
        
        return config

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
                    # Check if this is an improved template config (has custom enhancements)
                    if self._is_improved_template_config(config):
                        # Generate user-friendly YAML with comments
                        yaml_content = self._generate_improved_yaml(config)
                        f.write(yaml_content)
                    else:
                        # Use standard YAML format
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
        """Create a configuration template with custom functions and example data."""
        config = self.create_default_config()
        
        # Get project directory from template path
        project_dir = os.path.dirname(template_path)
        if not project_dir:
            project_dir = '.'
        
        # Add custom functions to config
        config = self._add_custom_functions_to_config(config, project_dir)
        
        # Copy example data to project directory
        self._copy_example_data(project_dir)
        
        # Generate custom modules templates first
        from script_generator import ScriptGenerator
        generator = ScriptGenerator()
        custom_modules_success = generator.generate_custom_modules_templates(project_dir)
        
        if not custom_modules_success:
            print("⚠️ Warning: Failed to generate some custom modules templates")
        
        # Now create the improved template with custom functions and parameters
        template_config = self._create_improved_template_config(config, project_dir)
        
        if self.save_config(template_config, template_path, format_type):
            print(f"✅ Template created at: {template_path}")
            print("📦 Custom modules created in: ./custom_modules/")
            print("📁 Sample data copied to: ./data/")
            print("🚀 Ready to train! The template includes working custom functions and sample data")
            print("💡 Run the generated train.py script to start training")

    def _copy_example_data(self, project_dir: str):
        """
        Copy example data to the project directory.
        
        Args:
            project_dir: Target project directory
        """
        import shutil
        
        # Define source and destination paths
        source_data_dir = os.path.join(os.path.dirname(__file__), 'example_data')
        dest_data_dir = os.path.join(project_dir, 'data')
        
        try:
            if os.path.exists(source_data_dir):
                # Remove existing data directory if it exists
                if os.path.exists(dest_data_dir):
                    shutil.rmtree(dest_data_dir)
                
                # Copy the example data
                shutil.copytree(source_data_dir, dest_data_dir)
                print(f"✅ Example data copied to: {dest_data_dir}")
                
                # Count files to show user what was copied
                total_files = 0
                for root, dirs, files in os.walk(dest_data_dir):
                    total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                print(f"📊 Copied {total_files} sample images across 3 classes")
                
            else:
                print(f"⚠️ Warning: Example data directory not found at {source_data_dir}")
                # Create minimal data structure
                os.makedirs(os.path.join(dest_data_dir, 'train', 'class_0'), exist_ok=True)
                os.makedirs(os.path.join(dest_data_dir, 'train', 'class_1'), exist_ok=True) 
                os.makedirs(os.path.join(dest_data_dir, 'train', 'class_2'), exist_ok=True)
                os.makedirs(os.path.join(dest_data_dir, 'val', 'class_0'), exist_ok=True)
                os.makedirs(os.path.join(dest_data_dir, 'val', 'class_1'), exist_ok=True)
                os.makedirs(os.path.join(dest_data_dir, 'val', 'class_2'), exist_ok=True)
                print("📁 Created empty data directory structure")
                
        except Exception as e:
            print(f"❌ Error copying example data: {str(e)}")
            # Create minimal structure as fallback
            try:
                os.makedirs(os.path.join(dest_data_dir, 'train'), exist_ok=True)
                os.makedirs(os.path.join(dest_data_dir, 'val'), exist_ok=True)
                print("📁 Created basic data directory structure")
            except Exception as e2:
                print(f"❌ Error creating data directories: {str(e2)}")

    def _create_improved_template_config(self, config: Dict[str, Any], project_dir: str = '.') -> Dict[str, Any]:
        """
        Create an improved template configuration with user-friendly comments and enhancements.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Enhanced configuration with user-friendly structure
        """
        # Start with the base configuration
        improved_config = config.copy()
        
        # Add custom augmentation option (disabled by default)
        if 'data' in improved_config['configuration'] and 'augmentation' in improved_config['configuration']['data']:
            improved_config['configuration']['data']['augmentation']['Custom Augmentation'] = {
                'enabled': False,
                'function_name': 'custom_augmentation_function', 
                'file_path': './custom_modules/custom_augmentations.py',
                'probability': 0.5
            }
        
        # Add custom preprocessing option with extracted parameters
        if 'data' in improved_config['configuration'] and 'preprocessing' in improved_config['configuration']['data']:
            preprocessing_params = self._extract_function_parameters('adaptive_histogram_equalization', 
                                                                   './custom_modules/custom_preprocessing.py', 
                                                                   project_dir)
            custom_preprocessing = {
                'enabled': False,
                'function_name': 'adaptive_histogram_equalization',
                'file_path': './custom_modules/custom_preprocessing.py'
            }
            custom_preprocessing.update(preprocessing_params)
            improved_config['configuration']['data']['preprocessing']['Custom Preprocessing'] = custom_preprocessing
        
        # Add custom callback option
        if 'model' in improved_config['configuration'] and 'callbacks' in improved_config['configuration']['model']:
            improved_config['configuration']['model']['callbacks']['Custom Callback'] = {
                'enabled': False,
                'callback_name': 'custom_callback_name',
                'file_path': './custom_modules/custom_callbacks.py'
            }
        
        # Remove custom optimizer from metadata (if present) since it's rarely used
        if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
            if 'optimizers' in improved_config['metadata']['custom_functions']:
                del improved_config['metadata']['custom_functions']['optimizers']
        
        # Remove references to non-existent function files from metadata
        if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
            # Keep only functions that have actual generated files
            existing_functions = {}
            
            # Check which custom modules were actually generated
            from script_generator import ScriptGenerator
            generator = ScriptGenerator()
            
            # These are the functions we know exist based on generated modules with their parameters
            known_functions = {
                'models': [{
                    'name': 'create_simple_cnn',
                    'file_path': './custom_modules/custom_models.py', 
                    'function_name': 'create_simple_cnn',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('create_simple_cnn', './custom_modules/custom_models.py', project_dir)
                }],
                'data_loaders': [{
                    'name': 'custom_image_data_loader',
                    'file_path': './custom_modules/custom_data_loaders.py',
                    'function_name': 'custom_image_data_loader', 
                    'type': 'function',
                    'parameters': self._extract_function_parameters('custom_image_data_loader', './custom_modules/custom_data_loaders.py', project_dir)
                }],
                'preprocessing': [{
                    'name': 'adaptive_histogram_equalization',
                    'file_path': './custom_modules/custom_preprocessing.py',
                    'function_name': 'adaptive_histogram_equalization',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('adaptive_histogram_equalization', './custom_modules/custom_preprocessing.py', project_dir)
                }],
                'training_loops': [{
                    'name': 'progressive_training_loop',
                    'file_path': './custom_modules/custom_training_loops.py',
                    'function_name': 'progressive_training_loop',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('progressive_training_loop', './custom_modules/custom_training_loops.py', project_dir)
                }]
            }
            
            improved_config['metadata']['custom_functions'] = known_functions
            
        return improved_config

    def _extract_function_parameters(self, function_name: str, file_path: str, project_dir: str = '.') -> Dict[str, Any]:
        """
        Extract function parameters from a custom function file.
        
        Args:
            function_name: Name of the function to extract parameters from
            file_path: Path to the file containing the function
            project_dir: Project directory for resolving relative paths
            
        Returns:
            Dictionary of function parameters with default values
        """
        import inspect
        import importlib.util
        import os
        
        try:
            # Convert relative path to absolute using project directory
            if not os.path.isabs(file_path):
                file_path = os.path.join(project_dir, file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️ Function parameter extraction: File {file_path} not found")
                return {}
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("custom_module", file_path)
            if spec is None or spec.loader is None:
                return {}
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the function
            if not hasattr(module, function_name):
                print(f"⚠️ Function {function_name} not found in {file_path}")
                return {}
            
            func = getattr(module, function_name)
            
            # Extract function signature
            sig = inspect.signature(func)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                # Skip the first parameter (usually 'data' or 'model')
                if param_name in ['data', 'model', 'self', 'cls']:
                    continue
                
                # Get default value
                if param.default != inspect.Parameter.empty:
                    default_value = param.default
                else:
                    # Provide sensible defaults based on parameter name
                    default_value = self._get_parameter_default_value(param_name, param.annotation)
                
                parameters[param_name] = default_value
            
            return parameters
            
        except Exception as e:
            print(f"⚠️ Error extracting parameters from {function_name}: {str(e)}")
            return {}

    def _get_parameter_default_value(self, param_name: str, param_annotation) -> Any:
        """Get a sensible default value for a parameter based on its name and type annotation."""
        # Common parameter name patterns and their defaults
        default_mappings = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'lr': 0.001,
            'epochs': 100,
            'dropout_rate': 0.5,
            'clip_limit': 2.0,
            'tile_grid_size': 8,
            'buffer_size': 10000,
            'image_size': [224, 224],
            'input_shape': [224, 224, 3],
            'num_classes': 1000,
            'shuffle': True,
            'augment': False,
            'enabled': False,
            'probability': 0.5,
            'patience': 10,
            'monitor': 'val_loss',
            'factor': 0.5,
            'min_lr': 1e-7,
            'initial_resolution': 32,
            'final_resolution': 224,
            'progression_schedule': 'linear'
        }
        
        # Check if parameter name matches known patterns
        for pattern, default in default_mappings.items():
            if pattern in param_name.lower():
                return default
        
        # Fall back to type-based defaults
        if param_annotation == int:
            return 1
        elif param_annotation == float:
            return 0.1
        elif param_annotation == bool:
            return False
        elif param_annotation == str:
            return ""
        elif param_annotation == list:
            return []
        else:
            return None

    def _is_improved_template_config(self, config: Dict[str, Any]) -> bool:
        """Check if this is an improved template configuration that needs custom YAML formatting."""
        # Check for custom augmentation/preprocessing/callback options that indicate improved template
        try:
            data_config = config.get('configuration', {}).get('data', {})
            model_config = config.get('configuration', {}).get('model', {})
            
            has_custom_aug = 'Custom Augmentation' in data_config.get('augmentation', {})
            has_custom_prep = 'Custom Preprocessing' in data_config.get('preprocessing', {})
            has_custom_callback = 'Custom Callback' in model_config.get('callbacks', {})
            
            return has_custom_aug or has_custom_prep or has_custom_callback
        except:
            return False

    def _generate_improved_yaml(self, config: Dict[str, Any]) -> str:
        """Generate user-friendly YAML with helpful comments."""
        yaml_lines = []
        
        # Header with instructions and options reference
        yaml_lines.extend([
            "# ModelGardener Configuration Template - Ready to run with custom functions and sample data",
            "",
            "# INSTRUCTIONS:",
            "# 1. Sample data has been copied to ./data/ directory with 3 classes", 
            "# 2. Custom functions are configured in metadata section below",
            "# 3. Modify parameters below to customize training behavior",
            "# 4. Run training with: python train.py",
            "",
            "# AVAILABLE OPTIONS REFERENCE:",
            "# - Optimizers: [Adam, SGD, RMSprop, Adagrad, AdamW, Adadelta, Adamax, Nadam, FTRL]",
            "# - Loss Functions: [Categorical Crossentropy, Sparse Categorical Crossentropy, Binary Crossentropy, Mean Squared Error, Mean Absolute Error, Focal Loss, Huber Loss]", 
            "# - Metrics: [Accuracy, Categorical Accuracy, Sparse Categorical Accuracy, Top-K Categorical Accuracy, Precision, Recall, F1 Score, AUC, Mean Squared Error, Mean Absolute Error]",
            "# - Training Loops: [Standard Training, Progressive Training, Curriculum Learning, Adversarial Training, Self-Supervised Training]",
            ""
        ])
        
        # Generate configuration section with comments
        configuration = config.get('configuration', {})
        yaml_lines.append("configuration:")
        
        # Task type
        task_type = configuration.get('task_type', 'image_classification')
        yaml_lines.append(f"  task_type: {task_type}")
        
        # Data section
        data_config = configuration.get('data', {})
        yaml_lines.append("  data:")
        yaml_lines.append(f"    train_dir: {data_config.get('train_dir', './data/train')}")
        yaml_lines.append(f"    val_dir: {data_config.get('val_dir', './data/val')}")
        
        # Add data loader section
        data_loader = data_config.get('data_loader', {})
        yaml_lines.extend([
            "    data_loader:",
            f"      selected_data_loader: {data_loader.get('selected_data_loader', 'Default')}",
            f"      use_for_train: {str(data_loader.get('use_for_train', True)).lower()}",
            f"      use_for_val: {str(data_loader.get('use_for_val', True)).lower()}",
            "      parameters:"
        ])
        
        params = data_loader.get('parameters', {})
        for key, value in params.items():
            yaml_lines.append(f"        {key}: {value}")
        
        # Preprocessing section with custom option
        preprocessing = data_config.get('preprocessing', {})
        yaml_lines.append("    preprocessing:")
        
        # Standard preprocessing options
        for key, value in preprocessing.items():
            if key != 'Custom Preprocessing':
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
        
        # Add custom preprocessing comment and option
        if 'Custom Preprocessing' in preprocessing:
            yaml_lines.extend([
                "      # Custom preprocessing (disabled by default)",
                "      Custom Preprocessing:"
            ])
            custom_prep = preprocessing['Custom Preprocessing']
            for key, value in custom_prep.items():
                yaml_lines.append(f"        {key}: {value}")
                
        # Augmentation section with custom option
        augmentation = data_config.get('augmentation', {})
        yaml_lines.append("    augmentation:")
        yaml_lines.append("      # Built-in augmentation options")
        
        for key, value in augmentation.items():
            if key != 'Custom Augmentation':
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
        
        # Add custom augmentation
        if 'Custom Augmentation' in augmentation:
            yaml_lines.extend([
                "      # Custom augmentation (disabled - file not included in this template)",
                "      # To add: Create ./custom_modules/custom_augmentations.py with desired functions",
                "      Custom Augmentation:"
            ])
            custom_aug = augmentation['Custom Augmentation']
            for key, value in custom_aug.items():
                yaml_lines.append(f"        {key}: {value}")
                
        # Model section
        model_config = configuration.get('model', {})
        yaml_lines.append("  model:")
        yaml_lines.append(f"    model_family: {model_config.get('model_family', 'custom_model')}")
        yaml_lines.append(f"    model_name: {model_config.get('model_name', 'create_simple_cnn')}")
        
        # Model parameters
        model_params = model_config.get('model_parameters', {})
        yaml_lines.append("    model_parameters:")
        for key, value in model_params.items():
            if isinstance(value, dict):
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
            else:
                yaml_lines.append(f"      {key}: {value}")
                
        # Optimizer section with comment
        optimizer = model_config.get('optimizer', {})
        yaml_lines.append("    optimizer:")
        yaml_lines.append("      # Available optimizers: [Adam, SGD, RMSprop, Adagrad, AdamW, Adadelta, Adamax, Nadam, FTRL]")
        for key, value in optimizer.items():
            yaml_lines.append(f"      {key}:")
            self._add_nested_yaml(yaml_lines, value, 8)
            
        # Loss functions with comment
        loss_functions = model_config.get('loss_functions', {})
        yaml_lines.append("    loss_functions:")
        yaml_lines.append("      # Available loss functions: [Categorical Crossentropy, Sparse Categorical Crossentropy, Binary Crossentropy, Mean Squared Error, Mean Absolute Error, Focal Loss, Huber Loss]")
        for key, value in loss_functions.items():
            yaml_lines.append(f"      {key}:")
            self._add_nested_yaml(yaml_lines, value, 8)
            
        # Metrics with comment
        metrics = model_config.get('metrics', {})
        yaml_lines.append("    metrics:")
        yaml_lines.append("      # Available metrics: [Accuracy, Categorical Accuracy, Sparse Categorical Accuracy, Top-K Categorical Accuracy, Precision, Recall, F1 Score, AUC, Mean Squared Error, Mean Absolute Error]")
        for key, value in metrics.items():
            yaml_lines.append(f"      {key}:")
            self._add_nested_yaml(yaml_lines, value, 8)
            
        # Callbacks with custom option
        callbacks = model_config.get('callbacks', {})
        yaml_lines.append("    callbacks:")
        
        for key, value in callbacks.items():
            if key != 'Custom Callback':
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
        
        # Add custom callback
        if 'Custom Callback' in callbacks:
            yaml_lines.extend([
                "      # Custom callback (disabled - file not included in this template)",
                "      # To add: Create ./custom_modules/custom_callbacks.py with desired callbacks", 
                "      Custom Callback:"
            ])
            custom_callback = callbacks['Custom Callback']
            for key, value in custom_callback.items():
                yaml_lines.append(f"        {key}: {value}")
                
        # Training section
        training = configuration.get('training', {})
        yaml_lines.append("  training:")
        for key, value in training.items():
            if key == 'training_loop':
                yaml_lines.append("    training_loop:")
                yaml_lines.append("      # Available training strategies: [Standard Training, Progressive Training, Curriculum Learning, Adversarial Training, Self-Supervised Training]")
                for sub_key, sub_value in value.items():
                    yaml_lines.append(f"      {sub_key}: {sub_value}")
            elif isinstance(value, dict):
                yaml_lines.append(f"    {key}:")
                self._add_nested_yaml(yaml_lines, value, 6)
            else:
                yaml_lines.append(f"    {key}: {value}")
                
        # Runtime section
        runtime = configuration.get('runtime', {})
        yaml_lines.append("  runtime:")
        for key, value in runtime.items():
            yaml_lines.append(f"    {key}: {value}")
            
        # Metadata section
        metadata = config.get('metadata', {})
        yaml_lines.append("metadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                yaml_lines.append(f"  {key}:")
                self._add_nested_yaml(yaml_lines, value, 4)
            else:
                yaml_lines.append(f"  {key}: {value}")
        
        return '\n'.join(yaml_lines)

    def _add_nested_yaml(self, yaml_lines: List[str], value: Any, indent_level: int):
        """Add nested YAML content with proper indentation."""
        indent = " " * indent_level
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    yaml_lines.append(f"{indent}{k}:")
                    self._add_nested_yaml(yaml_lines, v, indent_level + 2)
                elif isinstance(v, list):
                    yaml_lines.append(f"{indent}{k}:")
                    for item in v:
                        if isinstance(item, dict):
                            yaml_lines.append(f"{indent}- name: {item.get('name', '')}")
                            for sub_k, sub_v in item.items():
                                if sub_k != 'name':  # name already added
                                    if sub_k == 'parameters' and isinstance(sub_v, dict):
                                        # Add parameters as nested structure
                                        yaml_lines.append(f"{indent}  {sub_k}:")
                                        for param_k, param_v in sub_v.items():
                                            yaml_lines.append(f"{indent}    {param_k}: {param_v}")
                                    else:
                                        yaml_lines.append(f"{indent}  {sub_k}: {sub_v}")
                        else:
                            yaml_lines.append(f"{indent}- {item}")
                else:
                    yaml_lines.append(f"{indent}{k}: {v}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yaml_lines.append(f"{indent}- name: {item.get('name', '')}")
                    for k, v in item.items():
                        if k != 'name':  # name already added
                            if k == 'parameters' and isinstance(v, dict):
                                # Add parameters as nested structure
                                yaml_lines.append(f"{indent}  {k}:")
                                for param_k, param_v in v.items():
                                    yaml_lines.append(f"{indent}    {param_k}: {param_v}")
                            else:
                                yaml_lines.append(f"{indent}  {k}: {v}")
                else:
                    yaml_lines.append(f"{indent}- {item}")
        else:
            yaml_lines.append(f"{indent}{value}")


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
