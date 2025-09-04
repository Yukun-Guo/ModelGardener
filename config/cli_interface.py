"""
Main CLI interface for ModelGardener configuration.
"""

import os
import argparse
from typing import Dict, Any, Optional, List
from .base_config import BaseConfig
from .model_config import ModelConfig
from .data_config import DataConfig
from .loss_metrics_config import LossMetricsConfig
from .preprocessing_config import PreprocessingConfig
from .augmentation_config import AugmentationConfig


class CLIInterface(BaseConfig):
    """Main CLI interface that orchestrates all configuration modules."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize all configuration modules
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.loss_metrics_config = LossMetricsConfig()
        self.preprocessing_config = PreprocessingConfig()
        self.augmentation_config = AugmentationConfig()
        
        # Available options
        self.available_models = self.model_config.available_models
        self.available_data_loaders = self.data_config.available_data_loaders
        self.available_losses = self.loss_metrics_config.available_losses
        self.available_metrics = self.loss_metrics_config.available_metrics
        self.available_optimizers = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']

    def interactive_configuration(self) -> Dict[str, Any]:
        """
        Main interactive configuration flow.
        
        Returns:
            Complete configuration dictionary
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return {}
            
        print("\n🌱 ModelGardener CLI Configuration Tool")
        print("=" * 50)
        
        # Initialize configuration with defaults
        config = self.create_default_config()
        
        # 1. Task Type Selection
        task_types = ['image_classification', 'object_detection', 'semantic_segmentation']
        task_type = inquirer.list_input(
            "Select task type",
            choices=task_types,
            default='image_classification'
        )
        config['configuration']['task_type'] = task_type
        
        # 2. Data Configuration
        print("\n📁 Data Configuration")
        train_dir = inquirer.text("Enter training data directory", default="./example_data/train")
        val_dir = inquirer.text("Enter validation data directory", default="./example_data/val")
        
        config['configuration']['data']['train_dir'] = train_dir
        config['configuration']['data']['val_dir'] = val_dir
        
        # 3. Data Loader Configuration
        print("\n📊 Data Loader Configuration")
        data_loader = inquirer.list_input(
            "Select data loader",
            choices=self.available_data_loaders,
            default='ImageDataGenerator'
        )
        
        batch_size = inquirer.text("Enter batch size", default="32")
        try:
            batch_size = int(batch_size)
        except ValueError:
            batch_size = 32
            self.print_warning("Invalid batch size, using default: 32")
        
        config['configuration']['data']['data_loader']['selected_data_loader'] = data_loader
        config['configuration']['data']['data_loader']['parameters']['batch_size'] = batch_size
        
        # Handle custom data loader
        if data_loader == 'Custom':
            custom_loader_path = inquirer.text(
                "Enter path to custom data loader file",
                default="./example_funcs/example_custom_data_loaders.py"
            )
            if custom_loader_path and os.path.exists(custom_loader_path):
                loader_name, loader_info = self.data_config.interactive_custom_data_loader_selection(custom_loader_path)
                if loader_name:
                    config['configuration']['data']['data_loader']['custom_loader'] = {
                        'name': loader_name,
                        'file_path': custom_loader_path,
                        'parameters': loader_info.get('user_parameters', {})
                    }
        
        # 4. Augmentation Configuration
        augmentation_config = self.augmentation_config.configure_augmentation(config)
        config['configuration']['data']['augmentation'] = augmentation_config
        
        # 5. Preprocessing Configuration  
        preprocessing_config = self.preprocessing_config.configure_preprocessing(config)
        config['configuration']['data']['preprocessing'] = preprocessing_config
        
        # 6. Model Configuration
        print("\n🤖 Model Configuration")
        model_family = inquirer.list_input(
            "Select model family",
            choices=list(self.available_models.keys()),
            default='resnet'
        )
        
        config['configuration']['model']['model_family'] = model_family
        
        if model_family == 'custom':
            # Custom model configuration
            print("\n📁 Custom Model Configuration")
            custom_model_path = inquirer.text(
                "Enter path to Python file containing custom model",
                default="./example_funcs/example_custom_models.py"
            )
            
            if custom_model_path and os.path.exists(custom_model_path):
                model_name, model_info = self.model_config.interactive_custom_model_selection(custom_model_path)
                if model_name:
                    config['configuration']['model']['model_name'] = model_name
                    config['configuration']['model']['model_parameters']['custom_info'] = model_info
                    config['configuration']['model']['model_parameters']['file_path'] = custom_model_path
                    
                    # Configure custom model parameters
                    if model_info.get('parameters'):
                        print(f"\n⚙️  Custom model parameters found: {len(model_info['parameters'])}")
                        for param_name, param_info in model_info['parameters'].items():
                            default_val = param_info.get('default', '')
                            user_value = inquirer.text(
                                f"Enter {param_name} (default: {default_val})",
                                default=str(default_val) if default_val is not None else ""
                            )
                            if user_value:
                                config['configuration']['model']['model_parameters'][param_name] = user_value
                        
                        # Additional parameters
                        kwargs = inquirer.text("Enter kwargs", default="")
                        if kwargs:
                            config['configuration']['model']['model_parameters']['kwargs'] = kwargs
            else:
                self.print_error("Custom model file not found")
                return {}
        else:
            # Standard model selection
            available_models = self.available_models[model_family]
            model_name = inquirer.list_input(
                f"Select {model_family} model",
                choices=available_models,
                default=available_models[0]
            )
            config['configuration']['model']['model_name'] = model_name
        
        # 7. Input Shape Configuration
        print("\n📐 Input Shape Configuration")
        height = inquirer.text("Enter image height", default="224")
        width = inquirer.text("Enter image width", default="224")
        channels = inquirer.text("Enter image channels", default="3")
        num_classes = inquirer.text("Enter number of classes", default="1000")
        
        try:
            config['configuration']['model']['model_parameters']['input_shape'] = {
                'height': int(height),
                'width': int(width),
                'channels': int(channels)
            }
            config['configuration']['model']['model_parameters']['classes'] = int(num_classes)
        except ValueError:
            self.print_warning("Invalid input shape values, using defaults")
        
        # 8. Optimizer Configuration
        print("\n⚡ Optimizer Configuration")
        optimizer = inquirer.list_input(
            "Select optimizer",
            choices=self.available_optimizers,
            default='Adam'
        )
        learning_rate = inquirer.text("Enter learning rate", default="0.001")
        
        try:
            config['configuration']['training']['optimizer']['name'] = optimizer
            config['configuration']['training']['optimizer']['learning_rate'] = float(learning_rate)
        except ValueError:
            self.print_warning("Invalid learning rate, using default: 0.001")
        
        # 9. Analyze model outputs for loss/metrics configuration
        print("\n📊 Loss Function Configuration")
        num_outputs, output_names = self.model_config.analyze_model_outputs(config)
        
        if num_outputs > 1:
            print(f"Detected {num_outputs} model outputs: {', '.join(output_names)}")
        
        # Configure loss functions
        loss_config = self._configure_loss_functions(config, num_outputs, output_names)
        config['configuration']['training']['loss_function'] = loss_config
        
        # 10. Configure metrics
        print("\n📈 Metrics Configuration")
        metrics_config = self._configure_metrics(config, loss_config, num_outputs, output_names)
        config['configuration']['training']['metrics'] = metrics_config
        
        # 11. Training Configuration
        print("\n🏃 Training Configuration")
        epochs = inquirer.text("Enter number of epochs", default="100")
        try:
            config['configuration']['training']['epochs'] = int(epochs)
        except ValueError:
            config['configuration']['training']['epochs'] = 100
            self.print_warning("Invalid epochs, using default: 100")
        
        # 12. Runtime Configuration
        print("\n⚙️  Runtime Configuration")
        model_dir = inquirer.text("Enter model output directory", default="./logs")
        use_gpu = inquirer.confirm("Use GPU training?", default=True)
        
        config['configuration']['runtime']['model_dir'] = model_dir
        config['configuration']['runtime']['use_gpu'] = use_gpu
        
        if use_gpu:
            num_gpus = inquirer.text("Enter number of GPUs", default="1")
            try:
                config['configuration']['runtime']['num_gpus'] = int(num_gpus)
            except ValueError:
                config['configuration']['runtime']['num_gpus'] = 1
                self.print_warning("Invalid GPU count, using default: 1")
        
        # 13. Validation
        self._validate_configuration(config)
        
        return config

    def _configure_loss_functions(self, config: Dict[str, Any], num_outputs: int, output_names: List[str]) -> Dict[str, Any]:
        """Configure loss functions based on model outputs."""
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required")
            return {}
            
        if num_outputs == 1:
            # Single output - single loss
            loss_choice = inquirer.list_input(
                "Select loss function",
                choices=self.available_losses + ['Load Custom Loss Functions'],
                default='Categorical Crossentropy'
            )
            
            if loss_choice == 'Load Custom Loss Functions':
                custom_loss_path = inquirer.text(
                    "Enter path to custom loss functions file",
                    default="./example_funcs/example_custom_loss_functions.py"
                )
                if custom_loss_path and os.path.exists(custom_loss_path):
                    loss_name, loss_info = self.loss_metrics_config.interactive_custom_loss_selection(custom_loss_path)
                    if loss_name:
                        return {
                            'strategy': 'single_loss',
                            'configuration': {
                                'name': loss_name,
                                'type': 'custom',
                                'file_path': custom_loss_path,
                                'parameters': loss_info.get('user_parameters', {})
                            }
                        }
                return {'strategy': 'single_loss', 'configuration': {'name': 'categorical_crossentropy'}}
            else:
                loss_name = loss_choice.lower().replace(' ', '_')
                return {'strategy': 'single_loss', 'configuration': {'name': loss_name}}
        else:
            # Multiple outputs
            print(f"Detected {num_outputs} model outputs: {', '.join(output_names)}")
            loss_strategy = inquirer.list_input(
                "Select loss strategy for multiple outputs",
                choices=[
                    'single_loss_all_outputs - Use the same loss function for all outputs',
                    'different_loss_each_output - Use different loss functions for each output'
                ],
                default='single_loss_all_outputs - Use the same loss function for all outputs'
            )
            
            if 'single_loss_all_outputs' in loss_strategy:
                loss_choice = inquirer.list_input(
                    "Select loss function",
                    choices=self.available_losses,
                    default='Categorical Crossentropy'
                )
                loss_name = loss_choice.lower().replace(' ', '_')
                return {
                    'strategy': 'single_loss_all_outputs',
                    'configuration': {'name': loss_name}
                }
            else:
                # Different loss for each output
                loss_configs = {}
                for output_name in output_names:
                    loss_choice = inquirer.list_input(
                        f"Select loss function for {output_name}",
                        choices=self.available_losses,
                        default='Categorical Crossentropy'
                    )
                    loss_name = loss_choice.lower().replace(' ', '_')
                    loss_configs[output_name] = {'name': loss_name}
                
                return {
                    'strategy': 'different_loss_each_output',
                    'configuration': loss_configs
                }

    def _configure_metrics(self, config: Dict[str, Any], loss_config: Dict[str, Any], 
                          num_outputs: int, output_names: List[str]) -> Dict[str, Any]:
        """Configure metrics based on model outputs."""
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required")
            return {}
            
        print(f"Using same outputs as loss functions: {num_outputs} outputs: {', '.join(output_names)}")
        
        if num_outputs == 1:
            # Single output metrics
            selected_metrics = inquirer.checkbox(
                "Select metrics (use space to select, enter to confirm)",
                choices=self.available_metrics + ['Load Custom Metrics Functions'],
                default=['Accuracy']
            )
            
            if 'Load Custom Metrics Functions' in selected_metrics:
                selected_metrics.remove('Load Custom Metrics Functions')
                custom_metrics_path = inquirer.text(
                    "Enter path to custom metrics functions file",
                    default="./example_funcs/example_custom_metrics_functions.py"
                )
                if custom_metrics_path and os.path.exists(custom_metrics_path):
                    success, metrics_info = self.loss_metrics_config.analyze_custom_metrics_file(custom_metrics_path)
                    if success:
                        custom_metrics = self.loss_metrics_config.interactive_custom_metrics_selection(custom_metrics_path, metrics_info)
                        selected_metrics.extend(custom_metrics)
            
            return {
                'strategy': 'single_output',
                'configuration': [metric.lower().replace(' ', '_') for metric in selected_metrics]
            }
        else:
            # Multiple outputs
            metrics_strategy = inquirer.list_input(
                "Select metrics strategy for multiple outputs",
                choices=[
                    'shared_metrics_all_outputs - Use the same metrics for all outputs',
                    'different_metrics_per_output - Use different metrics for each output'
                ],
                default='shared_metrics_all_outputs - Use the same metrics for all outputs'
            )
            
            if 'shared_metrics_all_outputs' in metrics_strategy:
                selected_metrics = inquirer.checkbox(
                    "Select metrics (use space to select, enter to confirm)",
                    choices=self.available_metrics,
                    default=['Accuracy', 'Precision']
                )
                
                return {
                    'strategy': 'shared_metrics_all_outputs',
                    'configuration': [metric.lower().replace(' ', '_') for metric in selected_metrics]
                }
            else:
                # Different metrics for each output
                metrics_configs = {}
                for output_name in output_names:
                    selected_metrics = inquirer.checkbox(
                        f"Select metrics for {output_name} (use space to select, enter to confirm)",
                        choices=self.available_metrics,
                        default=['Accuracy']
                    )
                    metrics_configs[output_name] = [metric.lower().replace(' ', '_') for metric in selected_metrics]
                
                return {
                    'strategy': 'different_metrics_per_output',
                    'configuration': metrics_configs
                }

    def _validate_configuration(self, config: Dict[str, Any]):
        """Validate the configuration and show warnings."""
        train_dir = config['configuration']['data']['train_dir']
        val_dir = config['configuration']['data']['val_dir']
        
        if not os.path.exists(train_dir):
            self.print_warning(f"Training directory does not exist: {train_dir}")
        if not os.path.exists(val_dir):
            self.print_warning(f"Validation directory does not exist: {val_dir}")
        
        self.print_success("Configuration validation passed")

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration structure."""
        return super().create_default_structure()

    def generate_config_summary(self, config: Dict[str, Any]) -> str:
        """Generate a human-readable configuration summary."""
        try:
            if 'configuration' not in config:
                return "Configuration Summary\n" + "=" * 20 + "\nNo configuration data found"
                
            cfg = config['configuration']
            model_cfg = cfg.get('model', {})
            data_cfg = cfg.get('data', {})
            training_cfg = cfg.get('training', {})
            
            summary = []
            summary.append("📋 Configuration Summary")
            summary.append("=" * 50)
            summary.append(f"Task Type: {cfg.get('task_type', 'N/A')}")
            summary.append(f"Training Data: {data_cfg.get('train_dir', 'N/A')}")
            summary.append(f"Validation Data: {data_cfg.get('val_dir', 'N/A')}")
            summary.append(f"Batch Size: {data_cfg.get('data_loader', {}).get('parameters', {}).get('batch_size', 'N/A')}")
            summary.append(f"Model: {model_cfg.get('model_name', 'N/A')} ({model_cfg.get('model_family', 'N/A')})")
            
            input_shape = model_cfg.get('model_parameters', {}).get('input_shape', {})
            if input_shape:
                summary.append(f"Input Shape: {input_shape.get('height', 'N/A')}x{input_shape.get('width', 'N/A')}x{input_shape.get('channels', 'N/A')}")
            
            summary.append(f"Classes: {model_cfg.get('model_parameters', {}).get('classes', 'N/A')}")
            
            optimizer_cfg = training_cfg.get('optimizer', {})
            summary.append(f"Optimizer: {optimizer_cfg.get('name', 'N/A')}")
            summary.append(f"Learning Rate: {optimizer_cfg.get('learning_rate', 'N/A')}")
            
            loss_cfg = training_cfg.get('loss_function', {})
            if loss_cfg.get('strategy') == 'single_loss':
                summary.append(f"Loss Function: {loss_cfg.get('configuration', {}).get('name', 'N/A')}")
            else:
                summary.append(f"Loss Function: {loss_cfg.get('name', 'N/A')}")
            
            metrics_cfg = training_cfg.get('metrics', [])
            if isinstance(metrics_cfg, list):
                summary.append(f"Metrics: {','.join(metrics_cfg) if metrics_cfg else 'N/A'}")
            else:
                summary.append(f"Metrics: {metrics_cfg}")
            
            summary.append(f"Epochs: {training_cfg.get('epochs', 'N/A')}")
            summary.append(f"Model Directory: {cfg.get('runtime', {}).get('model_dir', 'N/A')}")
            summary.append(f"GPUs: {cfg.get('runtime', {}).get('num_gpus', 'N/A')}")
            summary.append("=" * 50)
            
            return '\n'.join(summary)
            
        except Exception as e:
            return f"Error generating summary: {e}"
