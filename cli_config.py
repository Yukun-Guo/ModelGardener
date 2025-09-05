#!/usr/bin/env python3
"""
CLI Configuration Tool for ModelGardener
Provides a command-line interface to configure model_config.json without the GUI.
"""

# Suppress TensorFlow warnings as early as possible
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import argparse
import ast
import json
import yaml
import sys
import copy
import inspect
import importlib.util
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import inquirer
from dataclasses import dataclass
from config_manager import ConfigManager
import helper_funcs as hf

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
            'unet': ['UNet','ResUNet'],
            'custom': ['CustomModel']
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
        self.available_data_loaders = [
            'ImageDataGenerator', 'DirectoryDataLoader', 'TFRecordDataLoader', 'CSVDataLoader',
            'NPZDataLoader', 'Custom'
        ]


    def copy_custom_function_to_modules(self, source_path: str, function_type: str, destination_dir: str = None) -> str:
        """
        Copy a custom function file to the custom_modules directory and return the relative path.
        
        Args:
            source_path: Path to the source custom function file
            function_type: Type of function (e.g., 'preprocessing', 'augmentation', etc.)
            destination_dir: Target directory path (defaults to current working directory)
            
        Returns:
            Relative path to the copied file in ./custom_modules/ format
        """
        if not os.path.exists(source_path):
            print(f"⚠️  Source file does not exist: {source_path}")
            return source_path
            
        # Use current working directory if destination_dir is not provided
        if destination_dir is None:
            destination_dir = os.getcwd()
            
        # Create custom_modules directory if it doesn't exist
        custom_modules_dir = os.path.join(destination_dir, "custom_modules")
        os.makedirs(custom_modules_dir, exist_ok=True)
        
        # Generate destination filename
        source_filename = os.path.basename(source_path)
        
        # If the source file is not already named properly, rename it to match the function type
        if not source_filename.startswith(f"custom_{function_type}"):
            # Extract the original name without extension
            name_without_ext = os.path.splitext(source_filename)[0]
            dest_filename = f"custom_{function_type}.py"
        else:
            dest_filename = source_filename
            
        dest_path = os.path.join(custom_modules_dir, dest_filename)
        
        try:
            # Copy the file
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied custom function: {source_path} -> {dest_path}")
            
            # Return relative path for config
            return f"./custom_modules/{dest_filename}"
            
        except Exception as e:
            print(f"⚠️  Failed to copy custom function file: {e}")
            return source_path  # Return original path if copy fails


    def analyze_custom_model_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Analyze a Python file to extract custom model functions."""
        try:
            if not os.path.exists(file_path):
                return False, {"error": f"File does not exist: {file_path}"}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_model", file_path)
            if spec is None or spec.loader is None:
                return False, {"error": f"Cannot load module from: {file_path}"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find model functions/classes
            found_models = {}
            for name in dir(module):
                if name.startswith('_'):
                    continue
                    
                obj = getattr(module, name)
                if hf.is_model_function(obj, name):
                    model_info = {
                        'name': name,
                        'type': 'function' if inspect.isfunction(obj) else 'class',
                        'file_path': file_path,
                        'parameters': hf.extract_model_parameters(obj)
                    }
                    
                    # Add signature info for functions
                    if inspect.isfunction(obj):
                        try:
                            sig = inspect.signature(obj)
                            model_info['signature'] = str(sig)
                        except Exception:
                            model_info['signature'] = 'N/A'
                    
                    # Add docstring if available
                    if obj.__doc__:
                        model_info['description'] = obj.__doc__.strip().split('\n')[0]
                    else:
                        model_info['description'] = f"Custom {model_info['type']}: {name}"
                    
                    found_models[name] = model_info
            
            if not found_models:
                return False, {"error": "No valid model functions found in the file"}
            
            return True, found_models
            
        except Exception as e:
            return False, {"error": f"Failed to analyze file: {str(e)}"}

    def interactive_custom_model_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Interactive selection of custom model from analyzed file."""
        success, analysis_result = self.analyze_custom_model_file(file_path)
        
        if not success:
            print(f"❌ Error analyzing file: {analysis_result.get('error', 'Unknown error')}")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} custom model(s) in {os.path.basename(file_path)}")
        
        # Create choices for inquirer - show only signatures, not full descriptions
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} (function)"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = f"{name}"
            choices.append((choice_text, name))
        
        # Let user select the model
        selected = inquirer.list_input(
            "Select custom model to use",
            choices=[choice[0] for choice in choices],
            default=choices[0][0] if choices else None
        )
        
        # Find the selected model name
        selected_name = None
        for choice_text, name in choices:
            if choice_text == selected:
                selected_name = name
                break
        
        if selected_name and selected_name in analysis_result:
            return selected_name, analysis_result[selected_name]
        
        return None, {}

    def analyze_custom_data_loader_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract data loader functions and classes.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (success, data_loader_info)
        """
        import importlib.util
        import inspect
        
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_data_loaders", file_path)
            if spec is None or spec.loader is None:
                return False, {}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            data_loader_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if hf.is_data_loader_function(obj, name):
                    info = hf.extract_data_loader_parameters(obj)
                    if info:
                        data_loader_info[name] = info
            
            return len(data_loader_info) > 0, data_loader_info
            
        except Exception as e:
            print(f"❌ Error analyzing data loader file: {str(e)}")
            return False, {}

    def interactive_custom_data_loader_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom data loader from analyzed file.
        
        Args:
            file_path: Path to the custom data loader file
            
        Returns:
            Tuple of (selected_loader_name, loader_info)
        """
        success, analysis_result = self.analyze_custom_data_loader_file(file_path)
        
        if not success or not analysis_result:
            print("❌ No valid data loader functions found in the file")
            return None, {}
        
        print(f"\n✅ Found {len(analysis_result)} data loader function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user - show only name, not full descriptions
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function' in info:
                choice_text = f"{name} (function)"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = f"{name}"
            
            choices.append(choice_text)
        
        # Let user select
        selected_choice = inquirer.list_input(
            "Select custom data loader to use",
            choices=choices
        )
        
        # Extract the name from the choice (before any space or parenthesis)
        selected_name = selected_choice.split(' ')[0] if ' ' in selected_choice else selected_choice
        
        if selected_name in analysis_result:
            info = analysis_result[selected_name]
            print(f"\n✅ Selected custom data loader: {selected_name}")
            print(f"   Type: {info['type']}")
            
            # Ask for parameters if any
            parameters = {}
            if 'parameters' in info and info['parameters']:
                param_count = len([p for p in info['parameters'].values() if not p['required']])
                if param_count > 0:
                    print(f"\n⚙️  Custom data loader parameters found: {param_count}")
                    
                    for param_name, param_info in info['parameters'].items():
                        if not param_info['required']:  # Only ask for optional parameters
                            default_val = param_info.get('default', '')
                            user_value = inquirer.text(
                                f"Enter {param_name} (default: {default_val})",
                                default=str(default_val) if default_val is not None else ""
                            )
                            
                            # Convert to appropriate type
                            if user_value:
                                try:
                                    if param_info['type'] == 'int':
                                        parameters[param_name] = int(user_value)
                                    elif param_info['type'] == 'float':
                                        parameters[param_name] = float(user_value)
                                    elif param_info['type'] == 'bool':
                                        parameters[param_name] = user_value.lower() in ['true', '1', 'yes', 'on']
                                    else:
                                        parameters[param_name] = user_value
                                except ValueError:
                                    parameters[param_name] = user_value
            
            result_info = info.copy()
            if parameters:
                result_info['user_parameters'] = parameters
            
            return selected_name, result_info
        
        return None, {}

    def analyze_custom_loss_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract loss functions.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (success, loss_info)
        """
        import importlib.util
        import inspect
        
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_losses", file_path)
            if spec is None or spec.loader is None:
                return False, {}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            loss_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if hf.is_loss_function(obj, name):
                    info = hf.extract_loss_parameters(obj)
                    if info:
                        loss_info[name] = info
            
            return len(loss_info) > 0, loss_info
            
        except Exception as e:
            print(f"❌ Error analyzing loss function file: {str(e)}")
            return False, {}

    def interactive_custom_loss_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom loss function from analyzed file.
        
        Args:
            file_path: Path to the custom loss function file
            
        Returns:
            Tuple of (selected_loss_name, loss_info)
        """
        success, analysis_result = self.analyze_custom_loss_file(file_path)
        
        if not success or not analysis_result:
            print("❌ No valid loss functions found in the file")
            return None, {}
        
        print(f"\n✅ Found {len(analysis_result)} loss function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user - show only signatures, not full descriptions
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} {info['signature']}"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = f"{name} ({info['type']})"
            
            choices.append(choice_text)
        
        # Let user select
        selected_choice = inquirer.list_input(
            "Select custom loss function to use",
            choices=choices
        )
        
        # Extract the name from the choice (before any space or parenthesis)
        selected_name = selected_choice.split(' ')[0] if ' ' in selected_choice else selected_choice
        
        if selected_name in analysis_result:
            info = analysis_result[selected_name]
            print(f"\n✅ Selected custom loss function: {selected_name}")
            print(f"   Type: {info['type']}")
            
            # Ask for parameters if any
            parameters = {}
            if 'parameters' in info and info['parameters']:
                param_count = len([p for p in info['parameters'].values() if not p['required']])
                if param_count > 0:
                    print(f"\n⚙️  Custom loss function parameters found: {param_count}")
                    
                    for param_name, param_info in info['parameters'].items():
                        if not param_info['required']:  # Only ask for optional parameters
                            default_val = param_info.get('default', '')
                            user_value = inquirer.text(
                                f"Enter {param_name} (default: {default_val})",
                                default=str(default_val) if default_val is not None else ""
                            )
                            
                            # Convert to appropriate type
                            if user_value:
                                try:
                                    if param_info['type'] == 'int':
                                        parameters[param_name] = int(user_value)
                                    elif param_info['type'] == 'float':
                                        parameters[param_name] = float(user_value)
                                    elif param_info['type'] == 'bool':
                                        parameters[param_name] = user_value.lower() in ['true', '1', 'yes', 'on']
                                    else:
                                        parameters[param_name] = user_value
                                except ValueError:
                                    parameters[param_name] = user_value
            
            result_info = info.copy()
            if parameters:
                result_info['user_parameters'] = parameters
            
            return selected_name, result_info
        
        return None, {}

    def analyze_custom_metrics_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom metrics functions.
        Similar to analyze_custom_loss_file but for metrics functions.
        
        Args:
            file_path: Path to the Python file containing custom metrics
            
        Returns:
            Tuple of (success, analysis_result)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use AST to parse the file
            tree = ast.parse(content)
            
            functions_info = {}
            
            # Look for function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Skip private functions
                    if func_name.startswith('_'):
                        continue
                    
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    
                    # Look for typical metrics function patterns
                    if hf.is_likely_metrics_function(func_name, args, content):
                        # Extract parameters
                        parameters = hf.extract_function_parameters(node, content)
                        
                        # Get signature string
                        signature = f"({', '.join(args)})"
                        
                        functions_info[func_name] = {
                            'type': 'function',
                            'function_name': func_name,
                            'signature': signature,
                            'parameters': parameters,
                            'file_path': file_path
                        }
                
                # Look for class definitions that might be custom metrics
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Skip private classes
                    if class_name.startswith('_'):
                        continue
                    
                    # Look for classes that might be metrics (typically have __call__ or compute methods)
                    has_call = any(isinstance(n, ast.FunctionDef) and n.name == '__call__' 
                                 for n in node.body)
                    has_compute = any(isinstance(n, ast.FunctionDef) and n.name == 'compute' 
                                    for n in node.body)
                    
                    if has_call or has_compute or 'metric' in class_name.lower():
                        functions_info[class_name] = {
                            'type': 'class',
                            'function_name': class_name,
                            'signature': '(class)',
                            'parameters': {},
                            'file_path': file_path
                        }
            
            return len(functions_info) > 0, functions_info
            
        except Exception as e:
            print(f"Error analyzing metrics file {file_path}: {e}")
            return False, {}

    def interactive_custom_metrics_selection(self, file_path: str, metrics_info: Dict[str, Any]) -> List[str]:
        """
        Interactive selection of custom metrics functions from analyzed file.
        
        Args:
            file_path: Path to the custom metrics function file
            metrics_info: Analysis result from analyze_custom_metrics_file
            
        Returns:
            List of selected metrics names
        """
        if not metrics_info:
            print("❌ No metrics functions found in the file")
            return []
        
        print(f"\n✅ Found {len(metrics_info)} metrics function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user - show only name, not full descriptions
        choices = []
        for name, info in metrics_info.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} (function)"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = name
            choices.append(choice_text)
        
        # Let user select multiple metrics
        if len(choices) == 1:
            # Only one function available
            selected_choices = inquirer.checkbox(
                "Select metrics functions to load",
                choices=choices,
                default=choices
            )
        else:
            selected_choices = inquirer.checkbox(
                "Select metrics functions to load (use space to select, enter to confirm)",
                choices=choices
            )
        
        # Extract actual function names from choices
        selected_metrics = []
        for choice in selected_choices:
            # Extract function name (everything before the space or parenthesis)
            actual_name = choice.split(' ')[0] if ' ' in choice else choice.split('(')[0]
            selected_metrics.append(f"{actual_name} (custom)")
        
        return selected_metrics

    def analyze_model_outputs(self, config: Dict[str, Any]) -> Tuple[int, List[str]]:
        """
        Analyze the model configuration to determine the number of outputs and their names.
        
        Args:
            config: The current configuration
            
        Returns:
            Tuple of (num_outputs, output_names)
        """
        model_config = config.get('configuration', {}).get('model', {})
        model_family = model_config.get('model_family', '')
        model_name = model_config.get('model_name', '')
        
        # Try to dynamically analyze custom models
        if model_family == 'custom':
            custom_model_info = model_config.get('model_parameters', {}).get('custom_info', {})
            if custom_model_info:
                file_path = custom_model_info.get('file_path', '')
                function_name = custom_model_info.get('function_name', '')
                
                if file_path and function_name:
                    try:
                        # Attempt to load and analyze the custom model
                        num_outputs, output_names = hf.analyze_custom_model_outputs(
                            file_path, function_name, model_config
                        )
                        if num_outputs > 0:
                            return num_outputs, output_names
                    except Exception:
                        # Silently fall back to default behavior
                        pass
        
        # For built-in models, most have single output by default
        # Check model name for hints about multiple outputs
        if 'multi' in model_name.lower() or 'multiple' in model_name.lower():
            # Ask user for number of outputs
            try:
                num_outputs = int(inquirer.text("Enter number of model outputs", default="2"))
                output_names = []
                for i in range(num_outputs):
                    name = inquirer.text(f"Enter name for output {i+1}", 
                                       default=f"output_{i+1}" if i > 0 else "main_output")
                    output_names.append(name)
                return num_outputs, output_names
            except ValueError:
                pass
        
        # Default: single output
        return 1, ['main_output']

    def configure_loss_functions(self, config: Dict[str, Any], num_outputs: int = 1) -> Dict[str, Any]:
        """Configure loss functions for single or multiple outputs with improved workflow."""
        print("\n📊 Loss Function Configuration")
        
        # Analyze model outputs automatically (silently)
        detected_outputs, detected_names = self.analyze_model_outputs(config)
        
        # Always use detected configuration - no confirmation needed
        if detected_outputs > 1:
            print(f"Detected {detected_outputs} model outputs: {', '.join(detected_names)}")
        
        # Determine loss strategy based on number of outputs
        if detected_outputs == 1:
            loss_strategy = 'single_loss_all_outputs'
        else:
            loss_strategy_choice = inquirer.list_input(
                "Select loss strategy for multiple outputs",
                choices=[
                    'single_loss_all_outputs - Use the same loss function for all outputs',
                    'different_loss_each_output - Use different loss functions for each output'
                ],
                default='single_loss_all_outputs - Use the same loss function for all outputs'
            )
            loss_strategy = loss_strategy_choice.split(' - ')[0]
        
        # Configure loss functions based on strategy
        if loss_strategy == 'single_loss_all_outputs':
            # Configure single loss function for all outputs
            loss_config = self._configure_single_loss([], {})
            return {
                'Model Output Configuration': {
                    'num_outputs': detected_outputs,
                    'output_names': ','.join(detected_names),
                    'loss_strategy': 'single_loss_all_outputs'
                },
                'Loss Selection': loss_config
            }
        else:
            # Configure different loss functions for each output
            loss_configs = self._configure_multiple_losses(detected_outputs, detected_names)
            return {
                'Model Output Configuration': {
                    'num_outputs': detected_outputs,
                    'output_names': ','.join(detected_names),
                    'loss_strategy': 'different_loss_each_output'
                },
                'Loss Selection': loss_configs
            }

    def _configure_single_loss(self, available_custom_losses: List[str] = None, loaded_custom_configs: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Configure a single loss function with preset or custom options."""
        # Add Custom option to available losses, plus any already loaded custom losses
        loss_choices = self.available_losses.copy()
        
        # Add previously loaded custom losses to the choices with (custom) indicator
        if available_custom_losses:
            custom_choices = [f"{loss} (custom)" for loss in available_custom_losses]
            loss_choices.extend(custom_choices)
        
        # Add option to load new custom losses
        loss_choices.append('Load Custom Loss Functions')
        
        loss_function = inquirer.list_input(
            "Select loss function",
            choices=loss_choices,
            default='Categorical Crossentropy'
        )
        
        if loss_function == 'Load Custom Loss Functions':
            print("\n🔧 Custom Loss Function Configuration")
            custom_loss_path = inquirer.text(
                "Enter path to Python file containing custom loss functions"
            )
            
            if not custom_loss_path or not os.path.exists(custom_loss_path):
                print("❌ Invalid file path. Using default loss function.")
                loss_name = 'Categorical Crossentropy'
                loss_params = {}
            else:
                # Analyze custom loss function file
                success, loss_info = self.analyze_custom_loss_file(custom_loss_path)
                
                if not success or not loss_info:
                    print("❌ No valid loss functions found in the file. Using default loss function.")
                    loss_name = 'Categorical Crossentropy'
                    loss_params = {}
                else:
                    print(f"\n✅ Found {len(loss_info)} loss function(s) in {custom_loss_path}")
                    
                    # Let user select from available loss functions
                    loss_name, loss_params = self.interactive_custom_loss_selection(custom_loss_path)
            
            return {
                'selected_loss': loss_name or 'Categorical Crossentropy',
                'custom_loss_path': custom_loss_path if loss_name else None,
                'parameters': loss_params.get('user_parameters', {}) if loss_params else {}
            }
        else:
            # Handle custom loss functions (remove "(custom)" indicator if present)
            actual_loss_name = loss_function.replace(' (custom)', '') if ' (custom)' in loss_function else loss_function
            
            # Check if this is a custom loss function
            is_custom = ' (custom)' in loss_function
            
            if is_custom and loaded_custom_configs and actual_loss_name in loaded_custom_configs:
                # Use the stored configuration for previously loaded custom loss
                stored_config = loaded_custom_configs[actual_loss_name]
                return {
                    'selected_loss': actual_loss_name,
                    'custom_loss_path': stored_config['custom_loss_path'],
                    'parameters': copy.deepcopy(stored_config['parameters'])
                }
            else:
                return {
                    'selected_loss': actual_loss_name,
                    'custom_loss_path': None,
                    'parameters': {}
                }

    def _configure_multiple_losses(self, num_outputs: int, output_names: List[str] = None) -> Dict[str, Any]:
        """Configure different loss functions for multiple outputs."""
        loss_configs = {}
        loaded_custom_losses = []  # Track custom loss names
        loaded_custom_configs = {}  # Track full configurations of loaded custom losses
        
        # Use provided names or generate default ones
        if output_names is None:
            output_names = [f"output_{i + 1}" for i in range(num_outputs)]
        
        for i in range(num_outputs):
            output_name = output_names[i] if i < len(output_names) else f"output_{i + 1}"
            print(f"\n🎯 Configuring loss function for '{output_name}':")
            
            # Pass previously loaded custom losses to avoid re-loading
            loss_config = self._configure_single_loss(loaded_custom_losses, loaded_custom_configs)
            loss_configs[output_name] = loss_config
            
            # If a custom loss was selected, add it to the available list for next outputs
            selected_loss = loss_config.get('selected_loss', '')
            if loss_config.get('custom_loss_path') and selected_loss not in loaded_custom_losses:
                loaded_custom_losses.append(selected_loss)
                # Store the full configuration for reuse (create deep copy to avoid YAML anchors)
                if loss_config.get('custom_loss_path') != 'previously_loaded':
                    loaded_custom_configs[selected_loss] = {
                        'custom_loss_path': loss_config.get('custom_loss_path'),
                        'parameters': copy.deepcopy(loss_config.get('parameters', {}))
                    }
        
        return loss_configs

    def configure_metrics(self, config: Dict[str, Any], loss_functions_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure metrics for single or multiple outputs with improved workflow similar to loss functions."""
        print("\n📈 Metrics Configuration")
        
        # Reuse output analysis from loss functions configuration
        model_output_config = loss_functions_config.get('Model Output Configuration', {})
        detected_outputs = model_output_config.get('num_outputs', 1)
        detected_names = model_output_config.get('output_names', 'main_output').split(',')
        detected_names = [name.strip() for name in detected_names]
        
        # Always use detected configuration from loss functions
        if detected_outputs > 1:
            print(f"Using same outputs as loss functions: {detected_outputs} outputs: {', '.join(detected_names)}")
        
        # Determine metrics strategy based on number of outputs
        if detected_outputs == 1:
            metrics_strategy = 'shared_metrics_all_outputs'
        else:
            metrics_strategy_choice = inquirer.list_input(
                "Select metrics strategy for multiple outputs",
                choices=[
                    'shared_metrics_all_outputs - Use the same metrics for all outputs',
                    'different_metrics_per_output - Use different metrics for each output'
                ],
                default='shared_metrics_all_outputs - Use the same metrics for all outputs'
            )
            metrics_strategy = metrics_strategy_choice.split(' - ')[0]
        
        # Configure metrics based on strategy
        if metrics_strategy == 'shared_metrics_all_outputs':
            # Configure shared metrics for all outputs
            metrics_config = self._configure_single_metrics([], {})
            return {
                'Model Output Configuration': {
                    'num_outputs': detected_outputs,
                    'output_names': ','.join(detected_names),
                    'metrics_strategy': 'shared_metrics_all_outputs'
                },
                'Metrics Selection': metrics_config
            }
        else:
            # Configure different metrics for each output
            metrics_configs = self._configure_multiple_metrics(detected_outputs, detected_names)
            return {
                'Model Output Configuration': {
                    'num_outputs': detected_outputs,
                    'output_names': ','.join(detected_names),
                    'metrics_strategy': 'different_metrics_per_output'
                },
                'Metrics Selection': metrics_configs
            }

    def _configure_single_metrics(self, available_custom_metrics: List[str] = None, loaded_custom_configs: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Configure metrics for single output or shared across outputs."""
        # Add Custom option to available metrics, plus any already loaded custom metrics
        metrics_choices = self.available_metrics.copy()
        
        # Add previously loaded custom metrics to the choices with (custom) indicator
        if available_custom_metrics:
            custom_choices = [f"{metric} (custom)" for metric in available_custom_metrics]
            metrics_choices.extend(custom_choices)
        
        # Add option to load new custom metrics
        metrics_choices.append('Load Custom Metrics Functions')
        
        # Use checkbox for multiple selection
        selected_metrics = inquirer.checkbox(
            "Select metrics (use space to select, enter to confirm)",
            choices=metrics_choices,
            default=['Accuracy']
        )
        
        # Track newly loaded custom metrics for return to caller
        newly_loaded_custom_metrics = []
        custom_metrics_path = None
        metrics_info = {}
        
        # Handle custom metrics loading
        if 'Load Custom Metrics Functions' in selected_metrics:
            selected_metrics.remove('Load Custom Metrics Functions')  # Remove the option from selection
            print("\n🔧 Custom Metrics Function Configuration")
            custom_metrics_path = inquirer.text(
                "Enter path to Python file containing custom metrics functions"
            )
            
            if not custom_metrics_path or not os.path.exists(custom_metrics_path):
                print("❌ Invalid file path. Using selected built-in metrics only.")
            else:
                # Analyze custom metrics function file
                success, metrics_info = self.analyze_custom_metrics_file(custom_metrics_path)
                
                if not success or not metrics_info:
                    print("❌ No valid metrics functions found in the file. Using selected built-in metrics only.")
                else:
                    print(f"\n✅ Found {len(metrics_info)} metrics function(s) in {custom_metrics_path}")
                    
                    # Let user select from available metrics functions
                    additional_custom_metrics = self.interactive_custom_metrics_selection(custom_metrics_path, metrics_info)
                    if additional_custom_metrics:
                        selected_metrics.extend(additional_custom_metrics)
                        
                        # Store the newly loaded custom metrics configurations
                        if loaded_custom_configs is None:
                            loaded_custom_configs = {}
                        
                        for custom_metric in additional_custom_metrics:
                            actual_name = custom_metric.replace(' (custom)', '')
                            newly_loaded_custom_metrics.append(actual_name)
                            if actual_name in metrics_info:
                                loaded_custom_configs[actual_name] = {
                                    'custom_metrics_path': custom_metrics_path,
                                    'parameters': {}  # Initialize with empty parameters
                                }
        
        # Build final metrics configuration
        final_metrics = []
        custom_metrics_configs = {}
        
        for metric in selected_metrics:
            if ' (custom)' in metric:
                # Handle custom metrics
                actual_metric_name = metric.replace(' (custom)', '')
                
                # First check if it's in loaded_custom_configs (from current or previous loads)
                if loaded_custom_configs and actual_metric_name in loaded_custom_configs:
                    stored_config = loaded_custom_configs[actual_metric_name]
                    custom_metrics_configs[actual_metric_name] = {
                        'custom_metrics_path': stored_config['custom_metrics_path'],
                        'parameters': copy.deepcopy(stored_config.get('parameters', {}))
                    }
                else:
                    # Fallback: try to find it in the current metrics_info if available
                    if metrics_info and actual_metric_name in metrics_info:
                        custom_metrics_configs[actual_metric_name] = {
                            'custom_metrics_path': custom_metrics_path,
                            'parameters': {}
                        }
                
                final_metrics.append(actual_metric_name)
            else:
                # Handle built-in metrics
                final_metrics.append(metric)
        
        result = {
            'selected_metrics': ','.join(final_metrics),
            'custom_metrics_configs': custom_metrics_configs
        }
        
        # Add information about newly loaded custom metrics for caller to track
        if newly_loaded_custom_metrics:
            result['_newly_loaded_custom_metrics'] = newly_loaded_custom_metrics
        
        return result

    def configure_callbacks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure callbacks with support for multiple preset and custom callbacks."""
        print("\n📞 Callbacks Configuration")
        print("=" * 40)
        print("📋 Callbacks help monitor and control training progress")
        print("💡 Multiple callbacks can be enabled simultaneously")
        
        callbacks_config = config['configuration']['model'].get('callbacks', {})
        
        # Configure preset callbacks
        print("\n🔧 Preset Callbacks Configuration")
        print("=" * 35)
        
        # Early Stopping
        print("\n⏹️  Early Stopping")
        print("💡 Stops training when performance stops improving")
        enable_early_stopping = inquirer.confirm("Enable Early Stopping?", default=True)
        callbacks_config['Early Stopping']['enabled'] = enable_early_stopping
        
        if enable_early_stopping:
            monitor = inquirer.text("Monitor metric", default="val_loss")
            patience = inquirer.text("Patience (epochs to wait)", default="10")
            min_delta = inquirer.text("Minimum delta for improvement", default="0.001")
            mode = inquirer.list_input("Mode", choices=['min', 'max', 'auto'], default='min')
            restore_weights = inquirer.confirm("Restore best weights?", default=True)
            
            callbacks_config['Early Stopping'].update({
                'monitor': monitor,
                'patience': int(patience) if patience.isdigit() else 10,
                'min_delta': float(min_delta) if min_delta.replace('.', '').isdigit() else 0.001,
                'mode': mode,
                'restore_best_weights': restore_weights
            })
        
        # Learning Rate Scheduler
        print("\n📉 Learning Rate Scheduler")
        print("💡 Adjusts learning rate during training")
        enable_lr_scheduler = inquirer.confirm("Enable Learning Rate Scheduler?", default=True)
        callbacks_config['Learning Rate Scheduler']['enabled'] = enable_lr_scheduler
        
        if enable_lr_scheduler:
            scheduler_types = ['ReduceLROnPlateau', 'ExponentialDecay', 'CosineDecay', 'StepDecay']
            scheduler_type = inquirer.list_input("Select scheduler type", choices=scheduler_types, default='ReduceLROnPlateau')
            monitor = inquirer.text("Monitor metric", default="val_loss")
            factor = inquirer.text("Learning rate reduction factor", default="0.5")
            patience = inquirer.text("Patience (epochs)", default="5")
            min_lr = inquirer.text("Minimum learning rate", default="1e-7")
            
            callbacks_config['Learning Rate Scheduler'].update({
                'scheduler_type': scheduler_type,
                'monitor': monitor,
                'factor': float(factor) if factor.replace('.', '').isdigit() else 0.5,
                'patience': int(patience) if patience.isdigit() else 5,
                'min_lr': float(min_lr) if 'e' in min_lr or min_lr.replace('.', '').isdigit() else 1e-7
            })
        
        # Model Checkpoint
        print("\n💾 Model Checkpoint")
        print("💡 Saves model during training")
        enable_checkpoint = inquirer.confirm("Enable Model Checkpoint?", default=True)
        callbacks_config['Model Checkpoint']['enabled'] = enable_checkpoint
        
        if enable_checkpoint:
            monitor = inquirer.text("Monitor metric", default="val_loss")
            save_best_only = inquirer.confirm("Save only best model?", default=True)
            save_weights_only = inquirer.confirm("Save weights only (not full model)?", default=False)
            mode = inquirer.list_input("Mode", choices=['min', 'max', 'auto'], default='min')
            save_freq = inquirer.list_input("Save frequency", choices=['epoch', 'batch'], default='epoch')
            filepath = inquirer.text("Model filename", default="best_model.keras")
            
            callbacks_config['Model Checkpoint'].update({
                'monitor': monitor,
                'save_best_only': save_best_only,
                'save_weights_only': save_weights_only,
                'mode': mode,
                'save_freq': save_freq,
                'filepath': filepath
            })
        
        # TensorBoard
        print("\n📊 TensorBoard")
        print("💡 Provides training visualization and monitoring")
        enable_tensorboard = inquirer.confirm("Enable TensorBoard?", default=True)
        callbacks_config['TensorBoard']['enabled'] = enable_tensorboard
        
        if enable_tensorboard:
            log_dir = inquirer.text("TensorBoard log directory", default="./logs/tensorboard")
            histogram_freq = inquirer.text("Histogram frequency (epochs)", default="1")
            write_graph = inquirer.confirm("Write computation graph?", default=True)
            write_images = inquirer.confirm("Write model images?", default=False)
            update_freq = inquirer.list_input("Update frequency", choices=['epoch', 'batch'], default='epoch')
            profile_batch = inquirer.text("Profile batch (0 to disable)", default="0")
            
            callbacks_config['TensorBoard'].update({
                'log_dir': log_dir,
                'histogram_freq': int(histogram_freq) if histogram_freq.isdigit() else 1,
                'write_graph': write_graph,
                'write_images': write_images,
                'update_freq': update_freq,
                'profile_batch': int(profile_batch) if profile_batch.isdigit() else 0
            })
        
        # CSV Logger
        print("\n📝 CSV Logger")
        print("💡 Logs training metrics to CSV file")
        enable_csv_logger = inquirer.confirm("Enable CSV Logger?", default=True)
        callbacks_config['CSV Logger']['enabled'] = enable_csv_logger
        
        if enable_csv_logger:
            filename = inquirer.text("CSV filename", default="./logs/training.csv")
            append = inquirer.confirm("Append to existing file?", default=False)
            
            callbacks_config['CSV Logger'].update({
                'filename': filename,
                'append': append
            })
        
        # Custom Callbacks
        print("\n🛠️  Custom Callbacks")
        print("💡 Load custom callback functions from Python files")
        enable_custom_callbacks = inquirer.confirm("Add custom callbacks?", default=False)
        
        if enable_custom_callbacks:
            custom_callbacks = []
            
            while True:
                callback_file = inquirer.text("Enter path to Python file containing custom callbacks")
                
                if not callback_file:
                    break
                
                if not os.path.exists(callback_file):
                    print(f"❌ File not found: {callback_file}")
                    continue
                
                # Analyze the custom callback file
                found_callbacks = self.analyze_custom_callback_file(callback_file)
                
                if not found_callbacks:
                    print("❌ No valid callback functions found in the file")
                    continue
                
                print(f"\n🔍 Found {len(found_callbacks)} callback(s):")
                for name, info in found_callbacks.items():
                    print(f"   • {name}: {info.get('description', 'Custom callback function')}")
                
                # Allow user to select multiple callbacks from this file
                callback_choices = [(f"{name} - {info.get('description', 'Custom callback')}", name) 
                                  for name, info in found_callbacks.items()]
                
                selected_callbacks = inquirer.checkbox(
                    "Select callbacks to use",
                    choices=callback_choices
                )
                
                for callback_name in selected_callbacks:
                    callback_info = found_callbacks[callback_name]
                    
                    # Configure parameters for each selected callback
                    print(f"\n⚙️  Configuring parameters for {callback_name}:")
                    callback_params = {}
                    
                    if 'parameters' in callback_info:
                        for param_name, param_info in callback_info['parameters'].items():
                            default_value = param_info.get('default', '')
                            param_value = inquirer.text(f"Enter {param_name}", default=str(default_value))
                            
                            # Try to convert to appropriate type
                            if param_info.get('type') == 'int':
                                try:
                                    callback_params[param_name] = int(param_value)
                                except ValueError:
                                    callback_params[param_name] = param_info.get('default', 0)
                            elif param_info.get('type') == 'float':
                                try:
                                    callback_params[param_name] = float(param_value)
                                except ValueError:
                                    callback_params[param_name] = param_info.get('default', 0.0)
                            elif param_info.get('type') == 'bool':
                                callback_params[param_name] = param_value.lower() in ['true', '1', 'yes', 'y']
                            else:
                                callback_params[param_name] = param_value
                    
                    custom_callback_config = {
                        'function_name': callback_name,
                        'file_path': callback_file,
                        'type': callback_info.get('type', 'function'),
                        'parameters': callback_params
                    }
                    
                    custom_callbacks.append(custom_callback_config)
                    print(f"✅ Added custom callback: {callback_name}")
                
                add_more = inquirer.confirm("Add callbacks from another file?", default=False)
                if not add_more:
                    break
            
            callbacks_config['Custom Callbacks'] = {
                'enabled': len(custom_callbacks) > 0,
                'callbacks': custom_callbacks
            }
        
        print("\n✅ Callbacks configuration complete!")
        
        return callbacks_config

    def analyze_custom_callback_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file to extract custom callback functions."""
        try:
            if not os.path.exists(file_path):
                return {}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_callbacks", file_path)
            if spec is None or spec.loader is None:
                return {}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find callback functions/classes
            found_callbacks = {}
            for name in dir(module):
                if name.startswith('_'):
                    continue
                    
                obj = getattr(module, name)
                if hf.is_callback_function(obj, name):
                    # Get function info
                    sig = inspect.signature(obj)
                    doc = inspect.getdoc(obj) or f"Custom callback function: {name}"
                    
                    # Extract parameters
                    parameters = {}
                    for param_name, param in sig.parameters.items():
                        param_info = {
                            'name': param_name,
                            'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'str',
                            'default': param.default if param.default != inspect.Parameter.empty else None
                        }
                        parameters[param_name] = param_info
                    
                    found_callbacks[name] = {
                        'type': 'function' if inspect.isfunction(obj) else 'class',
                        'description': doc.split('\n')[0] if doc else f"Custom callback: {name}",
                        'parameters': parameters
                    }
            
            return found_callbacks
            
        except Exception as e:
            print(f"Error analyzing callback file {file_path}: {str(e)}")
            return {}

    def _configure_multiple_metrics(self, num_outputs: int, output_names: List[str] = None) -> Dict[str, Any]:
        """Configure different metrics for multiple outputs."""
        metrics_configs = {}
        loaded_custom_metrics = []  # Track custom metrics names
        loaded_custom_configs = {}  # Track full configurations of loaded custom metrics
        
        # Use provided names or generate default ones
        if output_names is None:
            output_names = [f"output_{i + 1}" for i in range(num_outputs)]
        
        for i in range(num_outputs):
            output_name = output_names[i] if i < len(output_names) else f"output_{i + 1}"
            print(f"\n🎯 Configuring metrics for '{output_name}':")
            
            # Pass previously loaded custom metrics to avoid re-loading
            metrics_config = self._configure_single_metrics(loaded_custom_metrics, loaded_custom_configs)
            metrics_configs[output_name] = metrics_config
            
            # Handle newly loaded custom metrics from this configuration
            if '_newly_loaded_custom_metrics' in metrics_config:
                for new_metric_name in metrics_config['_newly_loaded_custom_metrics']:
                    if new_metric_name not in loaded_custom_metrics:
                        loaded_custom_metrics.append(new_metric_name)
                # Remove the temporary tracking info from the final config
                del metrics_config['_newly_loaded_custom_metrics']
            
            # If custom metrics were configured, add them to the available list for next outputs
            custom_configs = metrics_config.get('custom_metrics_configs', {})
            for metric_name, config_data in custom_configs.items():
                if metric_name not in loaded_custom_metrics:
                    loaded_custom_metrics.append(metric_name)
                # Always update/store the full configuration for reuse
                loaded_custom_configs[metric_name] = {
                    'custom_metrics_path': config_data.get('custom_metrics_path'),
                    'parameters': copy.deepcopy(config_data.get('parameters', {}))
                }
        
        return metrics_configs

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration structure."""
        return {
            "configuration": {
                "task_type": "image_classification",
                "data": {
                    "train_dir": "./data",
                    "val_dir": "./data",
                    "data_loader": {
                        "selected_data_loader": "Custom_load_cifar10_npz_data",
                        "use_for_train": True,
                        "use_for_val": True,
                        "parameters": {
                            "batch_size": 32,
                            "shuffle": True,
                            "buffer_size": 1000,
                            "npz_file_path": "./data/cifar10.npz"
                        }
                    },
                    "preprocessing": {
                        "Resizing": {
                            "enabled": False,
                            "target_size": {
                                "width": 32,
                                "height": 32,
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
                    "model_family": "custom_model",
                    "model_name": "create_simple_cnn",
                    "model_parameters": {
                        "input_shape": {"height": 32, "width": 32, "channels": 3},
                        "include_top": True,
                        "weights": "",
                        "pooling": "",
                        "classes": 10,
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
                            "enabled": True,
                            "monitor": "val_loss",
                            "patience": 10,
                            "min_delta": 0.001,
                            "mode": "min",
                            "restore_best_weights": True
                        },
                        "Learning Rate Scheduler": {
                            "enabled": True,
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
                            "save_freq": "epoch",
                            "filepath": "best_model.keras"
                        },
                        "TensorBoard": {
                            "enabled": True,
                            "log_dir": "./logs/tensorboard",
                            "histogram_freq": 1,
                            "write_graph": True,
                            "write_images": False,
                            "update_freq": "epoch",
                            "profile_batch": 0
                        },
                        "CSV Logger": {
                            "enabled": True,
                            "filename": "./logs/training.csv",
                            "append": False
                        },
                        "Custom Callbacks": {
                            "enabled": False,
                            "callbacks": []
                        }
                    }
                },
                "training": {
                    "epochs": 100,
                    "cross_validation": {
                        "enabled": False,
                        "k_folds": 5,
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
        
        # Dynamically discover custom functions from example_funcs directory
        augmentation_functions = hf.discover_custom_functions('./example_funcs/example_custom_augmentations.py')
        preprocessing_functions = hf.discover_custom_functions('./example_funcs/example_custom_preprocessing.py')
        
        # Define the custom functions to add based on discovered and generated files
        custom_functions = {
            'models': [{
                'name': 'create_simple_cnn',
                'file_path': './custom_modules/custom_models.py',
                'function_name': 'create_simple_cnn',
                'type': 'function'
            }],
            'data_loaders': [{
                'name': 'Custom_load_cifar10_npz_data',
                'file_path': './custom_modules/custom_data_loaders.py',
                'function_name': 'Custom_load_cifar10_npz_data',
                'type': 'function'
            }],
            'loss_functions': [{
                'name': 'dice_loss',
                'file_path': './custom_modules/custom_loss_functions.py',
                'function_name': 'dice_loss',
                'type': 'function'
            }],
            'optimizers': [{
                'name': 'adaptive_adam',
                'file_path': './custom_modules/custom_optimizers.py',
                'function_name': 'adaptive_adam',
                'type': 'function'
            }],
            'metrics': [{
                'name': 'balanced_accuracy',
                'file_path': './custom_modules/custom_metrics.py',
                'function_name': 'balanced_accuracy',
                'type': 'function'
            }],
            'callbacks': [{
                'name': 'MemoryUsageMonitor',
                'file_path': './custom_modules/custom_callbacks.py',
                'function_name': 'MemoryUsageMonitor',
                'type': 'class'
            }],
            'augmentations': [{
                'name': 'color_shift',
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': 'color_shift',
                'type': 'function'
            }],
            'preprocessing': [],
            'training_loops': [{
                'name': 'progressive_training_loop',
                'file_path': './custom_modules/custom_training_loops.py',
                'function_name': 'progressive_training_loop',
                'type': 'function'
            }]
        }
        
        # Add discovered augmentation functions
        for func_name, func_info in augmentation_functions.items():
            custom_functions['augmentations'].append({
                'name': func_name,
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': func_name,
                'type': 'function'
            })
        
        # Add discovered preprocessing functions  
        for func_name, func_info in preprocessing_functions.items():
            custom_functions['preprocessing'].append({
                'name': func_name,
                'file_path': './custom_modules/custom_preprocessing.py',
                'function_name': func_name,
                'type': 'function'
            })
        
        # Update metadata with custom functions
        config['metadata']['custom_functions'] = custom_functions
        
        # Update specific configuration sections to use some of the custom functions
        # Example: Use custom model in model configuration
        config['configuration']['model']['model_family'] = 'custom_model'
        config['configuration']['model']['model_name'] = 'create_simple_cnn'
        config['configuration']['model']['model_parameters'] = {
            'input_shape': {'width': 32, 'height': 32, 'channels': 3},
            'num_classes': 10,  # CIFAR-10 classes
            'dropout_rate': 0.5,
            'custom_model_file_path': None,
            'custom_info': {
                'file_path': None,
                'type': 'function'
            }
        }
        
        # Update data paths for CIFAR-10 dataset
        config['configuration']['data']['train_dir'] = './data'
        config['configuration']['data']['val_dir'] = './data'
        
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
        
        
        # Data Loader Selection
        print("\n📊 Data Loader Configuration")
        data_loader = inquirer.list_input(
            "Select data loader",
            choices=self.available_data_loaders,
            default='ImageDataGenerator'
        )
        config['configuration']['data']['data_loader']['selected_data_loader'] = data_loader
        
        # Handle custom data loader selection
        if data_loader == 'Custom':
            print("\n🔧 Custom Data Loader Configuration")
            custom_data_loader_path = inquirer.text(
                "Enter path to Python file containing custom data loader"
            )
            
            if not custom_data_loader_path or not os.path.exists(custom_data_loader_path):
                print("❌ Invalid file path. Using default data loader.")
                data_loader_name = 'ImageDataGenerator'
                data_loader_params = {}
            else:
                # Analyze custom data loader file
                success, loader_info = self.analyze_custom_data_loader_file(custom_data_loader_path)
                
                if not success or not loader_info:
                    print("❌ No valid data loader functions found in the file. Using default data loader.")
                    data_loader_name = 'ImageDataGenerator'
                    data_loader_params = {}
                else:
                    print(f"\n✅ Found {len(loader_info)} data loader function(s) in {custom_data_loader_path}")
                    
                    # Let user select from available data loaders
                    data_loader_name, data_loader_params = self.interactive_custom_data_loader_selection(custom_data_loader_path)
                    
                    # Add custom data loader path to config
                    config['configuration']['data']['data_loader']['custom_data_loader_path'] = custom_data_loader_path
            
            config['configuration']['data']['data_loader']['selected_data_loader'] = data_loader_name or 'ImageDataGenerator'
            
            # Update data loader parameters if available
            if data_loader_params and 'user_parameters' in data_loader_params:
                if 'parameters' not in config['configuration']['data']['data_loader']:
                    config['configuration']['data']['data_loader']['parameters'] = {}
                config['configuration']['data']['data_loader']['parameters'].update(data_loader_params['user_parameters'])
        
        # Batch size
        batch_size = inquirer.text("Enter batch size", default="32")
        try:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = int(batch_size)
        except ValueError:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = 32
            
            
        # Preprocessing Configuration - Ask immediately after data directories are configured
        print(f"\n✅ Data directories configured:")
        print(f"   📂 Training: {train_dir}")
        print(f"   📂 Validation: {val_dir}")
        
        print("\n🔧 Data Preprocessing Configuration")
        configure_preprocessing = inquirer.confirm(
            "Would you like to configure data preprocessing for your data? (Recommended)",
            default=True
        )
        
        if configure_preprocessing:
            preprocessing_config = self.configure_preprocessing(config)
            config['configuration']['data']['preprocessing'] = preprocessing_config
        else:
            # Use default minimal preprocessing
            config['configuration']['data']['preprocessing'] = {
                "Resizing": {
                    "enabled": True,
                    "target_size": {"width": 224, "height": 224, "depth": 1},
                    "interpolation": "bilinear",
                    "preserve_aspect_ratio": True,
                    "data_format": "2D"
                },
                "Normalization": {
                    "enabled": True,
                    "method": "zero-center",
                    "min_value": 0.0,
                    "max_value": 1.0
                }
            }
            print("✅ Using default preprocessing settings (resize to 224x224, normalize to [0,1])")
        
        # Augmentation Configuration
        print("\n🎲 Data Augmentation Configuration")
        configure_augmentation = inquirer.confirm(
            "Would you like to configure data augmentation?",
            default=False
        )
        
        if configure_augmentation:
            augmentation_config = self.configure_augmentation(config)
            config['configuration']['data']['augmentation'] = augmentation_config
        else:
            # Use minimal augmentation
            config['configuration']['data']['augmentation'] = {
                "Horizontal Flip": {
                    "enabled": False,
                    "probability": 0.5
                },
                "Rotation": {
                    "enabled": False,
                    "angle_range": 15.0,
                    "probability": 0.5
                }
            }
            print("✅ Using default augmentation settings (no augmentation)")
        
        
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
        
        # Handle custom model selection
        if model_family == 'custom':
            print("\n📁 Custom Model Configuration")
            custom_model_path = inquirer.text(
                "Enter path to Python file containing custom model"
            )
            
            # Validate file exists
            if not os.path.exists(custom_model_path):
                print(f"⚠️  File not found: {custom_model_path}")
                print("Using default custom model configuration...")
                model_name = 'CustomModel'
                model_parameters = {}
                custom_model_info = {
                    'file_path': custom_model_path,
                    'type': 'function'
                }
            else:
                # Analyze and let user select custom model
                selected_name, model_info = self.interactive_custom_model_selection(custom_model_path)
                
                if selected_name and model_info:
                    model_name = selected_name
                    model_parameters = model_info.get('parameters', {})
                    custom_model_info = {
                        'file_path': custom_model_path,
                        'type': model_info.get('type', 'function'),
                        'function_name': selected_name,
                        'description': model_info.get('description', '')
                    }
                    print(f"✅ Selected custom model: {model_name}")
                    print(f"   Type: {model_info.get('type', 'function')}")
                else:
                    print("⚠️  No valid model selected, using default...")
                    model_name = 'CustomModel'
                    model_parameters = {}
                    custom_model_info = {
                        'file_path': custom_model_path,
                        'type': 'function'
                    }
            
            # Store custom model information in config
            config['configuration']['model']['model_name'] = model_name
            config['configuration']['model']['model_parameters']['custom_model_file_path'] = custom_model_path
            config['configuration']['model']['model_parameters']['custom_info'] = custom_model_info
            
            # Add custom model parameters if any were found
            if model_parameters:
                print(f"\n⚙️  Custom model parameters found: {len(model_parameters)}")
                for param_name, param_info in model_parameters.items():
                    default_val = param_info.get('default', '')
                    param_type = param_info.get('type', 'str')
                    
                    if param_type == 'bool':
                        value = inquirer.confirm(f"Set {param_name}", default=bool(default_val) if default_val else False)
                    else:
                        prompt_text = f"Enter {param_name}"
                        if default_val is not None:
                            prompt_text += f" (default: {default_val})"
                        
                        value_str = inquirer.text(prompt_text, default=str(default_val) if default_val else "")
                        
                        # Convert to appropriate type
                        try:
                            if param_type == 'int':
                                value = int(value_str) if value_str else (default_val if default_val is not None else 0)
                            elif param_type == 'float':
                                value = float(value_str) if value_str else (default_val if default_val is not None else 0.0)
                            else:
                                value = value_str if value_str else (default_val if default_val is not None else "")
                        except ValueError:
                            value = default_val if default_val is not None else ""
                            print(f"⚠️  Invalid value for {param_name}, using default: {value}")
                    
                    config['configuration']['model']['model_parameters'][param_name] = value
        else:
            # Standard model selection
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
        
        # Loss Function Configuration - using improved workflow
        loss_functions_config = self.configure_loss_functions(config)
        config['configuration']['model']['loss_functions'] = loss_functions_config
        
        # Metrics Configuration - using improved workflow similar to loss functions
        metrics_config = self.configure_metrics(config, loss_functions_config)
        config['configuration']['model']['metrics'] = metrics_config
        
        # Callbacks Configuration
        callbacks_config = self.configure_callbacks(config)
        config['configuration']['model']['callbacks'] = callbacks_config
        
        # Training Configuration
        print("\n🏃 Training Configuration")
        epochs = inquirer.text("Enter number of epochs", default="100")
        try:
            config['configuration']['training']['epochs'] = int(epochs)
        except ValueError:
            config['configuration']['training']['epochs'] = 100
        
        # Cross-Validation Configuration
        print("\n🔄 Cross-Validation Configuration")
        enable_cv = inquirer.confirm("Enable cross-validation?", default=False)
        if enable_cv:
            cv_folds = inquirer.text("Enter number of k-folds", default="5")
            try:
                config['configuration']['training']['cross_validation']['enabled'] = True
                config['configuration']['training']['cross_validation']['k_folds'] = int(cv_folds)
            except ValueError:
                config['configuration']['training']['cross_validation']['k_folds'] = 5
            
            stratified = inquirer.confirm("Use stratified cross-validation?", default=True)
            config['configuration']['training']['cross_validation']['stratified'] = stratified
            
            shuffle_cv = inquirer.confirm("Shuffle data for cross-validation?", default=True)
            config['configuration']['training']['cross_validation']['shuffle'] = shuffle_cv
            
            random_seed = inquirer.text("Enter random seed for reproducibility", default="42")
            try:
                config['configuration']['training']['cross_validation']['random_seed'] = int(random_seed)
            except ValueError:
                config['configuration']['training']['cross_validation']['random_seed'] = 42
            
            save_fold_models = inquirer.confirm("Save individual fold models?", default=False)
            config['configuration']['training']['cross_validation']['save_fold_models'] = save_fold_models
            
            if save_fold_models:
                fold_models_dir = inquirer.text("Enter directory for fold models", default="./logs/fold_models")
                config['configuration']['training']['cross_validation']['fold_models_dir'] = fold_models_dir
            
            aggregate_metrics = inquirer.confirm("Aggregate metrics across folds?", default=True)
            config['configuration']['training']['cross_validation']['aggregate_metrics'] = aggregate_metrics
            
            if aggregate_metrics:
                selection_metric = inquirer.text("Enter metric for fold selection", default="val_accuracy")
                config['configuration']['training']['cross_validation']['fold_selection_metric'] = selection_metric
        else:
            config['configuration']['training']['cross_validation']['enabled'] = False
        
        # Runtime Configuration
        print("\n⚙️  Runtime Configuration")
        model_dir = inquirer.text("Enter model output directory", default="./logs")
        config['configuration']['runtime']['model_dir'] = model_dir
        
        # Mixed Precision Configuration
        print("\n🔢 Mixed Precision Configuration")
        mixed_precision_choices = [
            ('None (Full precision)', None),
            ('mixed_float16 (Automatic mixed precision)', 'mixed_float16'),
            ('mixed_bfloat16 (Brain floating point 16)', 'mixed_bfloat16')
        ]
        
        mixed_precision_question = [
            inquirer.List('mixed_precision',
                         message="Select mixed precision policy",
                         choices=mixed_precision_choices,
                         default=None)
        ]
        mixed_precision_answer = inquirer.prompt(mixed_precision_question)
        config['configuration']['runtime']['mixed_precision'] = mixed_precision_answer['mixed_precision']
        
        if mixed_precision_answer['mixed_precision']:
            print(f"✅ Mixed precision enabled: {mixed_precision_answer['mixed_precision']}")
            print("💡 This can improve training speed and reduce memory usage on compatible hardware")
        else:
            print("✅ Using full precision (default)")
        
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
                    if hf.is_improved_template_config(config):
                        # Generate user-friendly YAML with comments
                        yaml_content = hf.generate_improved_yaml(config)
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
            hf.generate_python_scripts(config, file_path)
            
            return True
        except Exception as e:
            print(f"❌ Error saving configuration: {str(e)}")
            return False

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
        hf.copy_example_data(project_dir)
        
        # Create the improved template with custom functions and parameters
        # Note: Custom modules and scripts will be generated when save_config is called
        template_config = hf.create_improved_template_config(config, project_dir)
        
        if self.save_config(template_config, template_path, format_type):
            print(f"✅ Template created at: {template_path}")
            print(" Sample data copied to: ./data/")
            print("🚀 Ready to train! The template includes working custom functions and sample data")
            print("💡 Run the generated train.py script to start training")

    def interactive_configuration_with_existing(self, existing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive configuration using existing config as base."""
        print("\n🌱 ModelGardener CLI Configuration Modifier")
        print("=" * 50)
        print("✨ Current configuration will be used as the starting point")
        
        config = existing_config.copy()
        
        # Show current task type
        current_task = config.get('configuration', {}).get('task_type', 'image_classification')
        print(f"\n📋 Current task type: {current_task}")
        
        # Task Type Selection
        task_types = ['image_classification', 'object_detection', 'semantic_segmentation']
        change_task = inquirer.confirm(f"Change task type from '{current_task}'?", default=False)
        
        if change_task:
            task_type = inquirer.list_input(
                "Select new task type",
                choices=task_types,
                default=current_task
            )
            config['configuration']['task_type'] = task_type
        
        # Data Configuration
        current_train_dir = config.get('configuration', {}).get('data', {}).get('train_dir', './example_data/train')
        current_val_dir = config.get('configuration', {}).get('data', {}).get('val_dir', './example_data/val')
        
        print(f"\n📁 Current Data Configuration:")
        print(f"   Training directory: {current_train_dir}")
        print(f"   Validation directory: {current_val_dir}")
        
        change_data = inquirer.confirm("Modify data configuration?", default=False)
        
        if change_data:
            train_dir = inquirer.text("Enter training data directory", default=current_train_dir)
            val_dir = inquirer.text("Enter validation data directory", default=current_val_dir)
            
            config['configuration']['data']['train_dir'] = train_dir
            config['configuration']['data']['val_dir'] = val_dir
            
            # Data Loader Selection
            current_data_loader = config.get('configuration', {}).get('data', {}).get('data_loader', {}).get('selected_data_loader', 'ImageDataGenerator')
            print(f"\n📊 Current Data Loader: {current_data_loader}")
            
            change_data_loader = inquirer.confirm("Change data loader?", default=False)
            if change_data_loader:
                data_loader = inquirer.list_input(
                    "Select data loader",
                    choices=self.available_data_loaders,
                    default=current_data_loader if current_data_loader in self.available_data_loaders else 'ImageDataGenerator'
                )
                config['configuration']['data']['data_loader']['selected_data_loader'] = data_loader
                
                # Handle custom data loader selection
                if data_loader == 'Custom':
                    print("\n🔧 Custom Data Loader Configuration")
                    custom_data_loader_path = inquirer.text(
                        "Enter path to Python file containing custom data loader"
                    )
                    
                    if not custom_data_loader_path or not os.path.exists(custom_data_loader_path):
                        print("❌ Invalid file path. Using default data loader.")
                        data_loader_name = 'ImageDataGenerator'
                        data_loader_params = {}
                    else:
                        # Analyze custom data loader file
                        success, loader_info = self.analyze_custom_data_loader_file(custom_data_loader_path)
                        
                        if not success or not loader_info:
                            print("❌ No valid data loader functions found in the file. Using default data loader.")
                            data_loader_name = 'ImageDataGenerator'
                            data_loader_params = {}
                        else:
                            print(f"\n✅ Found {len(loader_info)} data loader function(s) in {custom_data_loader_path}")
                            
                            # Let user select from available data loaders
                            data_loader_name, data_loader_params = self.interactive_custom_data_loader_selection(custom_data_loader_path)
                            
                            # Add custom data loader path to config
                            config['configuration']['data']['data_loader']['custom_data_loader_path'] = custom_data_loader_path
                    
                    config['configuration']['data']['data_loader']['selected_data_loader'] = data_loader_name or 'ImageDataGenerator'
                    
                    # Update data loader parameters if available
                    if data_loader_params and 'user_parameters' in data_loader_params:
                        if 'parameters' not in config['configuration']['data']['data_loader']:
                            config['configuration']['data']['data_loader']['parameters'] = {}
                        config['configuration']['data']['data_loader']['parameters'].update(data_loader_params['user_parameters'])
        
        # Batch size
        current_batch_size = config.get('configuration', {}).get('data', {}).get('data_loader', {}).get('parameters', {}).get('batch_size', 32)
        print(f"\n📦 Current batch size: {current_batch_size}")
        
        change_batch = inquirer.confirm("Change batch size?", default=False)
        if change_batch:
            batch_size = inquirer.text("Enter batch size", default=str(current_batch_size))
            try:
                config['configuration']['data']['data_loader']['parameters']['batch_size'] = int(batch_size)
            except ValueError:
                print("⚠️  Invalid batch size, keeping current value")
        
        # Preprocessing Configuration
        current_preprocessing = config.get('configuration', {}).get('data', {}).get('preprocessing', {})
        current_resizing_enabled = current_preprocessing.get('Resizing', {}).get('enabled', False)
        current_normalization_enabled = current_preprocessing.get('Normalization', {}).get('enabled', True)
        
        print(f"\n🔧 Current Preprocessing:")
        print(f"   Resizing: {'Enabled' if current_resizing_enabled else 'Disabled'}")
        print(f"   Normalization: {'Enabled' if current_normalization_enabled else 'Disabled'}")
        
        change_preprocessing = inquirer.confirm("Configure preprocessing?", default=False)
        if change_preprocessing:
            preprocessing_config = self.configure_preprocessing(config)
            config['configuration']['data']['preprocessing'] = preprocessing_config
        
        # Model Configuration
        current_family = config.get('configuration', {}).get('model', {}).get('model_family', 'resnet')
        current_model = config.get('configuration', {}).get('model', {}).get('model_name', 'ResNet-50')
        
        print(f"\n🤖 Current Model: {current_family} - {current_model}")
        
        change_model = inquirer.confirm("Change model?", default=False)
        
        if change_model:
            # Model family selection
            model_families = list(self.available_models.keys())
            model_family = inquirer.list_input(
                "Select model family",
                choices=model_families,
                default=current_family if current_family in model_families else 'resnet'
            )
            config['configuration']['model']['model_family'] = model_family
            
            if model_family == 'custom':
                # Custom model handling
                print("\n🔧 Custom Model Configuration")
                print("=" * 40)
                
                custom_model_path = inquirer.text("Enter path to custom model Python file")
                
                if not custom_model_path or not os.path.exists(custom_model_path):
                    print("❌ Invalid file path. Using default custom model.")
                    model_name = 'CustomModel'
                    model_params = {}
                else:
                    # Analyze custom model file
                    success, model_info = self.analyze_custom_model_file(custom_model_path)
                    
                    if not success or not model_info:
                        print("❌ No valid model functions found in the file. Using default custom model.")
                        model_name = 'CustomModel'
                        model_params = {}
                    else:
                        print(f"\n✅ Found {len(model_info)} model function(s) in {custom_model_path}")
                        
                        # Let user select from available models
                        model_name, model_params = self.interactive_custom_model_selection(custom_model_path)
                        
                        # Add custom model path to config
                        config['configuration']['model']['custom_model_path'] = custom_model_path
                
                config['configuration']['model']['model_name'] = model_name
                
                # Update model parameters if available
                if model_params and 'parameters' in model_params:
                    if 'model_parameters' not in config['configuration']['model']:
                        config['configuration']['model']['model_parameters'] = {}
                    config['configuration']['model']['model_parameters'].update(model_params['parameters'])
            else:
                # Standard model name selection
                model_names = self.available_models[model_family]
                model_name = inquirer.list_input(
                    f"Select {model_family} model",
                    choices=model_names,
                    default=current_model if current_model in model_names else model_names[0]
                )
                config['configuration']['model']['model_name'] = model_name
        
        # Input shape configuration
        current_shape = config.get('configuration', {}).get('model', {}).get('model_parameters', {}).get('input_shape', {})
        current_height = current_shape.get('height', 224)
        current_width = current_shape.get('width', 224)
        current_channels = current_shape.get('channels', 3)
        
        print(f"\n📐 Current Input Shape: {current_height}x{current_width}x{current_channels}")
        
        change_shape = inquirer.confirm("Change input shape?", default=False)
        
        if change_shape:
            height = inquirer.text("Enter image height", default=str(current_height))
            width = inquirer.text("Enter image width", default=str(current_width))
            channels = inquirer.text("Enter image channels", default=str(current_channels))
            
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
                print("⚠️  Invalid input shape values, keeping current values")
        
        # Number of classes
        current_classes = config.get('configuration', {}).get('model', {}).get('model_parameters', {}).get('classes', 1000)
        print(f"\n🔢 Current number of classes: {current_classes}")
        
        change_classes = inquirer.confirm("Change number of classes?", default=False)
        if change_classes:
            num_classes = inquirer.text("Enter number of classes", default=str(current_classes))
            try:
                config['configuration']['model']['model_parameters']['classes'] = int(num_classes)
            except ValueError:
                print("⚠️  Invalid number of classes, keeping current value")
        
        # Optimizer Configuration
        current_optimizer = config.get('configuration', {}).get('model', {}).get('optimizer', {}).get('Optimizer Selection', {}).get('selected_optimizer', 'Adam')
        current_lr = config.get('configuration', {}).get('model', {}).get('optimizer', {}).get('Optimizer Selection', {}).get('learning_rate', 0.001)
        
        print(f"\n⚡ Current Optimizer: {current_optimizer} (lr: {current_lr})")
        
        change_optimizer = inquirer.confirm("Change optimizer settings?", default=False)
        
        if change_optimizer:
            optimizer = inquirer.list_input(
                "Select optimizer",
                choices=self.available_optimizers,
                default=current_optimizer if current_optimizer in self.available_optimizers else 'Adam'
            )
            config['configuration']['model']['optimizer']['Optimizer Selection']['selected_optimizer'] = optimizer
            
            learning_rate = inquirer.text("Enter learning rate", default=str(current_lr))
            try:
                config['configuration']['model']['optimizer']['Optimizer Selection']['learning_rate'] = float(learning_rate)
            except ValueError:
                print("⚠️  Invalid learning rate, keeping current value")
        
        # Loss Function Configuration
        current_loss_config = config.get('configuration', {}).get('model', {}).get('loss_functions', {})
        current_loss = current_loss_config.get('Loss Selection', {}).get('selected_loss', 'Categorical Crossentropy')
        current_num_outputs = current_loss_config.get('Model Output Configuration', {}).get('num_outputs', 1)
        current_loss_strategy = current_loss_config.get('Model Output Configuration', {}).get('loss_strategy', 'single_loss_all_outputs')
        
        print(f"\n📉 Current Loss Function: {current_loss}")
        print(f"    Number of outputs: {current_num_outputs}")
        print(f"    Loss strategy: {current_loss_strategy}")
        
        change_loss = inquirer.confirm("Change loss function configuration?", default=False)
        if change_loss:
            # Use improved loss configuration workflow
            loss_functions_config = self.configure_loss_functions(config)
            config['configuration']['model']['loss_functions'] = loss_functions_config
        
        # Metrics Configuration - Enhanced metrics workflow
        current_metrics_config = config.get('configuration', {}).get('model', {}).get('metrics', {})
        current_metrics = current_metrics_config.get('Metrics Selection', {}).get('selected_metrics', 'Accuracy')
        current_metrics_num_outputs = current_metrics_config.get('Model Output Configuration', {}).get('num_outputs', 1)
        current_metrics_strategy = current_metrics_config.get('Model Output Configuration', {}).get('metrics_strategy', 'shared_metrics_all_outputs')
        
        print(f"\n📈 Current Metrics: {current_metrics}")
        print(f"    Number of outputs: {current_metrics_num_outputs}")
        print(f"    Metrics strategy: {current_metrics_strategy}")
        
        change_metrics = inquirer.confirm("Change metrics configuration?", default=False)
        if change_metrics:
            # Use enhanced metrics configuration workflow - reuse loss functions config
            current_loss_config = config.get('configuration', {}).get('model', {}).get('loss_functions', {})
            metrics_config = self.configure_metrics(config, current_loss_config)
            config['configuration']['model']['metrics'] = metrics_config
        
        # Callbacks Configuration
        current_callbacks_config = config.get('configuration', {}).get('model', {}).get('callbacks', {})
        current_early_stopping = current_callbacks_config.get('Early Stopping', {}).get('enabled', False)
        current_lr_scheduler = current_callbacks_config.get('Learning Rate Scheduler', {}).get('enabled', False)
        current_checkpoint = current_callbacks_config.get('Model Checkpoint', {}).get('enabled', True)
        current_tensorboard = current_callbacks_config.get('TensorBoard', {}).get('enabled', False)
        current_csv_logger = current_callbacks_config.get('CSV Logger', {}).get('enabled', False)
        current_custom_callbacks = current_callbacks_config.get('Custom Callbacks', {}).get('enabled', False)
        
        print(f"\n📞 Current Callbacks Configuration:")
        print(f"    Early Stopping: {'Enabled' if current_early_stopping else 'Disabled'}")
        print(f"    LR Scheduler: {'Enabled' if current_lr_scheduler else 'Disabled'}")
        print(f"    Model Checkpoint: {'Enabled' if current_checkpoint else 'Disabled'}")
        print(f"    TensorBoard: {'Enabled' if current_tensorboard else 'Disabled'}")
        print(f"    CSV Logger: {'Enabled' if current_csv_logger else 'Disabled'}")
        print(f"    Custom Callbacks: {'Enabled' if current_custom_callbacks else 'Disabled'}")
        
        change_callbacks = inquirer.confirm("Change callbacks configuration?", default=False)
        if change_callbacks:
            callbacks_config = self.configure_callbacks(config)
            config['configuration']['model']['callbacks'] = callbacks_config
        
        # Training Configuration
        current_epochs = config.get('configuration', {}).get('training', {}).get('epochs', 10)
        
        print(f"\n🏃 Current training epochs: {current_epochs}")
        
        change_training = inquirer.confirm("Change training settings?", default=False)
        if change_training:
            epochs = inquirer.text("Enter number of epochs", default=str(current_epochs))
            try:
                config['configuration']['training']['epochs'] = int(epochs)
            except ValueError:
                print("⚠️  Invalid number of epochs, keeping current value")
        
        # Runtime Configuration
        current_model_dir = config.get('configuration', {}).get('runtime', {}).get('model_dir', './logs')
        current_gpus = config.get('configuration', {}).get('runtime', {}).get('num_gpus', 1)
        
        print(f"\n💾 Current Runtime Settings:")
        print(f"   Model directory: {current_model_dir}")
        print(f"   Number of GPUs: {current_gpus}")
        
        change_runtime = inquirer.confirm("Change runtime settings?", default=False)
        if change_runtime:
            model_dir = inquirer.text("Enter model output directory", default=current_model_dir)
            num_gpus = inquirer.text("Enter number of GPUs", default=str(current_gpus))
            config['configuration']['runtime']['model_dir'] = model_dir
            try:
                config['configuration']['runtime']['num_gpus'] = int(num_gpus)
            except ValueError:
                print("⚠️  Invalid number of GPUs, keeping current value")
        
        # Update modification timestamp
        from datetime import datetime
        if 'metadata' not in config:
            config['metadata'] = {}
        config['metadata']['last_modified'] = datetime.now().isoformat()
        config['metadata']['modified_via'] = 'CLI Interactive Mode'
        
        print("\n✅ Configuration modification completed!")
        
        return config

    def batch_configuration_with_existing(self, existing_config: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply batch modifications to existing configuration."""
        config = existing_config.copy()
        
        print("⚡ Applying batch modifications to existing configuration...")
        
        # Apply modifications
        if 'train_dir' in modifications:
            config['configuration']['data']['train_dir'] = modifications['train_dir']
            print(f"  ✓ Updated training directory: {modifications['train_dir']}")
        
        if 'val_dir' in modifications:
            config['configuration']['data']['val_dir'] = modifications['val_dir']
            print(f"  ✓ Updated validation directory: {modifications['val_dir']}")
        
        if 'batch_size' in modifications:
            config['configuration']['data']['data_loader']['parameters']['batch_size'] = modifications['batch_size']
            print(f"  ✓ Updated batch size: {modifications['batch_size']}")
        
        if 'model_family' in modifications:
            config['configuration']['model']['model_family'] = modifications['model_family']
            print(f"  ✓ Updated model family: {modifications['model_family']}")
        
        if 'model_name' in modifications:
            config['configuration']['model']['model_name'] = modifications['model_name']
            print(f"  ✓ Updated model name: {modifications['model_name']}")
        
        if 'epochs' in modifications:
            config['configuration']['training']['epochs'] = modifications['epochs']
            print(f"  ✓ Updated epochs: {modifications['epochs']}")
        
        if 'learning_rate' in modifications:
            config['configuration']['model']['optimizer']['Optimizer Selection']['learning_rate'] = modifications['learning_rate']
            print(f"  ✓ Updated learning rate: {modifications['learning_rate']}")
        
        if 'optimizer' in modifications:
            config['configuration']['model']['optimizer']['Optimizer Selection']['selected_optimizer'] = modifications['optimizer']
            print(f"  ✓ Updated optimizer: {modifications['optimizer']}")
        
        if 'loss_function' in modifications:
            config['configuration']['model']['loss_functions']['Loss Selection']['selected_loss'] = modifications['loss_function']
            print(f"  ✓ Updated loss function: {modifications['loss_function']}")
        
        if 'num_classes' in modifications:
            config['configuration']['model']['model_parameters']['classes'] = modifications['num_classes']
            print(f"  ✓ Updated number of classes: {modifications['num_classes']}")
        
        if 'model_dir' in modifications:
            config['configuration']['runtime']['model_dir'] = modifications['model_dir']
            print(f"  ✓ Updated model directory: {modifications['model_dir']}")
        
        if 'num_gpus' in modifications:
            config['configuration']['runtime']['num_gpus'] = modifications['num_gpus']
            print(f"  ✓ Updated number of GPUs: {modifications['num_gpus']}")
        
        # Update modification timestamp
        from datetime import datetime
        if 'metadata' not in config:
            config['metadata'] = {}
        config['metadata']['last_modified'] = datetime.now().isoformat()
        config['metadata']['modified_via'] = 'CLI Batch Mode'
        
        print("✅ Batch modifications applied successfully!")
        
        return config

    def analyze_custom_preprocessing_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom preprocessing functions.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (success, preprocessing_info)
        """
        import importlib.util
        import inspect
        
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_preprocessing", file_path)
            if spec is None or spec.loader is None:
                return False, {}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            preprocessing_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if hf.is_preprocessing_function(obj, name):
                    info = hf.extract_preprocessing_parameters(obj)
                    if info:
                        preprocessing_info[name] = info
            
            return len(preprocessing_info) > 0, preprocessing_info
            
        except Exception as e:
            print(f"❌ Error analyzing preprocessing file: {str(e)}")
            return False, {}

    def interactive_custom_preprocessing_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom preprocessing functions from analyzed file.
        
        Args:
            file_path: Path to the Python file containing preprocessing functions
            
        Returns:
            Tuple of (selected_function_name, function_info_with_user_params)
        """
        success, analysis_result = self.analyze_custom_preprocessing_file(file_path)
        
        if not success:
            print("❌ No valid preprocessing functions found in the file")
            return None, {}
        
        if not analysis_result:
            print("❌ No preprocessing functions found in the file")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} preprocessing function(s):")
        for name, info in analysis_result.items():
            print(f"   • {name}: {info.get('description', 'No description')}")
        
        # Let user select function
        function_names = list(analysis_result.keys())
        selected_function = inquirer.list_input(
            "Select preprocessing function",
            choices=function_names
        )
        
        if not selected_function:
            return None, {}
        
        function_info = analysis_result[selected_function]
        user_parameters = {}
        
        # Configure parameters for selected function
        if function_info.get('parameters'):
            print(f"\n⚙️  Configuring parameters for {selected_function}:")
            
            for param_name, param_info in function_info['parameters'].items():
                param_type = param_info.get('type', 'str')
                default_val = param_info.get('default')
                
                if param_type == 'bool':
                    value = inquirer.confirm(
                        f"Set {param_name}",
                        default=bool(default_val) if default_val is not None else False
                    )
                else:
                    prompt_text = f"Enter {param_name}"
                    if default_val is not None:
                        prompt_text += f" (default: {default_val})"
                    
                    value_str = inquirer.text(prompt_text, default=str(default_val) if default_val is not None else "")
                    
                    # Convert to appropriate type
                    try:
                        if param_type == 'int':
                            value = int(value_str) if value_str else (default_val if default_val is not None else 0)
                        elif param_type == 'float':
                            value = float(value_str) if value_str else (default_val if default_val is not None else 0.0)
                        elif param_type == 'list':
                            if value_str:
                                # Try to parse as list
                                try:
                                    value = ast.literal_eval(value_str)
                                except:
                                    value = value_str.split(',') if ',' in value_str else [value_str]
                            else:
                                value = default_val if default_val is not None else []
                        else:
                            value = value_str if value_str else (default_val if default_val is not None else "")
                    except ValueError:
                        value = default_val if default_val is not None else ""
                        print(f"⚠️  Invalid value for {param_name}, using default: {value}")
                
                user_parameters[param_name] = value
        
        # Return function info with user parameters
        result_info = function_info.copy()
        result_info['user_parameters'] = user_parameters
        result_info['file_path'] = file_path
        
        return selected_function, result_info

    def configure_preprocessing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive configuration of preprocessing settings.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Updated preprocessing configuration
        """
        print("\n🔧 Preprocessing Configuration")
        print("=" * 40)
        print("📋 Preprocessing transforms your raw data into a format suitable for training.")
        print("💡 Common steps: Resizing → Normalization → Data Format Conversion")
        
        preprocessing_config = {
            "Resizing": {
                "enabled": False,
                "target_size": {"width": 224, "height": 224, "depth": 1},
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
        }
        
        # 1. Configure Resizing
        print("\n📏 Step 1: Image Resizing")
        print("🎯 Most models expect fixed-size inputs (e.g., 224x224 for many pre-trained models)")
        
        # Select resizing strategy
        resizing_strategies = [
            'None (Disable resizing)',
            'Scaling (Resize to target size)', 
            'Pad-Cropping (Crop/pad to target size)'
        ]
        
        resizing_strategy = inquirer.list_input(
            "Select resizing strategy",
            choices=resizing_strategies,
            default='Scaling (Resize to target size)'
        )
        
        if resizing_strategy.startswith('None'):
            preprocessing_config["Resizing"]["enabled"] = False
            print("✅ Resizing disabled - original image sizes will be preserved")
        else:
            preprocessing_config["Resizing"]["enabled"] = True
            
            # Configure method-specific options
            if resizing_strategy.startswith('Scaling'):
                print("\n🔍 Scaling Method Configuration")
                scaling_modes = [
                    'nearest (Fast, blocky results)',
                    'bilinear (Good balance of speed and quality)', 
                    'bicubic (High quality, slower)',
                    'area (Good for downscaling)',
                    'lanczos (Highest quality, slowest)'
                ]
                
                scaling_mode = inquirer.list_input(
                    "Select scaling interpolation method",
                    choices=scaling_modes,
                    default='bilinear (Good balance of speed and quality)'
                )
                
                # Extract the method name
                method_name = scaling_mode.split(' ')[0]
                preprocessing_config["Resizing"]["interpolation"] = method_name
                preprocessing_config["Resizing"]["method"] = "scaling"
                
                # Ask about preserving aspect ratio
                preserve_aspect = inquirer.confirm(
                    "Preserve aspect ratio? (May result in different final size)",
                    default=True
                )
                preprocessing_config["Resizing"]["preserve_aspect_ratio"] = preserve_aspect
                
            elif resizing_strategy.startswith('Pad-Cropping'):
                print("\n✂️  Pad-Cropping Method Configuration")
                crop_modes = [
                    'center (Crop/pad from center)',
                    'random (Random crop/pad position - good for augmentation)'
                ]
                
                crop_mode = inquirer.list_input(
                    "Select crop/pad positioning",
                    choices=crop_modes,
                    default='center (Crop/pad from center)'
                )
                
                # Extract the method name
                method_name = crop_mode.split(' ')[0]
                preprocessing_config["Resizing"]["crop_method"] = method_name
                preprocessing_config["Resizing"]["method"] = "pad_crop"
                preprocessing_config["Resizing"]["interpolation"] = "nearest"  # Default for cropping
                
                # Ask about padding value
                pad_value = inquirer.text(
                    "Enter padding value (0-255 for RGB, 0.0-1.0 for normalized)",
                    default="0"
                )
                try:
                    preprocessing_config["Resizing"]["pad_value"] = float(pad_value)
                except ValueError:
                    preprocessing_config["Resizing"]["pad_value"] = 0.0
                    print("⚠️  Invalid padding value, using 0")
            
            # Configure target dimensions
            print("\n📐 Target Dimensions Configuration")
            
            # Ask about data format first
            data_format = inquirer.list_input(
                "Select data format",
                choices=['2D (images)', '3D (volumes/sequences)'],
                default='2D (images)'
            )
            
            if data_format == '3D (volumes/sequences)':
                preprocessing_config["Resizing"]["data_format"] = "3D"
                
                # Get 3D dimensions
                width = inquirer.text("Enter target width", default="224")
                height = inquirer.text("Enter target height", default="224") 
                depth = inquirer.text("Enter target depth (temporal/z dimension)", default="16")
                
                try:
                    preprocessing_config["Resizing"]["target_size"] = {
                        "width": int(width),
                        "height": int(height),
                        "depth": int(depth)
                    }
                    print(f"✅ 3D resizing configured: {width}x{height}x{depth}")
                except ValueError:
                    preprocessing_config["Resizing"]["target_size"] = {
                        "width": 224,
                        "height": 224,
                        "depth": 16
                    }
                    print("⚠️  Invalid dimensions, using 224x224x16")
            else:
                preprocessing_config["Resizing"]["data_format"] = "2D"
                
                # Preset or custom size for 2D
                size_options = [
                    '224x224 (Standard - ResNet, VGG)',
                    '299x299 (Inception networks)',
                    '512x512 (High resolution)',
                    '128x128 (Lightweight models)',
                    '32x32 (CIFAR-like datasets)',
                    'Custom size'
                ]
                
                size_choice = inquirer.list_input(
                    "Select target image size",
                    choices=size_options,
                    default='224x224 (Standard - ResNet, VGG)'
                )
                
                if size_choice.startswith('224x224'):
                    width, height = 224, 224
                elif size_choice.startswith('299x299'):
                    width, height = 299, 299
                elif size_choice.startswith('512x512'):
                    width, height = 512, 512
                elif size_choice.startswith('128x128'):
                    width, height = 128, 128
                elif size_choice.startswith('32x32'):
                    width, height = 32, 32
                else:  # Custom size
                    width = inquirer.text("Enter target width", default="224")
                    height = inquirer.text("Enter target height", default="224")
                    try:
                        width, height = int(width), int(height)
                    except ValueError:
                        print("⚠️  Invalid dimensions, using 224x224")
                        width, height = 224, 224
                
                preprocessing_config["Resizing"]["target_size"] = {
                    "width": width,
                    "height": height,
                    "depth": 1
                }
                print(f"✅ 2D resizing configured: {width}x{height}")
        
        # 2. Configure Normalization
        print("\n📊 Step 2: Data Normalization")
        print("🎯 Normalization scales pixel values to a standard range for better training")
        
        # Select normalization method including None option
        normalization_methods = [
            'None (Disable normalization - use raw pixel values)',
            'min-max (Scale to [min, max] range)',
            'zero-center (Subtract mean, divide by std)',
            'unit-norm (Normalize to unit vector)',
            'robust (Use median and IQR for outlier resistance)',
            'standard (Same as zero-center)',
            'layer-norm (Normalize across feature dimensions)'
        ]
        
        norm_method = inquirer.list_input(
            "Select normalization method",
            choices=normalization_methods,
            default='zero-center (Subtract mean, divide by std)'
        )
        
        if norm_method.startswith('None'):
            preprocessing_config["Normalization"]["enabled"] = False
            print("✅ Normalization disabled - raw pixel values will be used")
        else:
            preprocessing_config["Normalization"]["enabled"] = True
            
            # Extract method name and configure parameters
            method_name = norm_method.split(' ')[0]
            preprocessing_config["Normalization"]["method"] = method_name
            
            if method_name == 'min-max':
                print("\n⚙️  Min-Max Normalization Parameters")
                print("📝 Formula: (x - min_value) / (max_value - min_value)")
                
                # Common presets or custom
                minmax_presets = [
                    '[0, 1] (Standard for neural networks)',
                    '[-1, 1] (Common for GANs and some models)',
                    'Custom range'
                ]
                
                preset_choice = inquirer.list_input(
                    "Select min-max range",
                    choices=minmax_presets,
                    default='[0, 1] (Standard for neural networks)'
                )
                
                if preset_choice.startswith('[0, 1]'):
                    min_val, max_val = 0.0, 1.0
                elif preset_choice.startswith('[-1, 1]'):
                    min_val, max_val = -1.0, 1.0
                else:  # Custom range
                    min_val = inquirer.text("Enter minimum value", default="0.0")
                    max_val = inquirer.text("Enter maximum value", default="1.0")
                    try:
                        min_val, max_val = float(min_val), float(max_val)
                    except ValueError:
                        min_val, max_val = 0.0, 1.0
                        print("⚠️  Invalid values, using [0.0, 1.0]")
                
                preprocessing_config["Normalization"]["min_value"] = min_val
                preprocessing_config["Normalization"]["max_value"] = max_val
                print(f"✅ Min-max normalization: [{min_val}, {max_val}]")
                
            elif method_name in ['zero-center', 'standard']:
                print("\n⚙️  Zero-Center Normalization Parameters")
                print("📝 Formula: (x - mean) / std")
                
                # Preset statistics or custom
                stats_presets = [
                    'ImageNet (R: 0.485±0.229, G: 0.456±0.224, B: 0.406±0.225)',
                    'CIFAR-10 (R: 0.491±0.247, G: 0.482±0.243, B: 0.447±0.262)',
                    'Custom statistics',
                    'Compute from data (placeholder - not implemented yet)'
                ]
                
                stats_choice = inquirer.list_input(
                    "Select normalization statistics",
                    choices=stats_presets,
                    default='ImageNet (R: 0.485±0.229, G: 0.456±0.224, B: 0.406±0.225)'
                )
                
                if stats_choice.startswith('ImageNet'):
                    preprocessing_config["Normalization"]["mean"] = {"r": 0.485, "g": 0.456, "b": 0.406}
                    preprocessing_config["Normalization"]["std"] = {"r": 0.229, "g": 0.224, "b": 0.225}
                    print("✅ Using ImageNet statistics (best for transfer learning)")
                    
                elif stats_choice.startswith('CIFAR-10'):
                    preprocessing_config["Normalization"]["mean"] = {"r": 0.491, "g": 0.482, "b": 0.447}
                    preprocessing_config["Normalization"]["std"] = {"r": 0.247, "g": 0.243, "b": 0.262}
                    print("✅ Using CIFAR-10 statistics")
                    
                elif stats_choice.startswith('Custom'):
                    print("📝 Enter custom normalization statistics:")
                    
                    # Check if grayscale or RGB
                    color_mode = inquirer.list_input(
                        "Select color mode",
                        choices=['RGB (3 channels)', 'Grayscale (1 channel)', 'Other (specify channels)'],
                        default='RGB (3 channels)'
                    )
                    
                    if color_mode.startswith('RGB'):
                        mean_r = inquirer.text("Enter mean for R channel", default="0.485")
                        mean_g = inquirer.text("Enter mean for G channel", default="0.456")
                        mean_b = inquirer.text("Enter mean for B channel", default="0.406")
                        
                        std_r = inquirer.text("Enter std for R channel", default="0.229")
                        std_g = inquirer.text("Enter std for G channel", default="0.224")
                        std_b = inquirer.text("Enter std for B channel", default="0.225")
                        
                        try:
                            preprocessing_config["Normalization"]["mean"] = {
                                "r": float(mean_r),
                                "g": float(mean_g),
                                "b": float(mean_b)
                            }
                            preprocessing_config["Normalization"]["std"] = {
                                "r": float(std_r),
                                "g": float(std_g),
                                "b": float(std_b)
                            }
                            print("✅ Custom RGB statistics configured")
                        except ValueError:
                            print("⚠️  Invalid values, using ImageNet defaults")
                            preprocessing_config["Normalization"]["mean"] = {"r": 0.485, "g": 0.456, "b": 0.406}
                            preprocessing_config["Normalization"]["std"] = {"r": 0.229, "g": 0.224, "b": 0.225}
                            
                    elif color_mode.startswith('Grayscale'):
                        mean_val = inquirer.text("Enter mean value", default="0.5")
                        std_val = inquirer.text("Enter std value", default="0.5")
                        
                        try:
                            preprocessing_config["Normalization"]["mean"] = float(mean_val)
                            preprocessing_config["Normalization"]["std"] = float(std_val)
                            print("✅ Custom grayscale statistics configured")
                        except ValueError:
                            preprocessing_config["Normalization"]["mean"] = 0.5
                            preprocessing_config["Normalization"]["std"] = 0.5
                            print("⚠️  Invalid values, using defaults (0.5, 0.5)")
                    
                    else:  # Other channels
                        num_channels = inquirer.text("Enter number of channels", default="3")
                        try:
                            num_channels = int(num_channels)
                            means = []
                            stds = []
                            
                            for i in range(num_channels):
                                mean = inquirer.text(f"Enter mean for channel {i+1}", default="0.5")
                                std = inquirer.text(f"Enter std for channel {i+1}", default="0.5")
                                means.append(float(mean))
                                stds.append(float(std))
                            
                            preprocessing_config["Normalization"]["mean"] = means
                            preprocessing_config["Normalization"]["std"] = stds
                            print(f"✅ Custom {num_channels}-channel statistics configured")
                        except ValueError:
                            print("⚠️  Invalid values, using ImageNet defaults")
                            preprocessing_config["Normalization"]["mean"] = {"r": 0.485, "g": 0.456, "b": 0.406}
                            preprocessing_config["Normalization"]["std"] = {"r": 0.229, "g": 0.224, "b": 0.225}
                
                else:  # Compute from data
                    print("⚠️  Computing statistics from data not yet implemented, using ImageNet defaults")
                    preprocessing_config["Normalization"]["mean"] = {"r": 0.485, "g": 0.456, "b": 0.406}
                    preprocessing_config["Normalization"]["std"] = {"r": 0.229, "g": 0.224, "b": 0.225}
                
                # Additional parameters for zero-center normalization
                axis = inquirer.text("Enter normalization axis (-1 for last axis)", default="-1")
                epsilon = inquirer.text("Enter epsilon value (for numerical stability)", default="1e-07")
                
                try:
                    preprocessing_config["Normalization"]["axis"] = int(axis)
                    preprocessing_config["Normalization"]["epsilon"] = float(epsilon)
                except ValueError:
                    preprocessing_config["Normalization"]["axis"] = -1
                    preprocessing_config["Normalization"]["epsilon"] = 1e-07
                    print("⚠️  Invalid axis/epsilon values, using defaults")
                    
            elif method_name == 'unit-norm':
                print("\n⚙️  Unit-Norm Normalization Parameters")
                print("📝 Formula: x / ||x||")
                
                norm_type = inquirer.list_input(
                    "Select norm type",
                    choices=['L2 (Euclidean norm)', 'L1 (Manhattan norm)', 'L-inf (Maximum norm)'],
                    default='L2 (Euclidean norm)'
                )
                
                ord_map = {'L2': 2, 'L1': 1, 'L-inf': float('inf')}
                ord_value = ord_map[norm_type.split(' ')[0]]
                preprocessing_config["Normalization"]["ord"] = ord_value
                
                axis = inquirer.text("Enter normalization axis (-1 for last axis)", default="-1")
                try:
                    preprocessing_config["Normalization"]["axis"] = int(axis)
                except ValueError:
                    preprocessing_config["Normalization"]["axis"] = -1
                
                print(f"✅ Unit-norm ({norm_type.split(' ')[0]}) normalization configured")
                
            elif method_name == 'robust':
                print("\n⚙️  Robust Normalization Parameters")
                print("📝 Formula: (x - median) / IQR")
                
                # IQR calculation method
                iqr_method = inquirer.list_input(
                    "Select IQR calculation method",
                    choices=['Standard (Q3 - Q1)', 'Modified (1.5 * IQR)'],
                    default='Standard (Q3 - Q1)'
                )
                
                preprocessing_config["Normalization"]["iqr_method"] = "standard" if iqr_method.startswith('Standard') else "modified"
                
                axis = inquirer.text("Enter normalization axis (-1 for last axis)", default="-1")
                try:
                    preprocessing_config["Normalization"]["axis"] = int(axis)
                except ValueError:
                    preprocessing_config["Normalization"]["axis"] = -1
                
                print("✅ Robust normalization configured")
                
            elif method_name == 'layer-norm':
                print("\n⚙️  Layer Normalization Parameters")
                print("📝 Normalizes across feature dimensions")
                
                epsilon = inquirer.text("Enter epsilon value (for numerical stability)", default="1e-05")
                center = inquirer.confirm("Center data (subtract mean)?", default=True)
                scale = inquirer.confirm("Scale data (divide by std)?", default=True)
                
                try:
                    preprocessing_config["Normalization"]["epsilon"] = float(epsilon)
                except ValueError:
                    preprocessing_config["Normalization"]["epsilon"] = 1e-05
                
                preprocessing_config["Normalization"]["center"] = center
                preprocessing_config["Normalization"]["scale"] = scale
                
                print("✅ Layer normalization configured")
        
        # 3. Custom Preprocessing (Optional)
        print("\n🛠️  Step 3: Custom Preprocessing (Optional)")
        use_custom_preprocessing = inquirer.confirm(
            "Add custom preprocessing functions?",
            default=False
        )
        
        if use_custom_preprocessing:
            custom_preprocessing_path = inquirer.text(
                "Enter path to Python file containing custom preprocessing functions"
            )
            
            if custom_preprocessing_path and os.path.exists(custom_preprocessing_path):
                # Try to analyze and select custom preprocessing
                selected_function, function_info = self.interactive_custom_preprocessing_selection(custom_preprocessing_path)
                
                if selected_function and function_info:
                    # Copy the custom function to custom_modules and get the relative path
                    relative_path = self.copy_custom_function_to_modules(custom_preprocessing_path, "preprocessing")
                    
                    preprocessing_config["Custom Preprocessing"] = {
                        "enabled": True,
                        "function_name": selected_function,
                        "file_path": relative_path,
                        "parameters": function_info.get('user_parameters', {})
                    }
                    print(f"✅ Custom preprocessing configured: {selected_function}")
                else:
                    print("⚠️  No valid custom preprocessing function selected")
            else:
                print("⚠️  Invalid file path for custom preprocessing")
        
        print(f"\n✅ Preprocessing configuration complete!")
        return preprocessing_config

    def configure_augmentation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive configuration of augmentation settings.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Updated augmentation configuration
        """
        print("\n🎲 Augmentation Configuration")
        print("=" * 40)
        print("📋 Data augmentation creates variations of your training data to improve model generalization.")
        
        augmentation_config = {
            "Horizontal Flip": {
                "enabled": False,
                "probability": 0.5
            },
            "Rotation": {
                "enabled": False,
                "angle_range": 15.0,
                "probability": 0.5
            },
            "Brightness": {
                "enabled": False,
                "factor": 0.2,
                "probability": 0.5
            },
            "Contrast": {
                "enabled": False,
                "factor": 0.2,
                "probability": 0.5
            }
        }
        
        # Ask which augmentations to enable
        augmentation_types = [
            'Horizontal Flip',
            'Rotation', 
            'Brightness',
            'Contrast'
        ]
        
        enabled_augmentations = inquirer.checkbox(
            "Select augmentations to enable (use spacebar to select, enter to confirm)",
            choices=augmentation_types
        )
        
        # Configure each selected augmentation
        for aug_type in enabled_augmentations:
            augmentation_config[aug_type]["enabled"] = True
            
            if aug_type == "Rotation":
                angle = inquirer.text(f"Enter rotation angle range (degrees)", default="15.0")
                try:
                    augmentation_config[aug_type]["angle_range"] = float(angle)
                except ValueError:
                    pass
            
            elif aug_type in ["Brightness", "Contrast"]:
                factor = inquirer.text(f"Enter {aug_type.lower()} factor", default="0.2")
                try:
                    augmentation_config[aug_type]["factor"] = float(factor)
                except ValueError:
                    pass
            
            # Set probability for all augmentations
            prob = inquirer.text(f"Enter probability for {aug_type}", default="0.5")
            try:
                augmentation_config[aug_type]["probability"] = float(prob)
            except ValueError:
                pass
        
        if enabled_augmentations:
            print(f"✅ Enabled augmentations: {', '.join(enabled_augmentations)}")
        else:
            print("✅ No augmentations enabled")
        
        return augmentation_config

    def analyze_custom_augmentation_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom augmentation functions.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (success, augmentation_info)
        """
        import importlib.util
        import inspect
        
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_augmentation", file_path)
            if spec is None or spec.loader is None:
                return False, {}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            augmentation_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if hf.is_augmentation_function(obj, name):
                    info = hf.extract_augmentation_parameters(obj)
                    if info:
                        augmentation_info[name] = info
            
            return len(augmentation_info) > 0, augmentation_info
            
        except Exception as e:
            print(f"❌ Error analyzing augmentation file: {str(e)}")
            return False, {}

    def interactive_custom_augmentation_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom augmentation functions from analyzed file.
        
        Args:
            file_path: Path to the Python file containing augmentation functions
            
        Returns:
            Tuple of (selected_function_name, function_info_with_user_params)
        """
        success, analysis_result = self.analyze_custom_augmentation_file(file_path)
        
        if not success:
            print("❌ No valid augmentation functions found in the file")
            return None, {}
        
        if not analysis_result:
            print("❌ No augmentation functions found in the file")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} augmentation function(s):")
        for name, info in analysis_result.items():
            print(f"   • {name}: {info.get('description', 'No description')}")
        
        # Let user select function
        function_names = list(analysis_result.keys())
        selected_function = inquirer.list_input(
            "Select augmentation function",
            choices=function_names
        )
        
        if not selected_function:
            return None, {}
        
        function_info = analysis_result[selected_function]
        user_parameters = {}
        
        # Configure parameters for selected function
        if function_info.get('parameters'):
            print(f"\n⚙️  Configuring parameters for {selected_function}:")
            
            for param_name, param_info in function_info['parameters'].items():
                param_type = param_info.get('type', 'str')
                default_val = param_info.get('default')
                
                if param_type == 'bool':
                    value = inquirer.confirm(
                        f"Set {param_name}",
                        default=bool(default_val) if default_val is not None else False
                    )
                else:
                    prompt_text = f"Enter {param_name}"
                    if default_val is not None:
                        prompt_text += f" (default: {default_val})"
                    
                    value_str = inquirer.text(prompt_text, default=str(default_val) if default_val is not None else "")
                    
                    # Convert to appropriate type
                    try:
                        if param_type == 'int':
                            value = int(value_str) if value_str else (default_val if default_val is not None else 0)
                        elif param_type == 'float':
                            value = float(value_str) if value_str else (default_val if default_val is not None else 0.0)
                        elif param_type == 'list':
                            if value_str:
                                # Try to parse as list
                                try:
                                    value = ast.literal_eval(value_str)
                                except:
                                    value = value_str.split(',') if ',' in value_str else [value_str]
                            else:
                                value = default_val if default_val is not None else []
                        else:
                            value = value_str if value_str else (default_val if default_val is not None else "")
                    except ValueError:
                        value = default_val if default_val is not None else ""
                        print(f"⚠️  Invalid value for {param_name}, using default: {value}")
                
                user_parameters[param_name] = value
        
        # Return function info with user parameters
        result_info = function_info.copy()
        result_info['user_parameters'] = user_parameters
        result_info['file_path'] = file_path
        
        return selected_function, result_info

    def configure_augmentation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive configuration of data augmentation settings.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Updated augmentation configuration
        """
        print("\n🔄 Data Augmentation Configuration")
        print("=" * 45)
        
        # Default augmentation configuration with common augmentations
        augmentation_config = {
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
                "variance_limit": 0.01,
                "probability": 0.2
            },
            "Brightness Adjustment": {
                "enabled": False,
                "brightness_limit": 0.2,
                "probability": 0.5
            },
            "Contrast Adjustment": {
                "enabled": False,
                "contrast_limit": 0.2,
                "probability": 0.5
            },
            "Color Jittering": {
                "enabled": False,
                "hue_shift_limit": 20,
                "sat_shift_limit": 30,
                "val_shift_limit": 20,
                "probability": 0.5
            },
            "Random Cropping": {
                "enabled": False,
                "crop_area_min": 0.08,
                "crop_area_max": 1.0,
                "aspect_ratio_min": 0.75,
                "aspect_ratio_max": 1.33,
                "probability": 1.0
            }
        }
        
        # 1. Configure Preset Augmentations
        print("\n🎯 Step 1: Preset Augmentation Selection")
        
        # Present augmentations as categories
        geometric_augs = ["Horizontal Flip", "Vertical Flip", "Rotation", "Random Cropping"]
        color_augs = ["Brightness Adjustment", "Contrast Adjustment", "Color Jittering"]
        noise_augs = ["Gaussian Noise"]
        
        print("\n📐 Geometric Augmentations:")
        for aug_name in geometric_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                # Configure specific parameters for each augmentation
                if aug_name == "Horizontal Flip" or aug_name == "Vertical Flip":
                    prob = inquirer.text(f"Probability for {aug_name}", default="0.5")
                    try:
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid probability, using default")
                        
                elif aug_name == "Rotation":
                    angle = inquirer.text("Maximum rotation angle (degrees)", default="15.0")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["angle_range"] = float(angle)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
                        
                elif aug_name == "Random Cropping":
                    crop_min = inquirer.text("Minimum crop area (0.01-1.0)", default="0.08")
                    crop_max = inquirer.text("Maximum crop area (0.01-1.0)", default="1.0")
                    aspect_min = inquirer.text("Minimum aspect ratio", default="0.75")
                    aspect_max = inquirer.text("Maximum aspect ratio", default="1.33")
                    try:
                        augmentation_config[aug_name]["crop_area_min"] = float(crop_min)
                        augmentation_config[aug_name]["crop_area_max"] = float(crop_max)
                        augmentation_config[aug_name]["aspect_ratio_min"] = float(aspect_min)
                        augmentation_config[aug_name]["aspect_ratio_max"] = float(aspect_max)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
        
        print("\n🎨 Color Augmentations:")
        for aug_name in color_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                if aug_name == "Brightness Adjustment":
                    limit = inquirer.text("Brightness change limit (±)", default="0.2")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["brightness_limit"] = float(limit)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
                        
                elif aug_name == "Contrast Adjustment":
                    limit = inquirer.text("Contrast change limit (±)", default="0.2")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["contrast_limit"] = float(limit)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
                        
                elif aug_name == "Color Jittering":
                    hue = inquirer.text("Hue shift limit", default="20")
                    sat = inquirer.text("Saturation shift limit", default="30")
                    val = inquirer.text("Value shift limit", default="20")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["hue_shift_limit"] = int(hue)
                        augmentation_config[aug_name]["sat_shift_limit"] = int(sat)
                        augmentation_config[aug_name]["val_shift_limit"] = int(val)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
        
        print("\n🔊 Noise Augmentations:")
        for aug_name in noise_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                if aug_name == "Gaussian Noise":
                    variance = inquirer.text("Noise variance limit", default="0.01")
                    prob = inquirer.text("Probability", default="0.2")
                    try:
                        augmentation_config[aug_name]["variance_limit"] = float(variance)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        print("⚠️  Invalid values, using defaults")
        
        # 2. Custom Augmentation Functions
        print("\n🛠️  Step 2: Custom Augmentation Functions")
        add_custom = inquirer.confirm("Add custom augmentation functions?", default=False)
        
        custom_augmentations = []
        
        if add_custom:
            custom_augmentation_path = inquirer.text(
                "Enter path to Python file containing custom augmentation functions",
                default="./example_funcs/example_custom_augmentations.py"
            )
            
            if custom_augmentation_path and os.path.exists(custom_augmentation_path):
                # Analyze and select custom augmentation functions
                success, augmentation_info = self.analyze_custom_augmentation_file(custom_augmentation_path)
                
                if success and augmentation_info:
                    print(f"\n✅ Found {len(augmentation_info)} augmentation function(s)")
                    
                    # Allow multiple selections
                    add_more = True
                    while add_more:
                        func_name, func_info = self.interactive_custom_augmentation_selection(custom_augmentation_path)
                        
                        if func_name and func_info:
                            # Always add probability parameter for custom augmentations
                            probability = inquirer.text("Set probability for this augmentation", default="0.5")
                            try:
                                prob_value = float(probability)
                            except ValueError:
                                prob_value = 0.5
                                print("⚠️  Invalid probability, using 0.5")
                            
                            custom_func_config = {
                                "enabled": True,
                                "function_name": func_name,
                                "file_path": custom_augmentation_path,
                                "probability": prob_value,
                                "parameters": func_info.get('user_parameters', {})
                            }
                            
                            # Add to augmentation config with descriptive name
                            display_name = f"{func_name} (custom)"
                            augmentation_config[display_name] = custom_func_config
                            print(f"✅ Added custom augmentation: {display_name}")
                        
                        add_more = inquirer.confirm("Add another custom augmentation function?", default=False)
                else:
                    print("❌ No valid augmentation functions found in the file")
            else:
                print("❌ Invalid file path or file does not exist")
        
        return augmentation_config

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
