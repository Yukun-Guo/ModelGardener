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
import json
import yaml
import sys
import copy
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

    def _is_model_function(self, obj, name: str) -> bool:
        """Check if an object is likely a model function."""
        try:
            if inspect.isfunction(obj):
                # Check if function signature includes typical model parameters
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                
                # Look for common model function patterns
                model_indicators = [
                    'input_shape', 'num_classes', 'classes', 'inputs', 'outputs',
                    'input_tensor', 'model', 'layers', 'activation'
                ]
                
                # Check if function has model-related parameters
                has_model_params = any(indicator in ' '.join(params).lower() for indicator in model_indicators)
                
                # Check function name patterns
                name_lower = name.lower()
                name_patterns = [
                    'create', 'build', 'get', 'make', 'model', 'net', 'network',
                    'cnn', 'resnet', 'efficientnet', 'mobilenet', 'unet'
                ]
                has_model_name = any(pattern in name_lower for pattern in name_patterns)
                
                return has_model_params or has_model_name
                
            elif inspect.isclass(obj):
                # Check if class looks like a model class
                name_lower = name.lower()
                class_patterns = ['model', 'net', 'network', 'cnn', 'classifier']
                return any(pattern in name_lower for pattern in class_patterns)
                
        except Exception:
            pass
        
        return False

    def _extract_model_parameters(self, obj) -> Dict[str, Any]:
        """Extract parameters from model function/class."""
        parameters = {}
        try:
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'input_shape', 'num_classes']:
                        param_info = {'type': 'str', 'default': None}
                        if param.default != inspect.Parameter.empty:
                            param_info['default'] = param.default
                            # Infer type from default value
                            if isinstance(param.default, bool):
                                param_info['type'] = 'bool'
                            elif isinstance(param.default, int):
                                param_info['type'] = 'int'
                            elif isinstance(param.default, float):
                                param_info['type'] = 'float'
                        parameters[param_name] = param_info
            elif inspect.isclass(obj):
                # Extract from __init__ method
                init_method = getattr(obj, '__init__', None)
                if init_method:
                    sig = inspect.signature(init_method)
                    for param_name, param in sig.parameters.items():
                        if param_name not in ['self', 'input_shape', 'num_classes']:
                            param_info = {'type': 'str', 'default': None}
                            if param.default != inspect.Parameter.empty:
                                param_info['default'] = param.default
                                if isinstance(param.default, bool):
                                    param_info['type'] = 'bool'
                                elif isinstance(param.default, int):
                                    param_info['type'] = 'int'
                                elif isinstance(param.default, float):
                                    param_info['type'] = 'float'
                            parameters[param_name] = param_info
        except Exception:
            pass
        
        return parameters

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
                if self._is_model_function(obj, name):
                    model_info = {
                        'name': name,
                        'type': 'function' if inspect.isfunction(obj) else 'class',
                        'file_path': file_path,
                        'parameters': self._extract_model_parameters(obj)
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

    def _is_data_loader_function(self, obj, name: str) -> bool:
        """
        Check if an object is a valid data loader function or class.
        
        Args:
            obj: The object to check
            name: Name of the object
            
        Returns:
            bool: True if it's a valid data loader function/class
        """
        import inspect
        
        # Skip private functions, imports, and common utilities
        if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'os', 'sys', 'train_test_split', 'pd', 'pandas']:
            return False
        
        # Skip objects from imported modules
        if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
            if not obj.__module__.startswith('custom') and 'custom' not in obj.__module__:
                return False
            
        try:
            if inspect.isfunction(obj):
                # Check function signature for data loader patterns
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                
                # Must have data-related parameters
                data_loader_indicators = [
                    'data_dir', 'batch_size', 'split', 'train_dir', 'val_dir',
                    'dataset', 'images', 'labels', 'data_path', 'file_path',
                    'csv_path', 'npz_path', 'tfrecord_path'
                ]
                
                # Check if function has data loader-like parameters
                has_data_params = any(indicator in param.lower() for param in params for indicator in data_loader_indicators)
                
                # Must return tf.data.Dataset or similar
                return_annotation = sig.return_annotation
                valid_return_type = False
                if return_annotation != inspect.Signature.empty:
                    return_type_str = str(return_annotation)
                    if 'tf.data.Dataset' in return_type_str or 'Dataset' in return_type_str or 'DatasetV2' in return_type_str:
                        valid_return_type = True
                
                # Check docstring for data loader keywords
                docstring = inspect.getdoc(obj) or ""
                docstring_lower = docstring.lower()
                data_loader_keywords = ['data loader', 'dataset', 'load data', 'data loading']
                has_data_keywords = any(keyword in docstring_lower for keyword in data_loader_keywords)
                
                # Exclude simple utility functions like invalid_data_function
                if len(params) < 2:
                    return False
                
                # Must have either data params + valid return type OR data keywords + data params
                return (has_data_params and valid_return_type) or (has_data_keywords and has_data_params)
                
            elif inspect.isclass(obj):
                # Check if class has data loader-like methods
                methods = [method for method in dir(obj) if not method.startswith('_')]
                data_loader_methods = ['load', 'get_dataset', 'load_data', 'get_data']
                has_loader_methods = any(method.lower() in [m.lower() for m in data_loader_methods] for method in methods)
                
                # Check class docstring
                docstring = inspect.getdoc(obj) or ""
                docstring_lower = docstring.lower()
                class_keywords = ['data loader', 'dataset', 'load data', 'dataloader']
                has_class_keywords = any(keyword in docstring_lower for keyword in class_keywords)
                
                # Check constructor parameters
                init_method = getattr(obj, '__init__', None)
                has_data_params = False
                if init_method:
                    try:
                        sig = inspect.signature(init_method)
                        params = list(sig.parameters.keys())
                        data_indicators = ['data_dir', 'batch_size', 'data_path', 'npz_path', 'csv_path']
                        has_data_params = any(indicator in param.lower() for param in params for indicator in data_indicators)
                    except:
                        pass
                
                return (has_loader_methods or has_class_keywords) and has_data_params
                
        except Exception:
            return False
            
        return False

    def _extract_data_loader_parameters(self, obj) -> Dict[str, Any]:
        """
        Extract parameters from a data loader function or class.
        
        Args:
            obj: The data loader function or class
            
        Returns:
            Dict containing parameter information
        """
        import inspect
        
        try:
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                params = {}
                
                for param_name, param in sig.parameters.items():
                    # Skip common fixed parameters
                    if param_name in ['data_dir', 'train_dir', 'val_dir', 'split']:
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    # Infer parameter type
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
                
                return {
                    'type': 'function',
                    'parameters': params,
                    'signature': str(sig),
                    'description': inspect.getdoc(obj) or f"Data loader function: {obj.__name__}"
                }
                
            elif inspect.isclass(obj):
                # Get constructor parameters
                init_method = getattr(obj, '__init__', None)
                params = {}
                
                if init_method:
                    sig = inspect.signature(init_method)
                    for param_name, param in sig.parameters.items():
                        if param_name == 'self':
                            continue
                        if param_name in ['data_dir', 'train_dir', 'val_dir']:
                            continue
                            
                        param_info = {
                            'name': param_name,
                            'required': param.default == inspect.Parameter.empty,
                            'default': param.default if param.default != inspect.Parameter.empty else None,
                            'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                        }
                        
                        if param.annotation != inspect.Parameter.empty:
                            param_info['type'] = str(param.annotation)
                        elif param.default is not None:
                            param_info['type'] = type(param.default).__name__
                        else:
                            param_info['type'] = 'Any'
                        
                        params[param_name] = param_info
                
                return {
                    'type': 'class',
                    'parameters': params,
                    'signature': f"class {obj.__name__}",
                    'description': inspect.getdoc(obj) or f"Data loader class: {obj.__name__}"
                }
                
        except Exception:
            return {
                'type': 'unknown',
                'parameters': {},
                'signature': '',
                'description': f"Data loader: {getattr(obj, '__name__', 'Unknown')}"
            }
        
        return {}

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
                if self._is_data_loader_function(obj, name):
                    info = self._extract_data_loader_parameters(obj)
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

    def _is_loss_function(self, obj, name: str) -> bool:
        """
        Check if an object is a valid loss function.
        
        Args:
            obj: The object to check
            name: Name of the object
            
        Returns:
            bool: True if it's a valid loss function
        """
        import inspect
        
        # Skip private functions, imports, and common utilities
        if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'keras', 'K']:
            return False
        
        # Skip objects from imported modules (except custom ones)
        if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
            if not obj.__module__.startswith('custom') and 'custom' not in obj.__module__:
                return False
            
        try:
            if inspect.isfunction(obj):
                # Check function signature for loss function patterns
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                
                # Must have typical loss function parameters
                loss_indicators = ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction', 'labels', 'logits']
                has_loss_params = len(params) >= 2 and any(indicator in param.lower() for param in params for indicator in loss_indicators)
                
                # Check docstring for loss function keywords
                docstring = inspect.getdoc(obj) or ""
                docstring_lower = docstring.lower()
                loss_keywords = ['loss', 'cost', 'error', 'distance', 'divergence']
                has_loss_keywords = any(keyword in docstring_lower for keyword in loss_keywords)
                
                return has_loss_params or has_loss_keywords
                
            elif inspect.isclass(obj):
                # Check if class inherits from typical loss classes or has loss-like methods
                methods = [method for method in dir(obj) if not method.startswith('_')]
                loss_methods = ['call', '__call__', 'compute_loss', 'calculate_loss']
                has_loss_methods = any(method.lower() in [m.lower() for m in loss_methods] for method in methods)
                
                # Check class docstring
                docstring = inspect.getdoc(obj) or ""
                docstring_lower = docstring.lower()
                class_keywords = ['loss', 'cost function', 'objective function']
                has_class_keywords = any(keyword in docstring_lower for keyword in class_keywords)
                
                return has_loss_methods or has_class_keywords
                
        except Exception:
            return False
            
        return False

    def _extract_loss_parameters(self, obj) -> Dict[str, Any]:
        """
        Extract parameters from a loss function.
        
        Args:
            obj: The loss function or class
            
        Returns:
            Dict containing parameter information
        """
        import inspect
        
        try:
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                params = {}
                
                for param_name, param in sig.parameters.items():
                    # Skip y_true, y_pred parameters as they are provided during training
                    if param_name.lower() in ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction']:
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    # Infer parameter type
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
                
                return {
                    'type': 'function',
                    'parameters': params,
                    'signature': str(sig),
                    'description': inspect.getdoc(obj) or f"Loss function: {obj.__name__}"
                }
                
            elif inspect.isclass(obj):
                # Get constructor parameters
                init_method = getattr(obj, '__init__', None)
                params = {}
                
                if init_method:
                    sig = inspect.signature(init_method)
                    for param_name, param in sig.parameters.items():
                        if param_name == 'self':
                            continue
                            
                        param_info = {
                            'name': param_name,
                            'required': param.default == inspect.Parameter.empty,
                            'default': param.default if param.default != inspect.Parameter.empty else None,
                            'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                        }
                        
                        if param.annotation != inspect.Parameter.empty:
                            param_info['type'] = str(param.annotation)
                        elif param.default is not None:
                            param_info['type'] = type(param.default).__name__
                        else:
                            param_info['type'] = 'Any'
                        
                        params[param_name] = param_info
                
                return {
                    'type': 'class',
                    'parameters': params,
                    'signature': f"class {obj.__name__}",
                    'description': inspect.getdoc(obj) or f"Loss class: {obj.__name__}"
                }
                
        except Exception:
            return {
                'type': 'unknown',
                'parameters': {},
                'signature': '',
                'description': f"Loss function: {getattr(obj, '__name__', 'Unknown')}"
            }
        
        return {}

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
                if self._is_loss_function(obj, name):
                    info = self._extract_loss_parameters(obj)
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
                        num_outputs, output_names = self._analyze_custom_model_outputs(
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

    def _analyze_custom_model_outputs(self, file_path: str, function_name: str, 
                                    model_config: Dict[str, Any]) -> Tuple[int, List[str]]:
        """
        Analyze a custom model function to determine its outputs.
        
        Args:
            file_path: Path to the Python file containing the model
            function_name: Name of the model function
            model_config: Model configuration parameters
            
        Returns:
            Tuple of (num_outputs, output_names)
        """
        import importlib.util
        import inspect
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("custom_model", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from: {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the model function
        if not hasattr(module, function_name):
            raise AttributeError(f"Function {function_name} not found in {file_path}")
        
        model_func = getattr(module, function_name)
        
        # Try to build the model to analyze its structure
        try:
            # Complete suppression of TensorFlow warnings during model building
            import sys
            import contextlib
            from io import StringIO
            
            # Create context manager to suppress all output
            @contextlib.contextmanager
            def suppress_output():
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
            # Prepare model parameters
            model_params = model_config.get('model_parameters', {})
            input_shape = (
                model_params.get('input_shape', {}).get('height', 224),
                model_params.get('input_shape', {}).get('width', 224), 
                model_params.get('input_shape', {}).get('channels', 3)
            )
            num_classes = model_params.get('classes', 10)
            
            # Build the model with complete output suppression
            with suppress_output():
                import keras
                
                if inspect.isclass(model_func):
                    # If it's a class, instantiate it
                    model = model_func(input_shape=input_shape, num_classes=num_classes)
                else:
                    # If it's a function, call it
                    model = model_func(input_shape=input_shape, num_classes=num_classes)
            
            if hasattr(model, 'outputs') and hasattr(model.outputs, '__len__'):
                num_outputs = len(model.outputs)
                output_names = []
                
                for i, output in enumerate(model.outputs):
                    output_name = None
                    
                    if hasattr(output, 'name') and output.name:
                        # Extract clean name from tensor name (remove :0 suffix and path)
                        clean_name = output.name.split(':')[0].split('/')[-1]
                        
                        # Check if it's a meaningful name (not generic tensor names)
                        if clean_name and not any(generic in clean_name.lower() for generic in 
                                                ['keras_tensor', 'dense_', 'sequential_', 'functional_']):
                            output_name = clean_name
                        
                        # Special case: look for aux/auxiliary patterns
                        if 'aux' in clean_name.lower() or 'auxiliary' in clean_name.lower():
                            output_name = clean_name
                    
                    # If no meaningful name found, generate a sensible default
                    if not output_name:
                        if i == 0:
                            output_name = 'main_output'
                        else:
                            output_name = f'aux_output_{i}' if i == 1 else f'output_{i+1}'
                    
                    output_names.append(output_name)
                
                return num_outputs, output_names
            else:
                return 1, ['main_output']
                
        except Exception:
            # Fall back to source code analysis
            return self._analyze_model_source_code(model_func)

    def _analyze_model_source_code(self, model_func) -> Tuple[int, List[str]]:
        """
        Analyze model function source code to detect multiple outputs.
        
        Args:
            model_func: The model function to analyze
            
        Returns:
            Tuple of (num_outputs, output_names)
        """
        import inspect
        
        try:
            source = inspect.getsource(model_func)
            source_lower = source.lower()
            
            # Look for multiple outputs patterns
            multiple_output_patterns = [
                'model(inputs, [',  # keras.Model(inputs, [output1, output2])
                'return [',         # return [output1, output2]
                ', name=',         # multiple named outputs
                'outputs = [',     # outputs = [...]
                'aux_output',      # auxiliary outputs
            ]
            
            pattern_count = sum(1 for pattern in multiple_output_patterns if pattern in source_lower)
            
            if pattern_count >= 2 or 'aux_output' in source_lower:
                # Likely multiple outputs - try to extract names
                output_names = []
                
                # Look for name= patterns in layer definitions
                import re
                name_patterns = re.findall(r'name=[\'"]([^\'\"]+)[\'"]', source)
                for name in name_patterns:
                    if any(keyword in name.lower() for keyword in ['output', 'aux', 'auxiliary']):
                        output_names.append(name)
                
                # Look for variable names that suggest outputs
                variable_patterns = re.findall(r'(\w*(?:output|aux)\w*)\s*=', source_lower)
                for var_name in variable_patterns:
                    if var_name and var_name not in output_names:
                        output_names.append(var_name)
                
                # Clean up and validate output names
                clean_names = []
                for name in output_names:
                    if name and len(name) > 0:
                        clean_names.append(name)
                
                if not clean_names:
                    clean_names = ['main_output', 'aux_output']
                elif len(clean_names) == 1:
                    clean_names = ['main_output', clean_names[0]]
                
                # Limit to reasonable number of outputs
                if len(clean_names) > 5:
                    clean_names = clean_names[:5]
                
                return len(clean_names), clean_names
            
            return 1, ['main_output']
            
        except Exception:
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
        
        # Dynamically discover custom functions from example_funcs directory
        augmentation_functions = self._discover_custom_functions('./example_funcs/example_custom_augmentations.py')
        preprocessing_functions = self._discover_custom_functions('./example_funcs/example_custom_preprocessing.py')
        
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
        Copy CIFAR-10 NPZ dataset to the project directory.
        
        Args:
            project_dir: Target project directory
        """
        import shutil
        
        # Define source and destination paths
        source_data_dir = os.path.join(os.path.dirname(__file__), 'example_data')
        dest_data_dir = os.path.join(project_dir, 'data')
        cifar10_source = os.path.join(source_data_dir, 'cifar10.npz')
        cifar10_dest = os.path.join(dest_data_dir, 'cifar10.npz')
        
        try:
            # Create data directory
            os.makedirs(dest_data_dir, exist_ok=True)
            
            # Copy CIFAR-10 NPZ file
            if os.path.exists(cifar10_source):
                shutil.copy2(cifar10_source, cifar10_dest)
                print(f"✅ CIFAR-10 dataset copied to: {dest_data_dir}")
                
                # Load and show dataset info
                try:
                    import numpy as np
                    with np.load(cifar10_source) as data:
                        x_data = data['x']
                        y_data = data['y']
                        print(f"📊 CIFAR-10 dataset: {len(x_data)} samples, {x_data.shape[1:]} shape, {len(np.unique(y_data))} classes")
                except Exception as e:
                    print(f"📊 CIFAR-10 dataset copied (could not read metadata: {e})")
                    
            else:
                print(f"⚠️ Warning: CIFAR-10 dataset not found at {cifar10_source}")
                print("� Please run test_generate_subset.py to generate the CIFAR-10 dataset")
                
        except Exception as e:
            print(f"❌ Error copying CIFAR-10 data: {str(e)}")
            print("💡 Please ensure CIFAR-10 dataset is available in example_data/cifar10.npz")

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
        
        # Add all available custom augmentation functions
        if 'data' in improved_config['configuration'] and 'augmentation' in improved_config['configuration']['data']:
            augmentation_functions = self._discover_custom_functions('./example_funcs/example_custom_augmentations.py')
            for func_name, func_info in augmentation_functions.items():
                display_name = f"{func_name.replace('_', ' ').title()} (custom)"
                augmentation_config = {
                    'enabled': False,
                    'function_name': func_name, 
                    'file_path': './custom_modules/custom_augmentations.py'
                }
                # Add function-specific parameters
                augmentation_config.update(func_info.get('parameters', {}))
                improved_config['configuration']['data']['augmentation'][display_name] = augmentation_config
        
        # Add all available custom preprocessing functions
        if 'data' in improved_config['configuration'] and 'preprocessing' in improved_config['configuration']['data']:
            preprocessing_functions = self._discover_custom_functions('./example_funcs/example_custom_preprocessing.py')
            for func_name, func_info in preprocessing_functions.items():
                display_name = f"{func_name.replace('_', ' ').title()} (custom)"
                preprocessing_config = {
                    'enabled': False,
                    'function_name': func_name,
                    'file_path': './custom_modules/custom_preprocessing.py'
                }
                # Add function-specific parameters
                preprocessing_config.update(func_info.get('parameters', {}))
                improved_config['configuration']['data']['preprocessing'][display_name] = preprocessing_config
        
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
                    'name': 'Custom_load_cifar10_npz_data',
                    'file_path': './custom_modules/custom_data_loaders.py',
                    'function_name': 'Custom_load_cifar10_npz_data', 
                    'type': 'function',
                    'parameters': self._extract_function_parameters('Custom_load_cifar10_npz_data', './custom_modules/custom_data_loaders.py', project_dir)
                }],
                'loss_functions': [{
                    'name': 'dice_loss',
                    'file_path': './custom_modules/custom_loss_functions.py',
                    'function_name': 'dice_loss',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('dice_loss', './custom_modules/custom_loss_functions.py', project_dir)
                }],
                'optimizers': [{
                    'name': 'adaptive_adam',
                    'file_path': './custom_modules/custom_optimizers.py',
                    'function_name': 'adaptive_adam',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('adaptive_adam', './custom_modules/custom_optimizers.py', project_dir)
                }],
                'metrics': [{
                    'name': 'balanced_accuracy',
                    'file_path': './custom_modules/custom_metrics.py',
                    'function_name': 'balanced_accuracy',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('balanced_accuracy', './custom_modules/custom_metrics.py', project_dir)
                }],
                'callbacks': [{
                    'name': 'MemoryUsageMonitor',
                    'file_path': './custom_modules/custom_callbacks.py',
                    'function_name': 'MemoryUsageMonitor',
                    'type': 'class',
                    'parameters': self._extract_function_parameters('MemoryUsageMonitor', './custom_modules/custom_callbacks.py', project_dir)
                }],
                'augmentations': [{
                    'name': 'color_shift',
                    'file_path': './custom_modules/custom_augmentations.py',
                    'function_name': 'color_shift',
                    'type': 'function',
                    'parameters': self._extract_function_parameters('color_shift', './custom_modules/custom_augmentations.py', project_dir)
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
        yaml_lines.append(f"    train_dir: {data_config.get('train_dir', './data')}")
        yaml_lines.append(f"    val_dir: {data_config.get('val_dir', './data')}")
        
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
        
        # Preprocessing section with custom options
        preprocessing = data_config.get('preprocessing', {})
        yaml_lines.append("    preprocessing:")
        
        # Standard preprocessing options (non-custom)
        for key, value in preprocessing.items():
            if not key.endswith('(custom)'):
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
        
        # Add custom preprocessing functions
        custom_preprocessing_found = False
        for key, value in preprocessing.items():
            if key.endswith('(custom)'):
                if not custom_preprocessing_found:
                    yaml_lines.append("      # Custom preprocessing functions (disabled by default)")
                    custom_preprocessing_found = True
                yaml_lines.append(f"      {key}:")
                for sub_key, sub_value in value.items():
                    yaml_lines.append(f"        {sub_key}: {sub_value}")
                
        # Augmentation section with custom options
        augmentation = data_config.get('augmentation', {})
        yaml_lines.append("    augmentation:")
        yaml_lines.append("      # Built-in augmentation options")
        
        # Standard augmentation options (non-custom)
        for key, value in augmentation.items():
            if not key.endswith('(custom)'):
                yaml_lines.append(f"      {key}:")
                self._add_nested_yaml(yaml_lines, value, 8)
        
        # Add custom augmentation functions
        custom_augmentation_found = False
        for key, value in augmentation.items():
            if key.endswith('(custom)'):
                if not custom_augmentation_found:
                    yaml_lines.append("      # Custom augmentation functions (disabled by default)")
                    custom_augmentation_found = True
                yaml_lines.append(f"      {key}:")
                for sub_key, sub_value in value.items():
                    yaml_lines.append(f"        {sub_key}: {sub_value}")
                
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

    def _discover_custom_functions(self, file_path: str) -> Dict[str, Any]:
        """
        Dynamically discover and analyze custom functions from a Python file.
        
        Args:
            file_path: Path to the Python file containing custom functions
            
        Returns:
            Dictionary mapping function names to their information
        """
        import ast
        import inspect
        
        custom_functions = {}
        
        if not os.path.exists(file_path):
            return custom_functions
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Skip private functions
                    if func_name.startswith('_'):
                        continue
                    
                    # Extract function parameters and their defaults
                    params = {}
                    
                    # Get function arguments
                    args = node.args.args
                    defaults = node.args.defaults or []
                    
                    # Skip the first parameter (data/image)
                    if args:
                        param_args = args[1:]  # Skip first parameter
                        
                        # Match defaults with parameters (from right to left)
                        num_defaults = len(defaults)
                        num_params = len(param_args)
                        
                        for i, arg in enumerate(param_args):
                            param_name = arg.arg
                            
                            # Determine if this parameter has a default value
                            default_index = i - (num_params - num_defaults)
                            if default_index >= 0:
                                default_node = defaults[default_index]
                                if isinstance(default_node, ast.Constant):
                                    default_value = default_node.value
                                elif isinstance(default_node, ast.Num):  # Python < 3.8 compatibility
                                    default_value = default_node.n
                                elif isinstance(default_node, ast.Str):  # Python < 3.8 compatibility
                                    default_value = default_node.s
                                else:
                                    default_value = 0.5  # Fallback default
                                
                                params[param_name] = default_value
                    
                    # Extract docstring if available
                    docstring = ast.get_docstring(node) or f"Custom function: {func_name}"
                    
                    custom_functions[func_name] = {
                        'parameters': params,
                        'docstring': docstring,
                        'file_path': file_path,
                        'function_name': func_name
                    }
                    
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
        
        return custom_functions

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
