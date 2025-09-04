"""
Base configuration module with common utilities and base classes.
"""

import os
import sys
import json
import yaml
import importlib.util
import inspect
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path


class BaseConfig:
    """Base configuration class with common utilities."""
    
    def __init__(self):
        """Initialize base configuration."""
        pass
    
    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            return {}

    def save_config(self, config: Dict[str, Any], file_path: str, format_type: str = 'yaml') -> bool:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary
            file_path: Output file path
            format_type: Format ('json' or 'yaml')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                if format_type == 'yaml':
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False

    def load_custom_module(self, file_path: str, module_name: str = None):
        """
        Load a custom Python module from file path.
        
        Args:
            file_path: Path to Python file
            module_name: Optional module name
            
        Returns:
            Loaded module or None if failed
        """
        try:
            if not os.path.exists(file_path):
                return None
                
            if module_name is None:
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"❌ Error loading module {file_path}: {e}")
            return None

    def extract_function_parameters(self, obj) -> Dict[str, Any]:
        """
        Extract parameters from a function signature.
        
        Args:
            obj: Function object
            
        Returns:
            Dictionary of parameter information
        """
        try:
            sig = inspect.signature(obj)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': None if param.default == inspect.Parameter.empty else param.default,
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
                }
                parameters[param_name] = param_info
                
            return parameters
        except Exception:
            return {}

    def analyze_function_wrapper(self, obj, name: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze if a function follows the wrapper pattern and extract parameters.
        
        Args:
            obj: Function object
            name: Function name
            
        Returns:
            Tuple of (is_wrapper, parameters)
        """
        try:
            # Check if it's callable
            if not callable(obj):
                return False, {}
            
            # Get the outer function signature (wrapper parameters)
            sig = inspect.signature(obj)
            wrapper_params = {}
            
            for param_name, param in sig.parameters.items():
                wrapper_params[param_name] = {
                    'required': param.default == inspect.Parameter.empty,
                    'default': None if param.default == inspect.Parameter.empty else param.default
                }
            
            # Check if the function has a docstring indicating it's a wrapper
            has_wrapper_doc = False
            if obj.__doc__:
                doc_lower = obj.__doc__.lower()
                wrapper_keywords = ['wrapper', 'configuration', 'parameters', 'apply', 'data', 'label']
                has_wrapper_doc = any(keyword in doc_lower for keyword in wrapper_keywords)
            
            # Consider it a wrapper if it has parameters and appropriate naming/documentation
            is_wrapper = len(wrapper_params) > 0 and (
                name.startswith(('tf_', 'cv_', 'custom_')) or has_wrapper_doc
            )
            
            return is_wrapper, wrapper_params
            
        except Exception:
            return False, {}

    def validate_file_path(self, file_path: str, extensions: List[str] = None) -> bool:
        """
        Validate if a file path exists and has correct extension.
        
        Args:
            file_path: Path to validate
            extensions: List of allowed extensions (e.g., ['.py', '.yaml'])
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False
            
        if extensions:
            file_ext = os.path.splitext(file_path)[1].lower()
            return file_ext in extensions
            
        return True

    def get_relative_path(self, file_path: str, base_path: str = None) -> str:
        """
        Get relative path from base path.
        
        Args:
            file_path: Full file path
            base_path: Base path (defaults to current working directory)
            
        Returns:
            Relative path
        """
        if base_path is None:
            base_path = os.getcwd()
            
        try:
            return os.path.relpath(file_path, base_path)
        except Exception:
            return file_path

    def create_default_structure(self) -> Dict[str, Any]:
        """
        Create a default configuration structure.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'configuration': {
                'task_type': 'image_classification',
                'data': {
                    'train_dir': './data/train',
                    'val_dir': './data/val',
                    'data_loader': {
                        'selected_data_loader': 'ImageDataGenerator',
                        'parameters': {
                            'batch_size': 32,
                            'shuffle': True
                        }
                    }
                },
                'model': {
                    'model_family': 'resnet',
                    'model_name': 'ResNet50',
                    'model_parameters': {
                        'weights': 'imagenet',
                        'include_top': False
                    }
                },
                'training': {
                    'optimizer': {
                        'name': 'Adam',
                        'learning_rate': 0.001
                    },
                    'loss_function': {
                        'name': 'categorical_crossentropy'
                    },
                    'metrics': ['accuracy'],
                    'epochs': 100,
                    'callbacks': []
                },
                'runtime': {
                    'model_dir': './logs',
                    'use_gpu': True,
                    'num_gpus': 1
                }
            }
        }

    def print_success(self, message: str):
        """Print success message with green checkmark."""
        print(f"✅ {message}")

    def print_error(self, message: str):
        """Print error message with red X."""
        print(f"❌ {message}")

    def print_warning(self, message: str):
        """Print warning message with yellow triangle."""
        print(f"⚠️  {message}")

    def print_info(self, message: str):
        """Print info message with blue circle."""
        print(f"ℹ️  {message}")
