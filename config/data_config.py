"""
Data configuration module for ModelGardener CLI.
"""

import inspect
import importlib.util
import os
from typing import Dict, Any, List, Tuple, Optional
from .base_config import BaseConfig


class DataConfig(BaseConfig):
    """Data loader configuration handler."""
    
    def __init__(self):
        super().__init__()
        self.available_data_loaders = [
            'ImageDataGenerator', 'DirectoryDataLoader', 'TFRecordDataLoader', 'CSVDataLoader',
            'NPZDataLoader', 'Custom'
        ]

    def _is_data_loader_function(self, obj, name: str) -> bool:
        """
        Check if an object is a valid data loader function or class.
        
        Args:
            obj: The object to check
            name: Name of the object
            
        Returns:
            bool: True if it's a valid data loader function/class
        """
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
                
                # Exclude simple utility functions
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
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            module = self.load_custom_module(file_path, "custom_data_loaders")
            if module is None:
                return False, {}
            
            data_loader_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if self._is_data_loader_function(obj, name):
                    info = self._extract_data_loader_parameters(obj)
                    if info:
                        data_loader_info[name] = info
            
            return len(data_loader_info) > 0, data_loader_info
            
        except Exception as e:
            self.print_error(f"Error analyzing data loader file: {str(e)}")
            return False, {}

    def interactive_custom_data_loader_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom data loader from analyzed file.
        
        Args:
            file_path: Path to the custom data loader file
            
        Returns:
            Tuple of (selected_loader_name, loader_info)
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return None, {}
            
        success, analysis_result = self.analyze_custom_data_loader_file(file_path)
        
        if not success or not analysis_result:
            self.print_error("No valid data loader functions found in the file")
            return None, {}
        
        print(f"\n✅ Found {len(analysis_result)} data loader function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function':
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
        
        # Extract the name from the choice
        selected_name = selected_choice.split(' ')[0] if ' ' in selected_choice else selected_choice
        
        if selected_name in analysis_result:
            info = analysis_result[selected_name]
            self.print_success(f"Selected custom data loader: {selected_name}")
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
