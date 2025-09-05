"""
Data Loader Group Parameter for ModelGardener

This module provides a custom parameter group for configuring data loaders,
allowing users to load custom data loader functions from Python files for training and validation data.
"""

import os
import importlib.util
import ast
from typing import Dict, Any, List, Optional

# CLI-only message functions
def cli_info(title, message):
    print(f"[INFO] {title}: {message}")

def cli_warning(title, message):
    print(f"[WARNING] {title}: {message}")

def cli_error(title, message):
    print(f"[ERROR] {title}: {message}")

def cli_get_file_path(title="Select File", file_filter="Python Files (*.py)"):
    print(f"[CLI] File dialog requested: {title} - {file_filter}")
    print("[CLI] File dialogs not supported in CLI mode. Use config files to specify custom functions.")
    return "", ""


class DataLoaderGroup:
    """Custom data loader group that allows loading custom data loader functions from files."""
    
    def __init__(self, **opts):
        self.config = {
            'Data Loader Selection': {
                'selected_data_loader': 'Default',
                'use_for_train': True,
                'use_for_val': True,
                'batch_size': 32,
                'shuffle': True,
                'validation_split': 0.2
            },
            'Data Loading': {
                'data_source': 'files',
                'preprocessing': True,
                'augmentation': False
            },
            'Custom Parameters': {}
        }
        
        # Initialize custom data loader storage
        self._custom_data_loaders = {}
        self._custom_data_loader_parameters = {}
        
    def get_config(self):
        """Get the current data loader configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the data loader configuration."""
        self.config.update(config)
        
    def get_value(self, path):
        """Get a configuration value by path."""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def set_value(self, path, value):
        """Set a configuration value by path."""
        keys = path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        
        # Update dependent configurations if needed
        if path == 'Data Loader Selection.selected_data_loader':
            self._update_data_loader_parameters()
    
    def _get_data_loader_options(self):
        """Get list of available data loader names including custom ones."""
        base_options = [
            'Default',
            'TFRecordDataLoader',
            'ImageDataLoader',
            'CSVDataLoader',
            'NumpyDataLoader',
            'HDF5DataLoader',
            'JSONDataLoader'
        ]
        
        # Add custom data loaders if any
        if hasattr(self, '_custom_data_loaders'):
            custom_options = list(self._custom_data_loaders.keys())
            return base_options + custom_options
        
        return base_options
    
    def _update_data_loader_parameters(self):
        """Update data loader parameters based on selected data loader."""
        selected = self.config['Data Loader Selection']['selected_data_loader']
        
        # Set default parameters based on data loader type
        if selected == 'TFRecordDataLoader':
            self.config['Data Loading'].update({
                'file_pattern': '*.tfrecord',
                'compression_type': None,
                'buffer_size': 8 * 1024 * 1024
            })
        elif selected == 'ImageDataLoader':
            self.config['Data Loading'].update({
                'image_size': [224, 224],
                'color_mode': 'rgb',
                'image_format': 'jpeg'
            })
        elif selected == 'CSVDataLoader':
            self.config['Data Loading'].update({
                'separator': ',',
                'header': True,
                'target_column': 'label'
            })
        elif selected == 'NumpyDataLoader':
            self.config['Data Loading'].update({
                'file_extension': '.npy',
                'mmap_mode': None
            })
        elif selected == 'HDF5DataLoader':
            self.config['Data Loading'].update({
                'dataset_key': 'data',
                'label_key': 'labels'
            })
        elif selected == 'JSONDataLoader':
            self.config['Data Loading'].update({
                'json_format': 'records',
                'orient': 'index'
            })
    
    def load_custom_data_loaders(self, file_path):
        """Load custom data loader functions from a file."""
        try:
            custom_functions = self._extract_data_loader_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid data loader functions found in the selected file.\n\n"
                    "Functions should return data that can be used for training."
                )
                return False
            
            # Add custom functions to the available data loader options
            for func_name, func_info in custom_functions.items():
                self._add_custom_data_loader_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom data loader(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese data loaders are now available in the selection dropdown."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom data loaders from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_data_loader_option(self, func_name, func_info):
        """Add a custom data loader as an option."""
        # Store the function with metadata
        self._custom_data_loaders[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_data_loader_parameters[func_name] = func_info['parameters']
    
    def _extract_data_loader_functions(self, file_path):
        """Extract data loader function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely a data loader function
                    if self._is_data_loader_function(node):
                        func_info = self._analyze_data_loader_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's a data loader class
                    if self._is_data_loader_class(node):
                        class_info = self._analyze_data_loader_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_data_loader_function(self, node):
        """Check if a function node is likely a data loader function."""
        # Check function name for data loader indicators
        loader_keywords = ['load', 'data', 'batch', 'dataset', 'generator', 'reader']
        has_loader_name = any(keyword in node.name.lower() for keyword in loader_keywords)
        
        # Check if function has common data loader parameters
        args = [arg.arg for arg in node.args.args]
        loader_args = ['path', 'file_path', 'data_path', 'batch_size', 'shuffle']
        has_loader_args = any(arg in args for arg in loader_args)
        
        return has_loader_name or has_loader_args
    
    def _is_data_loader_class(self, node):
        """Check if a class node is likely a data loader class."""
        # Check class name for data loader indicators
        loader_keywords = ['loader', 'data', 'dataset', 'generator', 'reader']
        has_loader_name = any(keyword in node.name.lower() for keyword in loader_keywords)
        
        # Check if class has __iter__ or __getitem__ methods
        has_iteration = any(isinstance(child, ast.FunctionDef) and child.name in ['__iter__', '__getitem__', '__next__'] 
                          for child in node.body)
        
        return has_loader_name or has_iteration
    
    def _analyze_data_loader_function(self, node, file_path):
        """Analyze a data loader function and extract its metadata."""
        try:
            # Extract parameters
            parameters = []
            defaults = node.args.defaults
            default_values = [None] * (len(node.args.args) - len(defaults)) + defaults
            
            for i, arg in enumerate(node.args.args):
                param_info = {
                    'name': arg.arg,
                    'type': self._infer_parameter_type(arg.arg),
                    'default': self._extract_default_value(default_values[i])
                }
                parameters.append(param_info)
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Custom data loader function: {node.name}"
            
            return {
                'type': 'function',
                'file_path': file_path,
                'function_name': node.name,
                'parameters': parameters,
                'docstring': docstring,
                'source_lines': (node.lineno, node.end_lineno) if hasattr(node, 'end_lineno') else (node.lineno, node.lineno)
            }
            
        except Exception as e:
            cli_warning("Analysis Error", f"Error analyzing function {node.name}: {str(e)}")
            return None
    
    def _analyze_data_loader_class(self, node, file_path):
        """Analyze a data loader class and extract its metadata."""
        try:
            # Find __init__ method to extract parameters
            init_method = None
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    init_method = child
                    break
            
            parameters = []
            if init_method:
                defaults = init_method.args.defaults
                default_values = [None] * (len(init_method.args.args) - len(defaults)) + defaults
                
                for i, arg in enumerate(init_method.args.args):
                    if arg.arg not in ['self']:
                        param_info = {
                            'name': arg.arg,
                            'type': self._infer_parameter_type(arg.arg),
                            'default': self._extract_default_value(default_values[i])
                        }
                        parameters.append(param_info)
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Custom data loader class: {node.name}"
            
            return {
                'type': 'class',
                'file_path': file_path,
                'class_name': node.name,
                'parameters': parameters,
                'docstring': docstring,
                'source_lines': (node.lineno, node.end_lineno) if hasattr(node, 'end_lineno') else (node.lineno, node.lineno)
            }
            
        except Exception as e:
            cli_warning("Analysis Error", f"Error analyzing class {node.name}: {str(e)}")
            return None
    
    def _infer_parameter_type(self, param_name):
        """Infer the type of a parameter based on its name."""
        param_name_lower = param_name.lower()
        
        if any(word in param_name_lower for word in ['size', 'batch', 'num', 'count', 'length']):
            return 'int'
        elif any(word in param_name_lower for word in ['rate', 'ratio', 'split', 'factor']):
            return 'float'
        elif any(word in param_name_lower for word in ['shuffle', 'use', 'enable', 'disable']):
            return 'bool'
        elif any(word in param_name_lower for word in ['path', 'file', 'dir', 'url']):
            return 'str'
        else:
            return 'str'  # Default to string
    
    def _extract_default_value(self, default_node):
        """Extract default value from an AST node."""
        if default_node is None:
            return None
        elif isinstance(default_node, ast.Constant):
            return default_node.value
        elif isinstance(default_node, ast.Num):  # For older Python versions
            return default_node.n
        elif isinstance(default_node, ast.Str):  # For older Python versions
            return default_node.s
        elif isinstance(default_node, ast.NameConstant):  # For older Python versions
            return default_node.value
        else:
            return None
    
    def get_data_loader_config(self):
        """Get the current data loader configuration for training."""
        config = {}
        
        selection = self.config['Data Loader Selection']
        data_loading = self.config['Data Loading']
        
        config['data_loader'] = selection['selected_data_loader']
        config['batch_size'] = selection['batch_size']
        config['shuffle'] = selection['shuffle']
        config['validation_split'] = selection['validation_split']
        config['use_for_train'] = selection['use_for_train']
        config['use_for_val'] = selection['use_for_val']
        
        # Add data loading specific parameters
        config.update(data_loading)
        
        # Add custom parameters if using custom data loader
        if selection['selected_data_loader'] in self._custom_data_loaders:
            custom_info = self._custom_data_loaders[selection['selected_data_loader']]
            config['custom_data_loader'] = custom_info
        
        return config
    
    def load_custom_data_loader_from_metadata(self, metadata):
        """Load a custom data loader from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_data_loader", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_data_loaders[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom data loader: {str(e)}")
            return False
