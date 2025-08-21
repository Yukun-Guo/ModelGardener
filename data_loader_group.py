"""
Data Loader Group Parameter for ModelGardener

This module provides a custom parameter group for configuring data loaders,
allowing users to load custom data loader functions from Python files for training and validation data.
"""

import os
import importlib.util
import ast
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph.parametertree.parameterTypes as pTypes


class DataLoaderGroup(pTypes.GroupParameter):
    """Custom data loader group that allows loading custom data loader functions from files."""
    
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Initialize custom data loader storage
        self._custom_data_loaders = {}
        self._custom_data_loader_parameters = {}
        
        # Add data loader selection
        self._add_data_loader_selection()
        
        # Add custom data loader button
        self._add_custom_button()
    
    def _add_data_loader_selection(self):
        """Add data loader selection with preset and custom options."""
        data_loader_options = self._get_data_loader_options()
        
        self.addChild({
            'name': 'Data Loader Selection',
            'type': 'group',
            'children': [
                {'name': 'selected_data_loader', 'type': 'list', 'limits': data_loader_options, 'value': 'Default', 
                 'tip': 'Select the data loader to use for training and validation'},
                {'name': 'use_for_train', 'type': 'bool', 'value': True, 
                 'tip': 'Use this data loader for training data'},
                {'name': 'use_for_val', 'type': 'bool', 'value': True, 
                 'tip': 'Use this data loader for validation data'}
            ],
            'tip': 'Choose data loader type and configure its parameters'
        })
        
        # Add data loader parameters
        self._add_selected_data_loader_parameters()
    
    def _get_data_loader_options(self):
        """Get list of available data loader names including custom ones."""
        base_options = [
            'Default',
            'TFRecordDataLoader',
            'ImageDataLoader',
            'CSVDataLoader',
            'HDF5DataLoader'
        ]
        
        # Add custom data loaders if any
        if hasattr(self, '_custom_data_loaders') and self._custom_data_loaders:
            custom_options = list(self._custom_data_loaders.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_selected_data_loader_parameters(self):
        """Add parameters for the selected data loader."""
        parent = self.child('Data Loader Selection')
        if not parent:
            return
            
        # Connect selection change to parameter update
        if parent.child('selected_data_loader'):
            parent.child('selected_data_loader').sigValueChanged.connect(
                self._update_data_loader_parameters
            )
        
        # Add initial parameters
        self._update_data_loader_parameters()
    
    def _update_data_loader_parameters(self):
        """Update data loader parameters based on selection."""
        parent = self.child('Data Loader Selection')
        if not parent:
            return
            
        selected_data_loader = parent.child('selected_data_loader').value()
        
        # Remove existing parameters (except selection controls)
        existing_params = []
        for child in parent.children():
            if child.name() not in ['selected_data_loader', 'use_for_train', 'use_for_val']:
                existing_params.append(child)
        
        for param in existing_params:
            parent.removeChild(param)
        
        # Add parameters based on selected data loader
        data_loader_params = self._get_data_loader_parameters(selected_data_loader)
        for param_config in data_loader_params:
            parent.addChild(param_config)
    
    def _get_data_loader_parameters(self, data_loader_name):
        """Get parameters for a specific data loader."""
        # Check if it's a custom data loader
        if hasattr(self, '_custom_data_loader_parameters') and data_loader_name in self._custom_data_loader_parameters:
            return self._custom_data_loader_parameters[data_loader_name]
        
        # Return built-in data loader parameters
        data_loader_parameters = {
            'Default': [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for data loading'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle the data'},
                {'name': 'buffer_size', 'type': 'int', 'value': 10000, 'limits': (1, 100000),
                 'tip': 'Buffer size for shuffling'}
            ],
            'TFRecordDataLoader': [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for data loading'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle the data'},
                {'name': 'buffer_size', 'type': 'int', 'value': 10000, 'limits': (1, 100000),
                 'tip': 'Buffer size for shuffling'},
                {'name': 'num_parallel_calls', 'type': 'int', 'value': -1, 'limits': (-1, 64),
                 'tip': 'Number of parallel calls for data processing (-1 for auto)'},
                {'name': 'prefetch_buffer_size', 'type': 'int', 'value': -1, 'limits': (-1, 100),
                 'tip': 'Prefetch buffer size (-1 for auto)'}
            ],
            'ImageDataLoader': [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for data loading'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle the data'},
                {'name': 'image_size', 'type': 'list', 'value': [224, 224],
                 'tip': 'Target image size [height, width]'},
                {'name': 'color_mode', 'type': 'list', 'limits': ['rgb', 'grayscale'], 'value': 'rgb',
                 'tip': 'Color mode for images'},
                {'name': 'interpolation', 'type': 'list', 'limits': ['bilinear', 'nearest', 'bicubic'], 'value': 'bilinear',
                 'tip': 'Interpolation method for resizing'}
            ],
            'CSVDataLoader': [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for data loading'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle the data'},
                {'name': 'delimiter', 'type': 'str', 'value': ',',
                 'tip': 'CSV delimiter character'},
                {'name': 'header', 'type': 'bool', 'value': True,
                 'tip': 'Whether CSV has header row'},
                {'name': 'skip_blank_lines', 'type': 'bool', 'value': True,
                 'tip': 'Skip blank lines in CSV'}
            ],
            'HDF5DataLoader': [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for data loading'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle the data'},
                {'name': 'dataset_key', 'type': 'str', 'value': 'data',
                 'tip': 'Key for the dataset in HDF5 file'},
                {'name': 'label_key', 'type': 'str', 'value': 'labels',
                 'tip': 'Key for the labels in HDF5 file'},
                {'name': 'compression', 'type': 'list', 'limits': ['gzip', 'lzf', 'szip', 'none'], 'value': 'gzip',
                 'tip': 'Compression method for HDF5 file'}
            ]
        }
        
        return data_loader_parameters.get(data_loader_name, [])
    
    def _add_custom_button(self):
        """Add button for loading custom data loaders."""
        self.addChild({
            'name': 'Load Custom Data Loader',
            'type': 'action',
            'tip': 'Load custom data loader functions from Python files'
        })
        
        # Connect the action
        custom_button = self.child('Load Custom Data Loader')
        if custom_button:
            custom_button.sigActivated.connect(self._load_custom_data_loader)
    
    def _load_custom_data_loader(self):
        """Load custom data loader from Python file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Custom Data Loader Python File",
            "",
            "Python files (*.py)"
        )
        
        if not file_path:
            return
            
        try:
            # Parse the Python file to find data loader functions and classes
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            data_loaders = []
            
            # Find both functions and classes that could be data loaders
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function seems like a data loader
                    if self._is_valid_data_loader_function(node):
                        data_loaders.append(('function', node.name))
                elif isinstance(node, ast.ClassDef):
                    # Check if class seems like a data loader
                    if self._is_valid_data_loader_class(node):
                        data_loaders.append(('class', node.name))
            
            if not data_loaders:
                QMessageBox.warning(None, "No Data Loaders Found", 
                                  "No valid data loader functions or classes found in the selected Python file.\n\n"
                                  "Data loaders should be functions that return tf.data.Dataset or classes with appropriate methods.")
                return
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_data_loader", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Store custom data loaders and their parameters
            for loader_type, name in data_loaders:
                if hasattr(module, name):
                    loader = getattr(module, name)
                    custom_name = f"Custom_{name}"
                    
                    # Store the function/class
                    if not hasattr(self, '_custom_data_loaders'):
                        self._custom_data_loaders = {}
                    self._custom_data_loaders[custom_name] = {
                        'loader': loader,
                        'type': loader_type,
                        'file_path': file_path,
                        'original_name': name
                    }
                    
                    # Extract parameters from function/class signature
                    self._extract_custom_data_loader_parameters(custom_name, loader, loader_type)
            
            # Update data loader options
            self._refresh_data_loader_options()
            
            # Show success message
            QMessageBox.information(None, "Custom Data Loaders Loaded", 
                                  f"Successfully loaded {len(data_loaders)} custom data loader(s) from:\n{os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(None, "Error Loading Custom Data Loader", 
                               f"Failed to load custom data loader:\n{str(e)}")
    
    def _is_valid_data_loader_function(self, node):
        """Check if a function node is a valid data loader function."""
        func_name = node.name.lower()
        
        # Check function name patterns
        valid_patterns = ['load', 'dataset', 'data', 'batch', 'input_fn', 'get_data']
        if any(pattern in func_name for pattern in valid_patterns):
            return True
        
        # Check if function has common data loader parameters
        arg_names = [arg.arg.lower() for arg in node.args.args]
        data_loader_args = ['batch_size', 'shuffle', 'data_dir', 'file_pattern', 'split']
        if any(arg in arg_names for arg in data_loader_args):
            return True
            
        return False
    
    def _is_valid_data_loader_class(self, node):
        """Check if a class node is a valid data loader class."""
        class_name = node.name.lower()
        
        # Check class name patterns
        valid_patterns = ['loader', 'dataset', 'data', 'reader', 'input']
        if any(pattern in class_name for pattern in valid_patterns):
            return True
            
        # Check if class has common data loader methods
        method_names = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_names.append(item.name.lower())
        
        data_loader_methods = ['load', 'get_dataset', 'build_dataset', '__call__', '__iter__']
        if any(method in method_names for method in data_loader_methods):
            return True
            
        return False
    
    def _extract_custom_data_loader_parameters(self, data_loader_name, loader, loader_type):
        """Extract parameters from custom data loader function/class signature."""
        try:
            import inspect
            
            if loader_type == 'function':
                sig = inspect.signature(loader)
            else:  # class
                # Try to get __init__ method signature
                if hasattr(loader, '__init__'):
                    sig = inspect.signature(loader.__init__)
                else:
                    # Fallback to class signature
                    sig = inspect.signature(loader)
            
            parameters = []
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter for classes
                if param_name == 'self':
                    continue
                    
                param_config = {
                    'name': param_name,
                    'tip': f'Custom parameter: {param_name}'
                }
                
                # Infer parameter type and set defaults
                if param_name.lower() in ['batch_size', 'buffer_size', 'num_classes', 'epochs', 'steps']:
                    param_config['type'] = 'int'
                    param_config['value'] = param.default if param.default != inspect.Parameter.empty else 32
                    if param_name.lower() == 'batch_size':
                        param_config['limits'] = (1, 1024)
                    elif param_name.lower() in ['buffer_size', 'steps']:
                        param_config['limits'] = (1, 100000)
                    else:
                        param_config['limits'] = (1, 10000)
                elif param_name.lower() in ['learning_rate', 'dropout_rate', 'split_ratio']:
                    param_config['type'] = 'float'
                    param_config['value'] = param.default if param.default != inspect.Parameter.empty else 0.001
                    param_config['limits'] = (0.0, 1.0)
                elif param_name.lower() in ['shuffle', 'augment', 'normalize', 'cache']:
                    param_config['type'] = 'bool'
                    param_config['value'] = param.default if param.default != inspect.Parameter.empty else True
                else:
                    # Default to string for other parameters
                    param_config['type'] = 'str'
                    param_config['value'] = param.default if param.default != inspect.Parameter.empty else ''
                
                parameters.append(param_config)
            
            # Store custom parameters
            if not hasattr(self, '_custom_data_loader_parameters'):
                self._custom_data_loader_parameters = {}
            self._custom_data_loader_parameters[data_loader_name] = parameters
            
        except Exception as e:
            # If parameter extraction fails, create basic parameter set
            if not hasattr(self, '_custom_data_loader_parameters'):
                self._custom_data_loader_parameters = {}
            self._custom_data_loader_parameters[data_loader_name] = [
                {'name': 'batch_size', 'type': 'int', 'value': 32, 'limits': (1, 1024),
                 'tip': 'Batch size for custom data loader'},
                {'name': 'shuffle', 'type': 'bool', 'value': True,
                 'tip': 'Whether to shuffle data in custom loader'}
            ]
    
    def _refresh_data_loader_options(self):
        """Refresh the data loader selection options after loading custom data loaders."""
        selection_group = self.child('Data Loader Selection')
        if selection_group:
            data_loader_selector = selection_group.child('selected_data_loader')
            if data_loader_selector:
                new_options = self._get_data_loader_options()
                data_loader_selector.setLimits(new_options)
    
    def get_data_loader_config(self):
        """Get the current data loader configuration."""
        selection_group = self.child('Data Loader Selection')
        if not selection_group:
            return None
            
        config = {}
        
        # Get selected data loader
        data_loader_selector = selection_group.child('selected_data_loader')
        if data_loader_selector:
            config['selected_data_loader'] = data_loader_selector.value()
        
        # Get usage flags
        use_for_train = selection_group.child('use_for_train')
        if use_for_train:
            config['use_for_train'] = use_for_train.value()
            
        use_for_val = selection_group.child('use_for_val')
        if use_for_val:
            config['use_for_val'] = use_for_val.value()
        
        # Get parameters
        params = {}
        for child in selection_group.children():
            if child.name() not in ['selected_data_loader', 'use_for_train', 'use_for_val']:
                params[child.name()] = child.value()
        
        config['parameters'] = params
        
        # Add custom data loader info if applicable
        selected_name = config.get('selected_data_loader', '')
        if (hasattr(self, '_custom_data_loaders') and 
            selected_name in self._custom_data_loaders):
            config['custom_info'] = self._custom_data_loaders[selected_name]
        
        return config
    
    def set_data_loader_config(self, config):
        """Set the data loader configuration from loaded config data."""
        if not config or not isinstance(config, dict):
            return
            
        selection_group = self.child('Data Loader Selection')
        if not selection_group:
            return
            
        # Get the Data Loader Selection config
        selection_config = config.get('Data Loader Selection', {})
        if not selection_config:
            return
        
        # Set selected data loader if available in options
        selected_data_loader = selection_config.get('selected_data_loader')
        if selected_data_loader:
            data_loader_selector = selection_group.child('selected_data_loader')
            if data_loader_selector:
                # Check if the selected data loader is in the available options
                available_options = data_loader_selector.opts['limits']
                if selected_data_loader in available_options:
                    data_loader_selector.setValue(selected_data_loader)
                    # Update parameters after setting the value
                    self._update_data_loader_parameters()
                else:
                    # If the selected data loader is not available, keep default but log
                    print(f"Warning: Selected data loader '{selected_data_loader}' not found in available options: {available_options}")
            else:
                print("Warning: data_loader_selector parameter not found")
        
        # Set usage flags
        use_for_train = selection_config.get('use_for_train')
        if use_for_train is not None:
            train_param = selection_group.child('use_for_train')
            if train_param:
                train_param.setValue(use_for_train)
                
        use_for_val = selection_config.get('use_for_val')
        if use_for_val is not None:
            val_param = selection_group.child('use_for_val')
            if val_param:
                val_param.setValue(use_for_val)
        
        # Set parameter values
        for param_name, param_value in selection_config.items():
            if param_name not in ['selected_data_loader', 'use_for_train', 'use_for_val']:
                param = selection_group.child(param_name)
                if param:
                    try:
                        param.setValue(param_value)
                    except Exception as e:
                        print(f"Warning: Could not set parameter '{param_name}' to '{param_value}': {e}")
    
    def load_custom_data_loader_from_metadata(self, loader_info):
        """Load custom data loader from metadata info."""
        try:
            file_path = loader_info.get('file_path', '')
            function_name = loader_info.get('function_name', '')
            loader_type = loader_info.get('type', 'function')
            
            if not os.path.exists(file_path):
                print(f"Warning: Custom data loader file not found: {file_path}")
                return False
                
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_data_loader", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, function_name):
                print(f"Warning: Function '{function_name}' not found in {file_path}")
                return False
                
            loader = getattr(module, function_name)
            custom_name = f"Custom_{function_name}"
            
            # Store the function/class
            if not hasattr(self, '_custom_data_loaders'):
                self._custom_data_loaders = {}
                
            self._custom_data_loaders[custom_name] = {
                'loader': loader,
                'type': loader_type,
                'file_path': file_path,
                'original_name': function_name
            }
            
            # Extract parameters
            self._extract_custom_data_loader_parameters(custom_name, loader, loader_type)
            
            # Update data loader options
            self._refresh_data_loader_options()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom data loader from metadata: {e}")
            import traceback
            traceback.print_exc()
            return False
