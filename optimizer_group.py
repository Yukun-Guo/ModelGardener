"""
Optimizer Group Parameter for ModelGardener

This module provides a custom parameter group for configuring optimizers,
following the same pattern as loss functions group with parent tree item structure.
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


class OptimizerGroup:
    """Custom optimizer group that includes preset optimizers and allows loading custom optimizers from files."""
    
    def __init__(self, **opts):
        self.config = {
            'Optimizer Selection': {
                'selected_optimizer': 'Adam',
                'learning_rate': 0.001,
                'use_learning_rate_schedule': False
            },
            'Learning Rate Schedule': {
                'schedule_type': 'exponential_decay',
                'decay_steps': 1000,
                'decay_rate': 0.96,
                'staircase': False
            },
            'Custom Parameters': {}
        }
        
        # Initialize custom optimizer storage
        self._custom_optimizers = {}
        self._custom_optimizer_parameters = {}
        
    def get_config(self):
        """Get the current optimizer configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the optimizer configuration."""
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
        
        # Update dependent configurations
        if path == 'Optimizer Selection.selected_optimizer':
            self._update_optimizer_parameters()
    
    def _get_optimizer_options(self):
        """Get list of available optimizer names including custom ones."""
        base_options = [
            'Adam',
            'SGD',
            'RMSprop',
            'Adagrad',
            'Adadelta',
            'Adamax',
            'Nadam',
            'Ftrl'
        ]
        
        # Add custom optimizers if any
        if hasattr(self, '_custom_optimizers'):
            custom_options = list(self._custom_optimizers.keys())
            return base_options + custom_options
        
        return base_options
    
    def _update_optimizer_parameters(self):
        """Update optimizer parameters based on selected optimizer."""
        selected = self.config['Optimizer Selection']['selected_optimizer']
        
        # Set default parameters based on optimizer type
        if selected == 'Adam':
            self.config['Optimizer Selection'].update({
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-7,
                'amsgrad': False
            })
        elif selected == 'SGD':
            self.config['Optimizer Selection'].update({
                'momentum': 0.0,
                'nesterov': False
            })
        elif selected == 'RMSprop':
            self.config['Optimizer Selection'].update({
                'rho': 0.9,
                'momentum': 0.0,
                'epsilon': 1e-7,
                'centered': False
            })
        elif selected == 'Adagrad':
            self.config['Optimizer Selection'].update({
                'initial_accumulator_value': 0.1,
                'epsilon': 1e-7
            })
        elif selected == 'Adadelta':
            self.config['Optimizer Selection'].update({
                'rho': 0.95,
                'epsilon': 1e-7
            })
        elif selected == 'Adamax':
            self.config['Optimizer Selection'].update({
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-7
            })
        elif selected == 'Nadam':
            self.config['Optimizer Selection'].update({
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-7
            })
        elif selected == 'Ftrl':
            self.config['Optimizer Selection'].update({
                'learning_rate_power': -0.5,
                'initial_accumulator_value': 0.1,
                'l1_regularization_strength': 0.0,
                'l2_regularization_strength': 0.0
            })
    
    def load_custom_optimizers(self, file_path):
        """Load custom optimizer functions from a file."""
        try:
            custom_functions = self._extract_optimizer_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid optimizer functions found in the selected file.\n\n"
                    "Functions should be TensorFlow optimizer classes or functions that return optimizers."
                )
                return False
            
            # Add custom functions to the available optimizer options
            for func_name, func_info in custom_functions.items():
                self._add_custom_optimizer_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom optimizer(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese optimizers are now available in the selection dropdown."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom optimizers from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_optimizer_option(self, func_name, func_info):
        """Add a custom optimizer as an option."""
        # Store the function with metadata
        self._custom_optimizers[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_optimizer_parameters[func_name] = func_info['parameters']
    
    def _extract_optimizer_functions(self, file_path):
        """Extract optimizer function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely an optimizer function
                    if self._is_optimizer_function(node):
                        func_info = self._analyze_optimizer_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's an optimizer class
                    if self._is_optimizer_class(node):
                        class_info = self._analyze_optimizer_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_optimizer_function(self, node):
        """Check if a function node is likely an optimizer function."""
        # Check function name for optimizer indicators
        optimizer_keywords = ['optimizer', 'adam', 'sgd', 'rmsprop', 'adagrad']
        has_optimizer_name = any(keyword in node.name.lower() for keyword in optimizer_keywords)
        
        # Check if function has common optimizer parameters
        args = [arg.arg for arg in node.args.args]
        optimizer_args = ['learning_rate', 'lr', 'momentum', 'beta_1', 'beta_2']
        has_optimizer_args = any(arg in args for arg in optimizer_args)
        
        return has_optimizer_name or has_optimizer_args
    
    def _is_optimizer_class(self, node):
        """Check if a class node is likely an optimizer class."""
        # Check class name for optimizer indicators
        optimizer_keywords = ['optimizer', 'adam', 'sgd', 'rmsprop', 'adagrad']
        has_optimizer_name = any(keyword in node.name.lower() for keyword in optimizer_keywords)
        
        # Check if class has apply_gradients or minimize methods
        has_optimizer_methods = any(isinstance(child, ast.FunctionDef) and 
                                  child.name in ['apply_gradients', 'minimize', 'get_updates'] 
                                  for child in node.body)
        
        return has_optimizer_name or has_optimizer_methods
    
    def _analyze_optimizer_function(self, node, file_path):
        """Analyze an optimizer function and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom optimizer function: {node.name}"
            
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
    
    def _analyze_optimizer_class(self, node, file_path):
        """Analyze an optimizer class and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom optimizer class: {node.name}"
            
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
        
        if any(word in param_name_lower for word in ['rate', 'lr', 'momentum', 'beta', 'epsilon', 'rho', 'decay']):
            return 'float'
        elif any(word in param_name_lower for word in ['steps', 'iterations', 'accumulator']):
            return 'int'
        elif any(word in param_name_lower for word in ['use', 'enable', 'nesterov', 'amsgrad', 'centered', 'staircase']):
            return 'bool'
        else:
            return 'float'  # Default to float for optimizers
    
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
    
    def get_optimizer_config(self):
        """Get the current optimizer configuration for training."""
        config = {}
        
        selection = self.config['Optimizer Selection']
        lr_schedule = self.config['Learning Rate Schedule']
        
        config['optimizer'] = selection['selected_optimizer']
        config['learning_rate'] = selection['learning_rate']
        config['use_learning_rate_schedule'] = selection['use_learning_rate_schedule']
        
        # Add optimizer-specific parameters
        for key, value in selection.items():
            if key not in ['selected_optimizer', 'learning_rate', 'use_learning_rate_schedule']:
                config[key] = value
        
        # Add learning rate schedule if enabled
        if selection['use_learning_rate_schedule']:
            config['learning_rate_schedule'] = dict(lr_schedule)
        
        # Add custom parameters if using custom optimizer
        if selection['selected_optimizer'] in self._custom_optimizers:
            custom_info = self._custom_optimizers[selection['selected_optimizer']]
            config['custom_optimizer'] = custom_info
        
        return config
    
    def load_custom_optimizer_from_metadata(self, metadata):
        """Load a custom optimizer from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_optimizer", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_optimizers[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom optimizer: {str(e)}")
            return False
