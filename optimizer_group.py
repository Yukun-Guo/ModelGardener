"""
Optimizer Group Parameter for ModelGardener

This module provides a custom parameter group for configuring optimizers,
following the same pattern as loss functions group with parent tree item structure.
"""

import os
import importlib.util
import ast
from typing import Dict, Any, List, Optional
from PySide6.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph.parametertree.parameterTypes as pTypes


class OptimizerGroup(pTypes.GroupParameter):
    """Custom optimizer group that includes preset optimizers and allows loading custom optimizers from files."""
    
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Initialize custom optimizer storage
        self._custom_optimizers = {}
        self._custom_optimizer_parameters = {}
        
        # Add optimizer selection
        self._add_optimizer_selection()
        
        # Add custom optimizer button
        self._add_custom_button()
    
    def _add_optimizer_selection(self):
        """Add optimizer selection with preset options."""
        optimizer_options = self._get_optimizer_options()
        
        self.addChild({
            'name': 'Optimizer Selection',
            'type': 'group',
            'children': [
                {'name': 'selected_optimizer', 'type': 'list', 'limits': optimizer_options, 'value': 'Adam', 
                 'tip': 'Select the optimizer to use for training'}
            ],
            'tip': 'Choose optimizer type and configure its parameters'
        })
        
        # Add optimizer parameters
        self._add_selected_optimizer_parameters()
    
    def _get_optimizer_options(self):
        """Get list of available optimizer names including custom ones."""
        base_options = [
            'Adam',
            'SGD', 
            'RMSprop',
            'Adagrad',
            'AdamW',
            'Adadelta',
            'Adamax',
            'Nadam',
            'FTRL'
        ]
        
        # Add custom optimizers if any
        if hasattr(self, '_custom_optimizers') and self._custom_optimizers:
            custom_options = list(self._custom_optimizers.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_selected_optimizer_parameters(self):
        """Add parameters for the selected optimizer."""
        parent = self.child('Optimizer Selection')
        if not parent:
            return
            
        # Connect selection change to parameter update
        if parent.child('selected_optimizer'):
            parent.child('selected_optimizer').sigValueChanged.connect(
                self._update_optimizer_parameters
            )
        
        # Add initial parameters
        self._update_optimizer_parameters()
    
    def _update_optimizer_parameters(self):
        """Update optimizer parameters based on selection."""
        parent = self.child('Optimizer Selection')
        if not parent:
            return
            
        selected_optimizer = parent.child('selected_optimizer').value()
        
        # Remove existing parameters (except selected_optimizer)
        existing_params = []
        for child in parent.children():
            if child.name() not in ['selected_optimizer']:
                existing_params.append(child)
        
        for param in existing_params:
            parent.removeChild(param)
        
        # Add parameters based on selected optimizer
        optimizer_params = self._get_optimizer_parameters(selected_optimizer)
        for param_config in optimizer_params:
            parent.addChild(param_config)
    
    def _get_optimizer_parameters(self, optimizer_name):
        """Get parameters for a specific optimizer."""
        # Check if it's a custom optimizer
        if hasattr(self, '_custom_optimizer_parameters') and optimizer_name in self._custom_optimizer_parameters:
            return self._custom_optimizer_parameters[optimizer_name]
        
        # Return built-in optimizer parameters
        optimizer_parameters = {
            'Adam': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0), 
                 'tip': 'Learning rate for Adam optimizer'},
                {'name': 'beta_1', 'type': 'float', 'value': 0.9, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for first moment estimates'},
                {'name': 'beta_2', 'type': 'float', 'value': 0.999, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for second moment estimates'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'},
                {'name': 'amsgrad', 'type': 'bool', 'value': False, 
                 'tip': 'Whether to use AMSGrad variant'}
            ],
            'SGD': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.01, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for SGD optimizer'},
                {'name': 'momentum', 'type': 'float', 'value': 0.0, 'limits': (0.0, 1.0),
                 'tip': 'Momentum factor'},
                {'name': 'nesterov', 'type': 'bool', 'value': False, 
                 'tip': 'Whether to use Nesterov momentum'}
            ],
            'RMSprop': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for RMSprop optimizer'},
                {'name': 'rho', 'type': 'float', 'value': 0.9, 'limits': (0.0, 1.0),
                 'tip': 'Discounting factor for the history/coming gradient'},
                {'name': 'momentum', 'type': 'float', 'value': 0.0, 'limits': (0.0, 1.0),
                 'tip': 'Momentum factor'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'Adagrad': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for Adagrad optimizer'},
                {'name': 'initial_accumulator_value', 'type': 'float', 'value': 0.1, 'limits': (0.0, 10.0),
                 'tip': 'Starting value for the accumulators'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'AdamW': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for AdamW optimizer'},
                {'name': 'weight_decay', 'type': 'float', 'value': 0.01, 'limits': (0.0, 1.0),
                 'tip': 'Weight decay coefficient'},
                {'name': 'beta_1', 'type': 'float', 'value': 0.9, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for first moment estimates'},
                {'name': 'beta_2', 'type': 'float', 'value': 0.999, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for second moment estimates'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'Adadelta': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for Adadelta optimizer'},
                {'name': 'rho', 'type': 'float', 'value': 0.95, 'limits': (0.0, 1.0),
                 'tip': 'Coefficient used for computing a running average of squared gradients'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'Adamax': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for Adamax optimizer'},
                {'name': 'beta_1', 'type': 'float', 'value': 0.9, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for first moment estimates'},
                {'name': 'beta_2', 'type': 'float', 'value': 0.999, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for weighted infinity norm'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'Nadam': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for Nadam optimizer'},
                {'name': 'beta_1', 'type': 'float', 'value': 0.9, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for first moment estimates'},
                {'name': 'beta_2', 'type': 'float', 'value': 0.999, 'limits': (0.0, 1.0),
                 'tip': 'Exponential decay rate for second moment estimates'},
                {'name': 'epsilon', 'type': 'float', 'value': 1e-07, 'limits': (1e-10, 1e-3),
                 'tip': 'Small constant for numerical stability'}
            ],
            'FTRL': [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for FTRL optimizer'},
                {'name': 'learning_rate_power', 'type': 'float', 'value': -0.5, 'limits': (-1.0, 0.0),
                 'tip': 'Controls how the learning rate decreases during training'},
                {'name': 'initial_accumulator_value', 'type': 'float', 'value': 0.1, 'limits': (0.0, 10.0),
                 'tip': 'Starting value for the accumulators'},
                {'name': 'l1_regularization_strength', 'type': 'float', 'value': 0.0, 'limits': (0.0, 1.0),
                 'tip': 'L1 regularization strength'},
                {'name': 'l2_regularization_strength', 'type': 'float', 'value': 0.0, 'limits': (0.0, 1.0),
                 'tip': 'L2 regularization strength'}
            ]
        }
        
        return optimizer_parameters.get(optimizer_name, [])
    
    def _add_custom_button(self):
        """Add button for loading custom optimizers."""
        self.addChild({
            'name': 'Load Custom Optimizer',
            'type': 'action',
            'tip': 'Load custom optimizer functions from Python files'
        })
        
        # Connect the action
        custom_button = self.child('Load Custom Optimizer')
        if custom_button:
            custom_button.sigActivated.connect(self._load_custom_optimizer)
    
    def _load_custom_optimizer(self):
        """Load custom optimizer from Python file."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Custom Optimizer Python File",
            "",
            "Python files (*.py)"
        )
        
        if not file_path:
            return
            
        try:
            # Parse the Python file to find optimizer functions
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            if not functions:
                QMessageBox.warning(None, "No Functions Found", 
                                  "No functions found in the selected Python file.")
                return
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_optimizer", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Store custom optimizers and their parameters
            for function_name in functions:
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    custom_name = f"Custom_{function_name}"
                    
                    # Store the function with metadata
                    if not hasattr(self, '_custom_optimizers'):
                        self._custom_optimizers = {}
                    if not hasattr(self, '_custom_optimizer_metadata'):
                        self._custom_optimizer_metadata = {}
                    
                    self._custom_optimizers[custom_name] = func
                    self._custom_optimizer_metadata[custom_name] = {
                        'file_path': file_path,
                        'function_name': function_name,
                        'type': 'function'
                    }
                    
                    # Extract parameters from function signature
                    self._extract_custom_optimizer_parameters(custom_name, func)
            
            # Update optimizer options
            self._refresh_optimizer_options()
            
            # Show success message
            QMessageBox.information(None, "Custom Optimizers Loaded", 
                                  f"Successfully loaded {len(functions)} custom optimizer(s) from:\n{os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(None, "Error Loading Custom Optimizer", 
                               f"Failed to load custom optimizer:\n{str(e)}")
    
    def _extract_custom_optimizer_parameters(self, optimizer_name, func):
        """Extract parameters from custom optimizer function signature."""
        try:
            import inspect
            sig = inspect.signature(func)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                param_config = {
                    'name': param_name,
                    'type': 'float' if param_name in ['learning_rate', 'lr'] else 'str',
                    'value': param.default if param.default != inspect.Parameter.empty else (0.001 if param_name in ['learning_rate', 'lr'] else ''),
                    'tip': f'Custom parameter: {param_name}'
                }
                
                # Set appropriate limits for common parameters
                if param_name in ['learning_rate', 'lr']:
                    param_config['limits'] = (1e-8, 1.0)
                elif 'rate' in param_name.lower() or 'factor' in param_name.lower():
                    param_config['type'] = 'float'
                    param_config['limits'] = (0.0, 1.0)
                    if param.default == inspect.Parameter.empty:
                        param_config['value'] = 0.1
                
                parameters.append(param_config)
            
            # Store custom parameters
            if not hasattr(self, '_custom_optimizer_parameters'):
                self._custom_optimizer_parameters = {}
            self._custom_optimizer_parameters[optimizer_name] = parameters
            
        except Exception as e:
            # If parameter extraction fails, create basic parameter set
            if not hasattr(self, '_custom_optimizer_parameters'):
                self._custom_optimizer_parameters = {}
            self._custom_optimizer_parameters[optimizer_name] = [
                {'name': 'learning_rate', 'type': 'float', 'value': 0.001, 'limits': (1e-8, 1.0),
                 'tip': 'Learning rate for custom optimizer'}
            ]
    
    def _refresh_optimizer_options(self):
        """Refresh the optimizer selection options after loading custom optimizers."""
        selection_group = self.child('Optimizer Selection')
        if selection_group:
            optimizer_selector = selection_group.child('selected_optimizer')
            if optimizer_selector:
                new_options = self._get_optimizer_options()
                optimizer_selector.setLimits(new_options)
    
    def set_optimizer_config(self, config):
        """Set the optimizer configuration from loaded config data."""
        if not config or not isinstance(config, dict):
            return
            
        selection_group = self.child('Optimizer Selection')
        if not selection_group:
            return
            
        # Get the Optimizer Selection config
        selection_config = config.get('Optimizer Selection', {})
        if not selection_config:
            return
        
        # Set selected optimizer if available in options
        selected_optimizer = selection_config.get('selected_optimizer')
        if selected_optimizer:
            optimizer_selector = selection_group.child('selected_optimizer')
            if optimizer_selector:
                # Check if the selected optimizer is in the available options
                available_options = optimizer_selector.opts['limits']
                if selected_optimizer in available_options:
                    optimizer_selector.setValue(selected_optimizer)
                    # Update parameters after setting the value
                    self._update_optimizer_parameters()
                else:
                    # If the selected optimizer is not available, keep default but log
                    print(f"Warning: Selected optimizer '{selected_optimizer}' not found in available options: {available_options}")
            else:
                print("Warning: optimizer_selector parameter not found")
        
        # Set parameter values
        for param_name, param_value in selection_config.items():
            if param_name not in ['selected_optimizer']:
                try:
                    param = selection_group.child(param_name)
                    if param:
                        try:
                            param.setValue(param_value)
                        except Exception as e:
                            print(f"Warning: Could not set parameter '{param_name}' to '{param_value}': {e}")
                    else:
                        print(f"Warning: Parameter '{param_name}' not found in current optimizer configuration")
                except KeyError as e:
                    print(f"Warning: Parameter '{param_name}' not found in selection group: {e}")
                except Exception as e:
                    print(f"Warning: Error accessing parameter '{param_name}': {e}")
    
    def load_custom_optimizer_from_metadata(self, optimizer_info):
        """Load custom optimizer from metadata info."""
        try:
            file_path = optimizer_info.get('file_path', '')
            function_name = optimizer_info.get('function_name', '') or optimizer_info.get('original_name', '')
            optimizer_type = optimizer_info.get('type', 'function')
            
            # Check for empty function name
            if not function_name:
                print(f"Warning: Empty function name in custom optimizer metadata for {file_path}")
                return False
            
            if not os.path.exists(file_path):
                print(f"Warning: Custom optimizer file not found: {file_path}")
                return False
                
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_optimizer", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, function_name):
                print(f"Warning: Function '{function_name}' not found in {file_path}")
                return False
                
            optimizer = getattr(module, function_name)
            custom_name = f"Custom_{function_name}"
            
            # Store the function/class with metadata
            if not hasattr(self, '_custom_optimizers'):
                self._custom_optimizers = {}
            if not hasattr(self, '_custom_optimizer_metadata'):
                self._custom_optimizer_metadata = {}
                
            self._custom_optimizers[custom_name] = optimizer
            self._custom_optimizer_metadata[custom_name] = {
                'file_path': file_path,
                'function_name': function_name,
                'type': optimizer_type
            }
            
            # Extract parameters
            self._extract_custom_optimizer_parameters(custom_name, optimizer)
            
            # Update optimizer options
            self._refresh_optimizer_options()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom optimizer from metadata: {e}")
            import traceback
            traceback.print_exc()
            return False

