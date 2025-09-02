"""
Training Loop Group Parameter for ModelGardener

This module provides a custom parameter group for configuring training loops,
following the same pattern as other custom function groups with load button functionality.
"""

import os
import ast
import importlib.util
import inspect
from typing import Dict, Any, List, Optional

# CLI-only message functions (no GUI dialogs)
def cli_info(title, message):
    """CLI alternative to QMessageBox.information"""
    print(f"[INFO] {title}: {message}")

def cli_warning(title, message):
    """CLI alternative to QMessageBox.warning"""
    print(f"[WARNING] {title}: {message}")

def cli_error(title, message):
    """CLI alternative to QMessageBox.critical"""
    print(f"[ERROR] {title}: {message}")

def cli_get_file_path(title="Select File", file_filter="Python Files (*.py)"):
    """CLI alternative to QFileDialog.getOpenFileName - returns empty for CLI mode"""
    print(f"[CLI] File dialog requested: {title} - {file_filter}")
    print("[CLI] File dialogs not supported in CLI mode. Use config files to specify custom functions.")
    return "", ""

import pyqtgraph.parametertree.parameterTypes as pTypes


class TrainingLoopGroup(pTypes.GroupParameter):
    """Custom training loop group that includes preset training configurations and allows loading custom training loops from files."""
    
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add training strategy selection
        self._add_training_strategy()
        
        # Add custom training loop button
        self._add_custom_button()
    
    def _add_training_strategy(self):
        """Add training strategy selection."""
        self.addChild({
            'name': 'Training Strategy',
            'type': 'group',
            'children': [
                {
                    'name': 'selected_strategy',
                    'type': 'list',
                    'limits': self._get_training_strategy_options(),
                    'value': 'Standard Training',
                    'tip': 'Select the training strategy/loop to use'
                },
                {
                    'name': 'use_distributed',
                    'type': 'bool',
                    'value': False,
                    'tip': 'Enable distributed training across multiple devices'
                },
                {
                    'name': 'mixed_precision',
                    'type': 'bool',
                    'value': False,
                    'tip': 'Enable automatic mixed precision training'
                }
            ],
            'tip': 'Configure the training loop strategy and options'
        })
        
        # Connect strategy change to parameter update
        strategy_group = self.child('Training Strategy')
        strategy_group.child('selected_strategy').sigValueChanged.connect(self._update_strategy_parameters)
        
        # Add initial parameters
        self._add_strategy_parameters()
    
    def _get_training_strategy_options(self):
        """Get list of available training strategy names including custom ones."""
        base_options = [
            'Standard Training',
            'Progressive Training',
            'Curriculum Learning',
            'Adversarial Training',
            'Self-Supervised Training'
        ]
        
        # Add custom training strategies if any
        if hasattr(self, '_custom_training_strategies'):
            custom_options = list(self._custom_training_strategies.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_strategy_parameters(self):
        """Add parameters for the selected training strategy."""
        strategy_group = self.child('Training Strategy')
        if not strategy_group:
            return
            
        selected_strategy = strategy_group.child('selected_strategy').value()
        
        # Remove existing strategy-specific parameters
        for child in list(self.children()):
            if child.name().startswith('Strategy Config'):
                self.removeChild(child)
        
        # Add strategy-specific parameters
        strategy_params = self._get_strategy_parameters(selected_strategy)
        if strategy_params:
            self.addChild({
                'name': 'Strategy Config',
                'type': 'group',
                'children': strategy_params,
                'tip': f'Configuration parameters for {selected_strategy}'
            })
    
    def _get_strategy_parameters(self, strategy_name):
        """Get parameters for a specific training strategy."""
        if strategy_name == 'Standard Training':
            return [
                {'name': 'gradient_accumulation_steps', 'type': 'int', 'value': 1, 'limits': (1, 100), 'tip': 'Number of steps to accumulate gradients'},
                {'name': 'gradient_clipping', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': 'Gradient clipping value'},
                {'name': 'warmup_steps', 'type': 'int', 'value': 0, 'limits': (0, 10000), 'tip': 'Number of warmup steps'},
            ]
        elif strategy_name == 'Progressive Training':
            return [
                {'name': 'initial_resolution', 'type': 'int', 'value': 32, 'limits': (16, 512), 'tip': 'Initial image resolution'},
                {'name': 'final_resolution', 'type': 'int', 'value': 224, 'limits': (32, 1024), 'tip': 'Final image resolution'},
                {'name': 'progression_schedule', 'type': 'str', 'value': 'linear', 'tip': 'How to progress resolution: linear, exponential'},
            ]
        elif strategy_name == 'Curriculum Learning':
            return [
                {'name': 'difficulty_metric', 'type': 'str', 'value': 'loss', 'tip': 'Metric to determine sample difficulty'},
                {'name': 'curriculum_schedule', 'type': 'str', 'value': 'linear', 'tip': 'How to schedule curriculum: linear, exponential'},
                {'name': 'easy_samples_ratio', 'type': 'float', 'value': 0.3, 'limits': (0.1, 0.8), 'tip': 'Initial ratio of easy samples'},
            ]
        elif strategy_name == 'Adversarial Training':
            return [
                {'name': 'adversarial_method', 'type': 'list', 'limits': ['FGSM', 'PGD', 'C&W'], 'value': 'PGD', 'tip': 'Adversarial attack method'},
                {'name': 'epsilon', 'type': 'float', 'value': 0.3, 'limits': (0.01, 1.0), 'tip': 'Maximum perturbation magnitude'},
                {'name': 'adversarial_ratio', 'type': 'float', 'value': 0.5, 'limits': (0.1, 1.0), 'tip': 'Ratio of adversarial examples'},
            ]
        elif strategy_name == 'Self-Supervised Training':
            return [
                {'name': 'pretext_task', 'type': 'list', 'limits': ['rotation', 'jigsaw', 'colorization', 'contrastive'], 'value': 'contrastive', 'tip': 'Self-supervised pretext task'},
                {'name': 'pretraining_epochs', 'type': 'int', 'value': 100, 'limits': (10, 1000), 'tip': 'Number of pretraining epochs'},
                {'name': 'fine_tuning_lr', 'type': 'float', 'value': 0.001, 'limits': (1e-6, 0.1), 'tip': 'Learning rate for fine-tuning'},
            ]
        
        # Check if it's a custom strategy
        if hasattr(self, '_custom_training_strategies') and strategy_name in self._custom_training_strategies:
            custom_strategy = self._custom_training_strategies[strategy_name]
            return self._generate_custom_parameters(custom_strategy)
        
        return []
    
    def _generate_custom_parameters(self, custom_strategy):
        """Generate parameters for a custom training strategy based on function signature."""
        params = []
        
        for param_info in custom_strategy.get('parameters', []):
            param_name = param_info['name']
            param_type = param_info.get('type', 'str')
            default_value = param_info.get('default', None)
            
            param_config = {
                'name': param_name,
                'tip': f'Parameter for custom training strategy: {param_name}'
            }
            
            if param_type == 'int':
                param_config.update({
                    'type': 'int',
                    'value': default_value if default_value is not None else 1,
                    'limits': (1, 1000)
                })
            elif param_type == 'float':
                param_config.update({
                    'type': 'float', 
                    'value': default_value if default_value is not None else 1.0,
                    'limits': (0.0, 10.0)
                })
            elif param_type == 'bool':
                param_config.update({
                    'type': 'bool',
                    'value': default_value if default_value is not None else False
                })
            else:
                param_config.update({
                    'type': 'str',
                    'value': str(default_value) if default_value is not None else ''
                })
            
            params.append(param_config)
        
        return params
    
    def _update_strategy_parameters(self):
        """Update parameters when training strategy changes."""
        self._add_strategy_parameters()
    
    def _add_custom_button(self):
        """Add button to load custom training loops."""
        self.addChild({
            'name': 'Load Custom Training Loop',
            'type': 'action',
            'title': 'Load Custom Training Loop...',
            'tip': 'Load custom training loop function from a Python file'
        })
        
        # Connect the button
        load_button = self.child('Load Custom Training Loop')
        load_button.sigActivated.connect(self._load_custom_training_loop)
    
    def _load_custom_training_loop(self):
        """Load custom training loop functions from a Python file."""
        file_path, _ = cli_get_file_path(
            "Load Custom Training Loop",
            "Python Files (*.py);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_training_loop_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid training loop functions found in the selected file.\n\n"
                    "Functions should be named with 'train' or 'loop' in the name and accept appropriate training parameters."
                )
                return
            
            # Add custom functions to the available training strategy options
            for func_name, func_info in custom_functions.items():
                self._add_custom_training_strategy(func_name, func_info)
            
            # Update training strategy dropdown
            self._update_training_strategy_options()
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom training loop function(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese functions are now available in the training strategy selection."
            )
                
        except Exception as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom training loop functions from file:\n{str(e)}"
            )
    
    def _extract_training_loop_functions(self, file_path):
        """Extract training loop functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Parse the AST
            tree = ast.parse(file_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if this looks like a training loop function
                    if self._is_training_loop_function(node, func_name):
                        func_info = {
                            'name': func_name,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function',
                            'parameters': self._extract_function_parameters(node),
                            'description': ast.get_docstring(node) or f'Custom training loop: {func_name}'
                        }
                        
                        custom_functions[func_name] = func_info
                
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if this looks like a training loop class
                    if self._is_training_loop_class(node, class_name):
                        class_info = {
                            'name': class_name,
                            'file_path': file_path,
                            'class_name': class_name,
                            'type': 'class',
                            'parameters': self._extract_class_parameters(node),
                            'description': ast.get_docstring(node) or f'Custom training loop class: {class_name}'
                        }
                        
                        custom_functions[class_name] = class_info
        
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_training_loop_function(self, node, func_name):
        """Check if a function is likely a training loop function."""
        # Check function name for training-related keywords
        training_keywords = ['train', 'loop', 'epoch', 'step', 'fit', 'learn']
        name_lower = func_name.lower()
        
        if any(keyword in name_lower for keyword in training_keywords):
            return True
        
        # Check function arguments for training-related parameters
        arg_names = [arg.arg for arg in node.args.args]
        training_arg_keywords = ['model', 'optimizer', 'loss', 'data', 'epochs', 'steps']
        
        if any(any(keyword in arg.lower() for keyword in training_arg_keywords) for arg in arg_names):
            return True
        
        return False
    
    def _is_training_loop_class(self, node, class_name):
        """Check if a class is likely a training loop class."""
        # Check class name for training-related keywords
        training_keywords = ['train', 'loop', 'trainer', 'fit', 'learn']
        name_lower = class_name.lower()
        
        if any(keyword in name_lower for keyword in training_keywords):
            return True
        
        # Check if class has training-related methods
        method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        training_method_keywords = ['train', 'fit', 'step', 'epoch', 'forward', 'backward']
        
        if any(any(keyword in method.lower() for keyword in training_method_keywords) for method in method_names):
            return True
        
        return False
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition."""
        params = []
        
        # Skip 'self' parameter if present
        start_idx = 1 if func_node.args.args and func_node.args.args[0].arg == 'self' else 0
        
        for arg in func_node.args.args[start_idx:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'str',  # Default type
                'default': None
            }
            
            # Basic type inference based on parameter name
            if any(keyword in param_name.lower() for keyword in ['epoch', 'step', 'batch', 'size']):
                param_info['type'] = 'int'
                param_info['default'] = 1
            elif any(keyword in param_name.lower() for keyword in ['rate', 'factor', 'ratio', 'alpha', 'beta']):
                param_info['type'] = 'float'
                param_info['default'] = 1.0
            elif any(keyword in param_name.lower() for keyword in ['enable', 'use', 'is', 'has']):
                param_info['type'] = 'bool'
                param_info['default'] = False
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - start_idx
                if 0 <= param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
                        elif isinstance(default.value, str):
                            params[param_index]['type'] = 'str'
        
        return params
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find the __init__ method
        init_method = None
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_method = node
                break
        
        if init_method:
            params = self._extract_function_parameters(init_method)
        
        return params
    
    def _add_custom_training_strategy(self, func_name, func_info):
        """Add a custom training strategy as an option in dropdowns."""
        # Store custom training strategy info for later use
        if not hasattr(self, '_custom_training_strategies'):
            self._custom_training_strategies = {}
        
        display_name = f"{func_name} (custom)"
        self._custom_training_strategies[display_name] = func_info
    
    def _update_training_strategy_options(self):
        """Update the training strategy dropdown with custom options."""
        strategy_group = self.child('Training Strategy')
        if strategy_group:
            selected_strategy_param = strategy_group.child('selected_strategy')
            if selected_strategy_param:
                new_options = self._get_training_strategy_options()
                selected_strategy_param.setLimits(new_options)
    
    def get_training_loop_config(self):
        """Get the current training loop configuration."""
        config = {}
        
        # Get training strategy configuration
        strategy_group = self.child('Training Strategy')
        if strategy_group:
            config['Training Strategy'] = {}
            for child in strategy_group.children():
                config['Training Strategy'][child.name()] = child.value()
        
        # Get strategy-specific configuration
        strategy_config = self.child('Strategy Config')
        if strategy_config:
            config['Strategy Config'] = {}
            for child in strategy_config.children():
                config['Strategy Config'][child.name()] = child.value()
        
        # Include custom training strategy info if applicable
        if hasattr(self, '_custom_training_strategies'):
            selected_strategy = strategy_group.child('selected_strategy').value()
            if selected_strategy in self._custom_training_strategies:
                config['custom_info'] = self._custom_training_strategies[selected_strategy].copy()
        
        return config
    
    def set_training_loop_config(self, config):
        """Set the training loop configuration."""
        try:
            # Set training strategy configuration
            if 'Training Strategy' in config:
                strategy_group = self.child('Training Strategy')
                if strategy_group:
                    strategy_config = config['Training Strategy']
                    for param_name, value in strategy_config.items():
                        param = strategy_group.child(param_name)
                        if param:
                            param.setValue(value)
            
            # Set strategy-specific configuration  
            if 'Strategy Config' in config:
                strategy_config_group = self.child('Strategy Config')
                if strategy_config_group:
                    strategy_params = config['Strategy Config']
                    for param_name, value in strategy_params.items():
                        param = strategy_config_group.child(param_name)
                        if param:
                            param.setValue(value)
            
            # Load custom training strategy if needed
            if 'custom_info' in config:
                custom_info = config['custom_info']
                
                # Store the custom strategy info
                if not hasattr(self, '_custom_training_strategies'):
                    self._custom_training_strategies = {}
                
                strategy_name = custom_info['name'] + ' (custom)'
                self._custom_training_strategies[strategy_name] = custom_info
                
                # Update dropdown options
                self._update_training_strategy_options()
                
        except Exception as e:
            print(f"Error setting training loop config: {e}")
    
    def load_custom_training_loop_from_metadata(self, training_loop_info):
        """Load a custom training loop from metadata (for config loading)."""
        try:
            file_path = training_loop_info.get('file_path')
            function_name = training_loop_info.get('function_name') or training_loop_info.get('class_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
                
            # Extract functions from the file  
            custom_functions = self._extract_training_loop_functions(file_path)
            
            # Find the specific function
            target_function = None
            for func_name, func_info in custom_functions.items():
                if func_info['function_name'] == function_name or func_info.get('class_name') == function_name:
                    target_function = func_info
                    break
            
            if target_function:
                # Add the custom training strategy
                self._add_custom_training_strategy(function_name, target_function)
                self._update_training_strategy_options()
                return True
                
        except Exception as e:
            print(f"Error loading custom training loop from metadata: {e}")
        
        return False
