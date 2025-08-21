import ast
import os
import importlib.util
import pyqtgraph.parametertree.parameterTypes as pTypes
from PySide6.QtWidgets import QFileDialog, QMessageBox

# Custom loss functions group that includes preset loss functions and allows adding custom loss functions from files

class LossFunctionsGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add model output configuration first
        self._add_output_configuration()
        
        # Add loss function selection
        self._add_loss_selection()
        
        # Add custom loss function button
        self._add_custom_button()
    
    def _add_output_configuration(self):
        """Add model output configuration."""
        self.addChild({
            'name': 'Model Output Configuration',
            'type': 'group',
            'children': [
                {'name': 'num_outputs', 'type': 'int', 'value': 1, 'limits': (1, 10), 'tip': 'Number of model outputs (1 for single output, >1 for multiple outputs)'},
                {'name': 'output_names', 'type': 'str', 'value': 'main_output', 'tip': 'Comma-separated names for multiple outputs (e.g., "main_output,aux_output")'},
                {'name': 'loss_strategy', 'type': 'list', 'limits': ['single_loss_all_outputs', 'different_loss_per_output'], 'value': 'single_loss_all_outputs', 'tip': 'Loss strategy: same loss for all outputs or different loss per output'}
            ],
            'tip': 'Configure model outputs and loss assignment strategy'
        })
        
        # Connect output configuration change to update loss selection
        output_config = self.child('Model Output Configuration')
        output_config.child('num_outputs').sigValueChanged.connect(self._update_output_names)
        output_config.child('num_outputs').sigValueChanged.connect(self._update_loss_selection)
        output_config.child('loss_strategy').sigValueChanged.connect(self._update_loss_selection)
    
    def _update_output_names(self):
        """Update output names based on the number of outputs."""
        output_config = self.child('Model Output Configuration')
        num_outputs = output_config.child('num_outputs').value()
        output_names_param = output_config.child('output_names')
        
        # Generate default names based on number of outputs
        if num_outputs == 1:
            output_names_param.setValue('main_output')
        else:
            # Generate names like "output_1, output_2, output_3"
            names = [f'output_{i+1}' for i in range(num_outputs)]
            output_names_param.setValue(', '.join(names))
    
    def _add_loss_selection(self):
        """Add loss function selection based on output configuration."""
        # Initially add single loss selection
        self._update_loss_selection()
    
    def _update_loss_selection(self):
        """Update loss function selection based on output configuration."""
        # Remove existing loss selection if any
        existing_groups = []
        for child in self.children():
            if child.name().startswith('Loss Selection') or child.name().startswith('Output'):
                existing_groups.append(child)
        
        for group in existing_groups:
            self.removeChild(group)
        
        # Get current configuration
        output_config = self.child('Model Output Configuration')
        num_outputs = output_config.child('num_outputs').value()
        loss_strategy = output_config.child('loss_strategy').value()
        output_names = output_config.child('output_names').value().split(',')
        output_names = [name.strip() for name in output_names if name.strip()]
        
        if num_outputs == 1 or loss_strategy == 'single_loss_all_outputs':
            # Single loss function for all outputs
            self._add_single_loss_selection()
        else:
            # Different loss function per output
            self._add_multiple_loss_selection(num_outputs, output_names)
    
    def _add_single_loss_selection(self):
        """Add single loss function selection."""
        loss_options = self._get_loss_function_options()
        
        self.addChild({
            'name': 'Loss Selection',
            'type': 'group',
            'children': [
                {'name': 'selected_loss', 'type': 'list', 'limits': loss_options, 'value': 'Categorical Crossentropy', 'tip': 'Select the loss function to use'},
                {'name': 'loss_weight', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': 'Weight for this loss function'}
            ],
            'tip': 'Single loss function applied to all model outputs'
        })
        
        # Add loss function parameters
        self._add_selected_loss_parameters('Loss Selection')
    
    def _add_multiple_loss_selection(self, num_outputs, output_names):
        """Add multiple loss function selections for different outputs."""
        loss_options = self._get_loss_function_options()
        
        for i in range(num_outputs):
            output_name = output_names[i] if i < len(output_names) else f'output_{i+1}'
            
            self.addChild({
                'name': f'Output {i+1}: {output_name}',
                'type': 'group',
                'children': [
                    {'name': 'selected_loss', 'type': 'list', 'limits': loss_options, 'value': 'Categorical Crossentropy', 'tip': f'Select loss function for {output_name}'},
                    {'name': 'loss_weight', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': f'Weight for {output_name} loss function'}
                ],
                'tip': f'Loss function configuration for output: {output_name}'
            })
            
            # Add loss function parameters for this output
            self._add_selected_loss_parameters(f'Output {i+1}: {output_name}')
    
    def _get_loss_function_options(self):
        """Get list of available loss function names including custom ones."""
        base_options = [
            'Categorical Crossentropy',
            'Sparse Categorical Crossentropy', 
            'Binary Crossentropy',
            'Mean Squared Error',
            'Mean Absolute Error',
            'Focal Loss',
            'Huber Loss'
        ]
        
        # Add custom loss functions if any
        if hasattr(self, '_custom_loss_functions'):
            custom_options = list(self._custom_loss_functions.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_selected_loss_parameters(self, parent_name):
        """Add parameters for the selected loss function."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        # Connect selection change to parameter update
        if parent.child('selected_loss'):
            parent.child('selected_loss').sigValueChanged.connect(
                lambda: self._update_loss_parameters(parent_name)
            )
        
        # Add initial parameters
        self._update_loss_parameters(parent_name)
    
    def _update_loss_parameters(self, parent_name):
        """Update loss function parameters based on selection."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        selected_loss = parent.child('selected_loss').value()
        
        # Remove existing parameters (except selected_loss and loss_weight)
        existing_params = []
        for child in parent.children():
            if child.name() not in ['selected_loss', 'loss_weight']:
                existing_params.append(child)
        
        for param in existing_params:
            parent.removeChild(param)
        
        # Add parameters based on selected loss function
        loss_params = self._get_loss_function_parameters(selected_loss)
        for param_config in loss_params:
            parent.addChild(param_config)
    
    def _get_loss_function_parameters(self, loss_name):
        """Get parameters for a specific loss function."""
        # Check if it's a custom loss function
        if hasattr(self, '_custom_loss_parameters') and loss_name in self._custom_loss_parameters:
            return self._custom_loss_parameters[loss_name]
        
        # Return built-in loss function parameters
        loss_parameters = {
            'Categorical Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'label_smoothing', 'type': 'float', 'value': 0.0, 'limits': (0.0, 0.5), 'tip': 'Label smoothing factor'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Sparse Categorical Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Binary Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'label_smoothing', 'type': 'float', 'value': 0.0, 'limits': (0.0, 0.5), 'tip': 'Label smoothing factor'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Mean Squared Error': [
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Mean Absolute Error': [
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Focal Loss': [
                {'name': 'alpha', 'type': 'float', 'value': 0.25, 'limits': (0.0, 1.0), 'tip': 'Weighting factor for rare class'},
                {'name': 'gamma', 'type': 'float', 'value': 2.0, 'limits': (0.0, 5.0), 'tip': 'Focusing parameter'},
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Huber Loss': [
                {'name': 'delta', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': 'Threshold at which to change between MSE and MAE'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ]
        }
        
        return loss_parameters.get(loss_name, [])
    
    def _add_preset_loss_functions(self):
        """Add preset loss functions with their parameters - DEPRECATED."""
        # This method is now deprecated as we use selection-based approach
        pass
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom loss functions from files."""
        self.addChild({
            'name': 'Load Custom Loss Functions',
            'type': 'action',
            'tip': 'Click to load custom loss functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Loss Functions')
        custom_button.sigActivated.connect(self._load_custom_loss_functions)
    
    def _load_custom_loss_functions(self):
        """Load custom loss functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom loss functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_loss_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid loss functions found in the selected file.\n\n"
                    "Functions should accept 'y_true' and 'y_pred' parameters and return loss value."
                )
                return
            
            # Add custom functions to the available loss options
            for func_name, func_info in custom_functions.items():
                self._add_custom_loss_option(func_name, func_info)
            
            # Update all loss selection dropdowns
            self._update_all_loss_selections()
            
            QMessageBox.information(
                None,
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom loss function(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese functions are now available in the loss selection dropdowns."
            )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom loss functions from file:\n{str(e)}"
            )
    
    def _add_custom_loss_option(self, func_name, func_info):
        """Add a custom loss function as an option in dropdowns."""
        # Store custom loss function info for later use
        if not hasattr(self, '_custom_loss_functions'):
            self._custom_loss_functions = {}
        
        display_name = f"{func_name} (custom)"
        self._custom_loss_functions[display_name] = func_info
        
        # Add parameters for this custom loss function
        params = []
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            # Add limits for numeric types
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            elif param_info['type'] == 'list' and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            params.append(param_config)
        
        # Add metadata parameters
        params.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'loss_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of loss (function or class)'}
        ])
        
        # Store parameters for this custom loss function
        if not hasattr(self, '_custom_loss_parameters'):
            self._custom_loss_parameters = {}
        self._custom_loss_parameters[display_name] = params
    
    def _update_all_loss_selections(self):
        """Update all loss selection dropdowns with custom functions."""
        # Get updated loss function options
        loss_options = self._get_loss_function_options()
        
        # Find all loss selection parameters and update their options
        for child in self.children():
            if child.name().startswith('Loss Selection') or child.name().startswith('Output'):
                selected_loss_param = child.child('selected_loss')
                if selected_loss_param:
                    # Update the limits (available options)
                    selected_loss_param.setLimits(loss_options)

    def _extract_loss_functions(self, file_path):
        """Extract valid loss functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid loss function
                    if self._is_valid_loss_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom loss function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a valid loss class
                    if self._is_valid_loss_class(node):
                        # Extract class parameters from __init__ method
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom loss class: {class_name}"
                        
                        custom_functions[class_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': class_name,
                            'type': 'class'
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_loss_function(self, func_node):
        """Check if a function is a valid loss function."""
        # Check if function has at least two parameters (should be 'y_true', 'y_pred')
        if len(func_node.args.args) < 2:
            return False
        
        # Check if parameters are likely loss function parameters
        param_names = [arg.arg for arg in func_node.args.args]
        
        # Common loss function parameter names
        valid_patterns = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits'],
            ['ground_truth', 'predictions']
        ]
        
        for pattern in valid_patterns:
            if all(any(p in param.lower() for p in pattern) for param in param_names[:2]):
                return True
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return
    
    def _is_valid_loss_class(self, class_node):
        """Check if a class is a valid loss class."""
        class_name = class_node.name.lower()
        
        # Check class name for loss indicators
        if 'loss' in class_name:
            return True
        
        # Check if class has call method (indicating it's callable)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__call__':
                return True
        
        return False
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
                    param_name = arg.arg
                    
                    # Try to infer parameter type and default values
                    param_info = {
                        'name': param_name,
                        'type': 'float',  # Default type
                        'default': 1.0,   # Default value
                        'limits': (0.0, 10.0),
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference based on parameter name
                    if 'alpha' in param_name.lower() or 'weight' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 10.0)})
                    elif 'gamma' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 2.0, 'limits': (0.0, 5.0)})
                    elif 'delta' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
                    elif 'reduction' in param_name.lower():
                        param_info.update({'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'default': 'sum_over_batch_size'})
                    elif 'from_logits' in param_name.lower():
                        param_info.update({'type': 'bool', 'default': False})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'y_true', 'y_pred' parameters)."""
        params = []
        
        # Skip the first two parameters (y_true, y_pred) and extract others
        for arg in func_node.args.args[2:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 1.0,   # Default value
                'limits': (0.0, 10.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'alpha' in param_name.lower() or 'weight' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 10.0)})
            elif 'gamma' in param_name.lower():
                param_info.update({'type': 'float', 'default': 2.0, 'limits': (0.0, 5.0)})
            elif 'delta' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
            elif 'reduction' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'sum_over_batch_size'})
            elif 'from_logits' in param_name.lower():
                param_info.update({'type': 'bool', 'default': False})
            elif 'smooth' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.0, 'limits': (0.0, 0.5)})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 2  # -2 to skip y_true, y_pred
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    def set_loss_config(self, config):
        """Set the loss function configuration from loaded config data."""
        if not config or not isinstance(config, dict):
            return
        
        try:
            # Set Model Output Configuration
            output_config = config.get('Model Output Configuration', {})
            if output_config:
                model_output_group = self.child('Model Output Configuration')
                if model_output_group:
                    for param_name, param_value in output_config.items():
                        param = model_output_group.child(param_name)
                        if param:
                            try:
                                param.setValue(param_value)
                            except Exception as e:
                                print(f"Warning: Could not set output config parameter '{param_name}' to '{param_value}': {e}")
                    
                    # Update loss selection after setting output config
                    self._update_loss_selection()
            
            # Set Loss Selection configuration
            loss_selection_config = config.get('Loss Selection', {})
            if loss_selection_config:
                loss_selection_group = self.child('Loss Selection')
                if loss_selection_group:
                    # Set selected loss if available in options
                    selected_loss = loss_selection_config.get('selected_loss')
                    if selected_loss:
                        loss_selector = loss_selection_group.child('selected_loss')
                        if loss_selector:
                            # Check if the selected loss is in the available options
                            available_options = loss_selector.opts['limits']
                            if selected_loss in available_options:
                                loss_selector.setValue(selected_loss)
                                # Update parameters after setting the value
                                self._update_loss_parameters('Loss Selection')
                            else:
                                print(f"Warning: Selected loss '{selected_loss}' not found in available options: {available_options}")
                        else:
                            print("Warning: loss_selector parameter not found")
                    
                    # Set parameter values
                    for param_name, param_value in loss_selection_config.items():
                        if param_name not in ['selected_loss']:
                            param = loss_selection_group.child(param_name)
                            if param:
                                try:
                                    param.setValue(param_value)
                                except Exception as e:
                                    print(f"Warning: Could not set loss parameter '{param_name}' to '{param_value}': {e}")
            
            # Handle multiple outputs if they exist
            for child in self.children():
                if child.name().startswith('Output') and 'Loss' in child.name():
                    # This is a per-output loss configuration
                    output_config_data = config.get(child.name(), {})
                    if output_config_data:
                        # Set selected loss
                        selected_loss = output_config_data.get('selected_loss')
                        if selected_loss:
                            loss_selector = child.child('selected_loss')
                            if loss_selector:
                                available_options = loss_selector.opts['limits']
                                if selected_loss in available_options:
                                    loss_selector.setValue(selected_loss)
                                    self._update_loss_parameters(child.name())
                                else:
                                    print(f"Warning: Selected loss '{selected_loss}' not found for {child.name()}")
                        
                        # Set other parameters
                        for param_name, param_value in output_config_data.items():
                            if param_name not in ['selected_loss']:
                                param = child.child(param_name)
                                if param:
                                    try:
                                        param.setValue(param_value)
                                    except Exception as e:
                                        print(f"Warning: Could not set {child.name()} parameter '{param_name}': {e}")
                        
        except Exception as e:
            print(f"Error setting loss function configuration: {e}")
            import traceback
            traceback.print_exc()
    
    def load_custom_loss_from_metadata(self, loss_info):
        """Load custom loss function from metadata info."""
        try:
            file_path = loss_info.get('file_path', '')
            function_name = loss_info.get('function_name', '')
            loss_type = loss_info.get('type', 'function')
            
            if not os.path.exists(file_path):
                print(f"Warning: Custom loss function file not found: {file_path}")
                return False
            
            # Extract loss functions from the file
            custom_functions = self._extract_loss_functions(file_path)
            
            # Find the specific function we need
            target_function = None
            for func_name, func_info in custom_functions.items():
                if func_info['function_name'] == function_name:
                    target_function = func_info
                    break
            
            if not target_function:
                print(f"Warning: Function '{function_name}' not found in {file_path}")
                return False
            
            # Add the custom loss function
            self._add_custom_loss_option(function_name, target_function)
            
            # Update all loss selection dropdowns
            self._update_all_loss_selections()
            
            return True
            
        except Exception as e:
            print(f"Error loading custom loss function from metadata: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

