import ast
import os
import importlib.util
import pyqtgraph.parametertree.parameterTypes as pTypes
from PySide6.QtWidgets import QFileDialog, QMessageBox

# Custom callbacks group that includes preset callbacks and allows adding custom callbacks from files 

class CallbacksGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset callbacks
        self._add_preset_callbacks()
        
        # Add custom callbacks button
        self._add_custom_button()
    
    def _add_preset_callbacks(self):
        """Add preset callback methods with their parameters."""
        preset_callbacks = [
            {
                'name': 'Early Stopping',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable early stopping'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'patience', 'type': 'int', 'value': 10, 'limits': (1, 100), 'tip': 'Number of epochs with no improvement to wait'},
                    {'name': 'min_delta', 'type': 'float', 'value': 0.001, 'limits': (0.0, 1.0), 'tip': 'Minimum change to qualify as improvement'},
                    {'name': 'mode', 'type': 'list', 'limits': ['min', 'max', 'auto'], 'value': 'min', 'tip': 'Direction of improvement'},
                    {'name': 'restore_best_weights', 'type': 'bool', 'value': True, 'tip': 'Restore model weights from best epoch'}
                ],
                'tip': 'Stop training when monitored metric stops improving'
            },
            {
                'name': 'Learning Rate Scheduler',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable learning rate scheduling'},
                    {'name': 'scheduler_type', 'type': 'list', 'limits': ['ReduceLROnPlateau', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR'], 'value': 'ReduceLROnPlateau', 'tip': 'Type of learning rate scheduler'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'factor', 'type': 'float', 'value': 0.5, 'limits': (0.01, 1.0), 'tip': 'Factor by which learning rate is reduced'},
                    {'name': 'patience', 'type': 'int', 'value': 5, 'limits': (1, 50), 'tip': 'Number of epochs with no improvement to wait'},
                    {'name': 'min_lr', 'type': 'float', 'value': 1e-7, 'limits': (1e-10, 1e-2), 'tip': 'Minimum learning rate'},
                    {'name': 'step_size', 'type': 'int', 'value': 30, 'limits': (1, 1000), 'tip': 'Period of learning rate decay (for StepLR)'},
                    {'name': 'gamma', 'type': 'float', 'value': 0.1, 'limits': (0.01, 1.0), 'tip': 'Multiplicative factor of learning rate decay'}
                ],
                'tip': 'Adjust learning rate during training based on metrics or schedule'
            },
            {
                'name': 'Model Checkpoint',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable model checkpointing'},
                    {'name': 'filepath', 'type': 'str', 'value': './checkpoints/model-{epoch:02d}-{val_loss:.2f}.keras', 'tip': 'Path template for checkpoint files'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'save_best_only', 'type': 'bool', 'value': True, 'tip': 'Save only the best model'},
                    {'name': 'save_weights_only', 'type': 'bool', 'value': False, 'tip': 'Save only model weights (not full model)'},
                    {'name': 'mode', 'type': 'list', 'limits': ['min', 'max', 'auto'], 'value': 'min', 'tip': 'Direction of improvement'},
                    {'name': 'period', 'type': 'int', 'value': 1, 'limits': (1, 100), 'tip': 'Interval between checkpoints'}
                ],
                'tip': 'Save model checkpoints during training'
            },
            {
                'name': 'CSV Logger',
                'type': 'group', 
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable CSV logging'},
                    {'name': 'filename', 'type': 'str', 'value': './logs/training_log.csv', 'tip': 'Path to CSV log file'},
                    {'name': 'separator', 'type': 'str', 'value': ',', 'tip': 'Delimiter for CSV file'},
                    {'name': 'append', 'type': 'bool', 'value': False, 'tip': 'Append to existing file or create new'}
                ],
                'tip': 'Log training metrics to CSV file'
            },
            {
                'name': 'TensorBoard',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable TensorBoard logging'},
                    {'name': 'log_dir', 'type': 'str', 'value': './logs/tensorboard', 'tip': 'Directory for TensorBoard logs'},
                    {'name': 'histogram_freq', 'type': 'int', 'value': 1, 'limits': (0, 100), 'tip': 'Frequency for histogram computation'},
                    {'name': 'write_graph', 'type': 'bool', 'value': True, 'tip': 'Write model graph to TensorBoard'},
                    {'name': 'write_images', 'type': 'bool', 'value': False, 'tip': 'Write model weights as images'},
                    {'name': 'update_freq', 'type': 'list', 'limits': ['epoch', 'batch'], 'value': 'epoch', 'tip': 'Update frequency for logging'}
                ],
                'tip': 'Log training metrics and model graph to TensorBoard'
            }
        ]
        
        # Add all preset callbacks
        for callback in preset_callbacks:
            self.addChild(callback)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom callback functions from files."""
        self.addChild({
            'name': 'Load Custom Callbacks',
            'type': 'action',
            'tip': 'Click to load custom callback functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Callbacks')
        custom_button.sigActivated.connect(self._load_custom_callbacks)
    
    def _load_custom_callbacks(self):
        """Load custom callback functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom callback functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_callback_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid callback functions found in the selected file.\n\n"
                    "Functions should inherit from tf.keras.callbacks.Callback or implement callback interface."
                )
                return
            
            # Add each found function as a custom callback
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom callback function(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom callbacks from file:\n{str(e)}"
            )
    
    def _extract_callback_functions(self, file_path):
        """Extract valid callback functions from a Python file."""
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
                    
                    # Check if it's a valid callback function
                    if self._is_valid_callback_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom callback function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a callback class
                    if self._is_valid_callback_class(node):
                        # Extract class init parameters
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom callback class: {class_name}"
                        
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
    
    def _is_valid_callback_function(self, func_node):
        """Check if a function is a valid callback function."""
        # Look for common callback method names or parameters
        func_name = func_node.name.lower()
        callback_indicators = ['callback', 'on_epoch', 'on_batch', 'on_train', 'monitor', 'log']
        
        return any(indicator in func_name for indicator in callback_indicators)
    
    def _is_valid_callback_class(self, class_node):
        """Check if a class is a valid callback class."""
        class_name = class_node.name.lower()
        
        # Check class name for callback indicators
        if 'callback' in class_name:
            return True
        
        # Check if class has callback-like methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_name = node.name.lower()
                if any(method in method_name for method in ['on_epoch', 'on_batch', 'on_train']):
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
                    
                    param_info = {
                        'name': param_name,
                        'type': 'str',  # Default type
                        'default': '',
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference
                    if 'patience' in param_name.lower() or 'epoch' in param_name.lower():
                        param_info.update({'type': 'int', 'default': 10, 'limits': (1, 1000)})
                    elif 'rate' in param_name.lower() or 'factor' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 0.1, 'limits': (0.001, 1.0)})
                    elif 'enable' in param_name.lower():
                        param_info.update({'type': 'bool', 'default': True})
                    elif 'path' in param_name.lower() or 'dir' in param_name.lower():
                        param_info.update({'type': 'str', 'default': './logs'})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition."""
        params = []
        
        # Extract function arguments (skip common callback parameters like 'logs', 'epoch', etc.)
        skip_params = {'self', 'logs', 'epoch', 'batch', 'model'}
        
        for arg in func_node.args.args:
            param_name = arg.arg
            
            if param_name not in skip_params:
                param_info = {
                    'name': param_name,
                    'type': 'str',
                    'default': '',
                    'tip': f'Parameter for {param_name}'
                }
                
                # Basic type inference based on parameter name
                if 'patience' in param_name.lower() or 'step' in param_name.lower():
                    param_info.update({'type': 'int', 'default': 10, 'limits': (1, 1000)})
                elif 'rate' in param_name.lower() or 'threshold' in param_name.lower():
                    param_info.update({'type': 'float', 'default': 0.1, 'limits': (0.001, 1.0)})
                elif 'enable' in param_name.lower():
                    param_info.update({'type': 'bool', 'default': True})
                
                params.append(param_info)
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as a callback method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} callback'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            children.append(param_config)
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'callback_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of callback (function or class)'}
        ])
        
        # Create the callback method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Callbacks" button
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Callbacks':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            self.addChild(method_config)
        
        return True
    
    def set_callbacks_config(self, config):
        """Set the callbacks configuration from loaded config data."""
        if not config or not isinstance(config, dict):
            return
        
        try:
            # Set configuration for each callback
            for callback_name, callback_config in config.items():
                if callback_name == 'Load Custom Callbacks':
                    continue
                    
                callback_group = self.child(callback_name)
                if callback_group and isinstance(callback_config, dict):
                    # Set parameters for this callback
                    for param_name, param_value in callback_config.items():
                        param = callback_group.child(param_name)
                        if param:
                            try:
                                param.setValue(param_value)
                            except Exception as e:
                                print(f"Warning: Could not set callback parameter '{callback_name}.{param_name}' to '{param_value}': {e}")
                        else:
                            # This might be a custom callback parameter - let's try to create it if it doesn't exist
                            if callback_name.startswith('Custom') and 'file_path' in callback_config:
                                # This is likely a custom callback that needs to be loaded first
                                print(f"Note: Custom callback '{callback_name}' parameter '{param_name}' not found - may need to load custom callback first")
                else:
                    print(f"Warning: Callback group '{callback_name}' not found or config is not a dict")
                        
        except Exception as e:
            print(f"Error setting callbacks configuration: {e}")
            import traceback
            traceback.print_exc()
    
    def load_custom_callback_from_metadata(self, callback_info):
        """Load custom callback from metadata info."""
        try:
            file_path = callback_info.get('file_path', '')
            function_name = callback_info.get('function_name', '')
            callback_type = callback_info.get('type', 'function')
            
            if not os.path.exists(file_path):
                print(f"Warning: Custom callback file not found: {file_path}")
                return False
            
            # Extract callbacks from the file
            custom_functions = self._extract_callback_functions(file_path)
            
            # Find the specific function we need
            target_function = None
            for func_name, func_info in custom_functions.items():
                if func_info['function_name'] == function_name:
                    target_function = func_info
                    break
            
            if not target_function:
                print(f"Warning: Function '{function_name}' not found in {file_path}")
                return False
            
            # Add the custom callback function
            self._add_custom_function(function_name, target_function)
            
            return True
            
        except Exception as e:
            print(f"Error loading custom callback from metadata: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_custom_callback_option(self, option_name, callback_info):
        """Add a custom callback option to the callbacks selector."""
        selection_group = self.child('Callback Selection')
        if selection_group:
            callbacks_selector = selection_group.child('selected_callbacks')
            if callbacks_selector:
                # Get current options
                current_options = list(callbacks_selector.opts['limits'])
                
                # Add new option if not already present
                if option_name not in current_options:
                    current_options.append(option_name)
                    callbacks_selector.setLimits(current_options)
                    
                    # Store the callback info for later use
                    if not hasattr(self, '_custom_callback_functions'):
                        self._custom_callback_functions = {}
                    self._custom_callback_functions[option_name] = callback_info
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

