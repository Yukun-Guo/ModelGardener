import ast
import os
import importlib.util

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

# Custom callbacks group that includes preset callbacks and allows adding custom callbacks from files

class CallbacksGroup:
    def __init__(self, **opts):
        self.config = {
            'Callback Selection': {
                'selected_callbacks': ['ModelCheckpoint', 'EarlyStopping'],
                'enable_tensorboard': True,
                'enable_csv_logger': True
            },
            'ModelCheckpoint': {
                'enabled': True,
                'filepath': 'model_checkpoint.h5',
                'monitor': 'val_loss',
                'verbose': 1,
                'save_best_only': True,
                'save_weights_only': False,
                'mode': 'min',
                'save_freq': 'epoch'
            },
            'EarlyStopping': {
                'enabled': True,
                'monitor': 'val_loss',
                'min_delta': 0.001,
                'patience': 10,
                'verbose': 1,
                'mode': 'min',
                'restore_best_weights': True
            },
            'ReduceLROnPlateau': {
                'enabled': False,
                'monitor': 'val_loss',
                'factor': 0.2,
                'patience': 5,
                'min_lr': 1e-7,
                'verbose': 1,
                'mode': 'min'
            },
            'TensorBoard': {
                'enabled': True,
                'log_dir': './logs',
                'histogram_freq': 1,
                'write_graph': True,
                'write_images': True,
                'update_freq': 'epoch'
            },
            'CSVLogger': {
                'enabled': True,
                'filename': 'training_log.csv',
                'separator': ',',
                'append': False
            },
            'Custom Parameters': {}
        }
        
        # Initialize custom callbacks storage
        self._custom_callbacks = {}
        self._custom_callback_parameters = {}
        
    def get_config(self):
        """Get the current callbacks configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the callbacks configuration."""
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
    
    def _get_callback_options(self):
        """Get list of available callback names including custom ones."""
        base_options = [
            'ModelCheckpoint',
            'EarlyStopping',
            'ReduceLROnPlateau',
            'TensorBoard',
            'CSVLogger',
            'LearningRateScheduler',
            'ProgbarLogger',
            'History',
            'BaseLogger'
        ]
        
        # Add custom callbacks if any
        if hasattr(self, '_custom_callbacks'):
            custom_options = list(self._custom_callbacks.keys())
            return base_options + custom_options
        
        return base_options
    
    def load_custom_callbacks(self, file_path):
        """Load custom callback functions from a file."""
        try:
            custom_functions = self._extract_callback_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid callback functions found in the selected file.\n\n"
                    "Functions should be TensorFlow/Keras callback classes or functions that return callbacks."
                )
                return False
            
            # Add custom functions to the available callback options
            for func_name, func_info in custom_functions.items():
                self._add_custom_callback_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom callback(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese callbacks are now available for selection."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom callbacks from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_callback_option(self, func_name, func_info):
        """Add a custom callback as an option."""
        # Store the function with metadata
        self._custom_callbacks[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_callback_parameters[func_name] = func_info['parameters']
        
        # Add callback configuration section
        self.config[func_name] = {
            'enabled': False
        }
        
        # Add callback-specific parameters
        if 'parameters' in func_info:
            for param in func_info['parameters']:
                self.config[func_name][param['name']] = param.get('default', None)
    
    def _extract_callback_functions(self, file_path):
        """Extract callback function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely a callback function
                    if self._is_callback_function(node):
                        func_info = self._analyze_callback_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's a callback class
                    if self._is_callback_class(node):
                        class_info = self._analyze_callback_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_callback_function(self, node):
        """Check if a function node is likely a callback function."""
        # Check function name for callback indicators
        callback_keywords = ['callback', 'monitor', 'checkpoint', 'logger', 'scheduler']
        has_callback_name = any(keyword in node.name.lower() for keyword in callback_keywords)
        
        # Check if function returns a callback or has callback-like signature
        return has_callback_name
    
    def _is_callback_class(self, node):
        """Check if a class node is likely a callback class."""
        # Check class name for callback indicators
        callback_keywords = ['callback', 'monitor', 'checkpoint', 'logger', 'scheduler']
        has_callback_name = any(keyword in node.name.lower() for keyword in callback_keywords)
        
        # Check if class has callback methods
        callback_methods = ['on_epoch_begin', 'on_epoch_end', 'on_batch_begin', 'on_batch_end', 
                          'on_train_begin', 'on_train_end']
        has_callback_methods = any(isinstance(child, ast.FunctionDef) and child.name in callback_methods 
                                 for child in node.body)
        
        return has_callback_name or has_callback_methods
    
    def _analyze_callback_function(self, node, file_path):
        """Analyze a callback function and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom callback function: {node.name}"
            
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
    
    def _analyze_callback_class(self, node, file_path):
        """Analyze a callback class and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom callback class: {node.name}"
            
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
        
        if any(word in param_name_lower for word in ['patience', 'freq', 'verbose', 'steps']):
            return 'int'
        elif any(word in param_name_lower for word in ['delta', 'factor', 'lr', 'rate']):
            return 'float'
        elif any(word in param_name_lower for word in ['save', 'restore', 'enable', 'append', 'write']):
            return 'bool'
        elif any(word in param_name_lower for word in ['path', 'file', 'dir', 'monitor', 'mode']):
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
    
    def get_callbacks_config(self):
        """Get the current callbacks configuration for training."""
        config = {
            'callbacks': [],
            'callback_configs': {}
        }
        
        # Get enabled callbacks
        for callback_name, callback_config in self.config.items():
            if callback_name in ['Callback Selection', 'Custom Parameters']:
                continue
                
            if callback_config.get('enabled', False):
                config['callbacks'].append(callback_name)
                config['callback_configs'][callback_name] = {
                    k: v for k, v in callback_config.items() if k != 'enabled'
                }
        
        # Add custom callbacks if any
        custom_callbacks = []
        for callback_name in config['callbacks']:
            if callback_name in self._custom_callbacks:
                custom_callbacks.append({
                    'name': callback_name,
                    'function_info': self._custom_callbacks[callback_name],
                    'config': config['callback_configs'][callback_name]
                })
        
        if custom_callbacks:
            config['custom_callbacks'] = custom_callbacks
        
        return config
    
    def load_custom_callback_from_metadata(self, metadata):
        """Load a custom callback from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_callback", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_callbacks[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom callback: {str(e)}")
            return False
