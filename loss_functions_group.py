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

# Custom loss functions group that includes preset loss functions and allows adding custom loss functions from files

class LossFunctionsGroup:
    def __init__(self, **opts):
        self.config = {
            'Model Output Configuration': {
                'num_outputs': 1,
                'output_names': 'main_output',
                'loss_strategy': 'single_loss_all_outputs'
            },
            'Loss Selection': {
                'selected_loss': 'categorical_crossentropy',
                'loss_weight': 1.0,
                'from_logits': False
            },
            'Custom Parameters': {}
        }
        self._custom_loss_functions = {}
        self._custom_loss_parameters = {}
        
    def get_config(self):
        """Get the current loss functions configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the loss functions configuration."""
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
        if path.startswith('Model Output Configuration'):
            self._update_loss_selection()
    
    def _update_loss_selection(self):
        """Update loss function selection based on output configuration."""
        num_outputs = self.config['Model Output Configuration']['num_outputs']
        loss_strategy = self.config['Model Output Configuration']['loss_strategy']
        output_names = self.config['Model Output Configuration']['output_names'].split(',')
        output_names = [name.strip() for name in output_names if name.strip()]
        
        if num_outputs == 1 or loss_strategy == 'single_loss_all_outputs':
            # Single loss function for all outputs
            if 'Loss Selection' not in self.config:
                self.config['Loss Selection'] = {
                    'selected_loss': 'categorical_crossentropy',
                    'loss_weight': 1.0,
                    'from_logits': False
                }
        else:
            # Different loss function per output
            for i in range(num_outputs):
                output_name = output_names[i] if i < len(output_names) else f'output_{i+1}'
                key = f'Output {i+1}: {output_name}'
                if key not in self.config:
                    self.config[key] = {
                        'selected_loss': 'categorical_crossentropy',
                        'loss_weight': 1.0,
                        'from_logits': False
                    }
        
    def _get_loss_function_options(self):
        """Get list of available loss function names including custom ones."""
        base_options = [
            'categorical_crossentropy',
            'sparse_categorical_crossentropy', 
            'binary_crossentropy',
            'mean_squared_error',
            'mean_absolute_error',
            'focal_loss',
            'huber_loss',
            'log_cosh',
            'poisson',
            'kl_divergence',
            'cosine_similarity'
        ]
        
        # Add custom loss functions if any
        if hasattr(self, '_custom_loss_functions'):
            custom_options = list(self._custom_loss_functions.keys())
            return base_options + custom_options
        
        return base_options
        
    def load_custom_loss_functions(self, file_path):
        """Load custom loss functions from a file."""
        try:
            custom_functions = self._extract_loss_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid loss functions found in the selected file.\n\n"
                    "Functions should accept 'y_true' and 'y_pred' parameters or be TensorFlow loss classes."
                )
                return False
            
            # Add custom functions to the available loss options
            for func_name, func_info in custom_functions.items():
                self._add_custom_loss_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom loss function(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese functions are now available in the selection dropdowns."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom loss functions from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_loss_option(self, func_name, func_info):
        """Add a custom loss function as an option."""
        # Store the function with metadata
        self._custom_loss_functions[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_loss_parameters[func_name] = func_info['parameters']
    
    def _extract_loss_functions(self, file_path):
        """Extract loss function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely a loss function
                    if self._is_loss_function(node):
                        func_info = self._analyze_loss_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's a loss function class
                    if self._is_loss_class(node):
                        class_info = self._analyze_loss_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_loss_function(self, node):
        """Check if a function node is likely a loss function."""
        # Check function signature for common loss function parameters
        args = [arg.arg for arg in node.args.args]
        
        # Common loss function signatures
        loss_signatures = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits']
        ]
        
        for signature in loss_signatures:
            if all(arg in args for arg in signature):
                return True
        
        # Check if function name suggests it's a loss function
        loss_keywords = ['loss', 'error', 'crossentropy', 'mse', 'mae', 'focal']
        return any(keyword in node.name.lower() for keyword in loss_keywords)
    
    def _is_loss_class(self, node):
        """Check if a class node is likely a loss function class."""
        # Check if class has call method (callable loss)
        has_call = any(isinstance(child, ast.FunctionDef) and child.name == '__call__' 
                      for child in node.body)
        
        # Check class name for loss indicators
        loss_keywords = ['loss', 'error', 'crossentropy']
        has_loss_name = any(keyword in node.name.lower() for keyword in loss_keywords)
        
        return has_call or has_loss_name
    
    def _analyze_loss_function(self, node, file_path):
        """Analyze a loss function and extract its metadata."""
        try:
            # Extract parameters
            parameters = []
            defaults = node.args.defaults
            default_values = [None] * (len(node.args.args) - len(defaults)) + defaults
            
            for i, arg in enumerate(node.args.args):
                if arg.arg not in ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction', 'labels', 'logits']:
                    param_info = {
                        'name': arg.arg,
                        'type': 'float',  # Default type
                        'default': default_values[i].n if isinstance(default_values[i], ast.Num) else None
                    }
                    parameters.append(param_info)
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Custom loss function: {node.name}"
            
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
    
    def _analyze_loss_class(self, node, file_path):
        """Analyze a loss class and extract its metadata."""
        try:
            # Find __init__ method to extract parameters
            init_method = None
            call_method = None
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    if child.name == '__init__':
                        init_method = child
                    elif child.name == '__call__':
                        call_method = child
            
            parameters = []
            if init_method:
                defaults = init_method.args.defaults
                default_values = [None] * (len(init_method.args.args) - len(defaults)) + defaults
                
                for i, arg in enumerate(init_method.args.args):
                    if arg.arg not in ['self']:
                        param_info = {
                            'name': arg.arg,
                            'type': 'float',  # Default type
                            'default': default_values[i].n if isinstance(default_values[i], ast.Num) else None
                        }
                        parameters.append(param_info)
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Custom loss class: {node.name}"
            
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
    
    def get_loss_config(self):
        """Get the current loss configuration for training."""
        config = {}
        
        num_outputs = self.config['Model Output Configuration']['num_outputs']
        loss_strategy = self.config['Model Output Configuration']['loss_strategy']
        
        if num_outputs == 1 or loss_strategy == 'single_loss_all_outputs':
            # Single loss for all outputs
            loss_info = self.config['Loss Selection']
            config['loss'] = self._get_loss_implementation(loss_info['selected_loss'])
            config['loss_weights'] = loss_info['loss_weight']
        else:
            # Multiple losses for different outputs
            losses = {}
            loss_weights = {}
            output_names = self.config['Model Output Configuration']['output_names'].split(',')
            output_names = [name.strip() for name in output_names if name.strip()]
            
            for i in range(num_outputs):
                output_name = output_names[i] if i < len(output_names) else f'output_{i+1}'
                key = f'Output {i+1}: {output_name}'
                
                if key in self.config:
                    loss_info = self.config[key]
                    losses[output_name] = self._get_loss_implementation(loss_info['selected_loss'])
                    loss_weights[output_name] = loss_info['loss_weight']
            
            config['loss'] = losses
            config['loss_weights'] = loss_weights
        
        return config
    
    def _get_loss_implementation(self, loss_name):
        """Get the actual loss function implementation."""
        # Map loss names to TensorFlow implementations
        loss_mapping = {
            'categorical_crossentropy': 'categorical_crossentropy',
            'sparse_categorical_crossentropy': 'sparse_categorical_crossentropy',
            'binary_crossentropy': 'binary_crossentropy',
            'mean_squared_error': 'mean_squared_error',
            'mean_absolute_error': 'mean_absolute_error',
            'focal_loss': 'categorical_focal_crossentropy',
            'huber_loss': 'huber',
            'log_cosh': 'log_cosh',
            'poisson': 'poisson',
            'kl_divergence': 'kl_divergence',
            'cosine_similarity': 'cosine_similarity'
        }
        
        if loss_name in loss_mapping:
            return loss_mapping[loss_name]
        elif loss_name in self._custom_loss_functions:
            # Return custom loss function info
            return self._custom_loss_functions[loss_name]
        else:
            # Default fallback
            return 'categorical_crossentropy'
    
    def load_custom_loss_function_from_metadata(self, metadata):
        """Load a custom loss function from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_loss", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_loss_functions[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom loss function: {str(e)}")
            return False
