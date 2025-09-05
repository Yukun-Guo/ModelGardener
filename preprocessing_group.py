import ast
import os
import importlib.util

# CLI-only message functions (no GUI dialogs)
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

# Custom preprocessing group that includes preset methods and allows adding custom methods from files

class PreprocessingGroup:
    def __init__(self, **opts):
        self.config = {
            'Preprocessing Options': {
                'apply_normalization': True,
                'apply_resizing': True,
                'apply_custom': False,
                'target_size': [224, 224],
                'normalization_method': 'standard'
            },
            'Preprocessing Chain': {
                'resize': {
                    'enabled': True,
                    'target_size': [224, 224],
                    'interpolation': 'bilinear'
                },
                'normalize': {
                    'enabled': True,
                    'method': 'standard',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                },
                'augment': {
                    'enabled': False,
                    'rotation_range': 0.0,
                    'width_shift_range': 0.0,
                    'height_shift_range': 0.0,
                    'horizontal_flip': False
                }
            },
            'Custom Parameters': {}
        }
        
        # Initialize custom preprocessing storage
        self._custom_preprocessing_functions = {}
        self._custom_preprocessing_parameters = {}
        
    def get_config(self):
        """Get the current preprocessing configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the preprocessing configuration."""
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
    
    def _get_preprocessing_options(self):
        """Get list of available preprocessing methods including custom ones."""
        base_options = [
            'resize',
            'normalize',
            'standardize',
            'min_max_scale',
            'center_crop',
            'random_crop',
            'rotate',
            'flip',
            'brightness_adjust',
            'contrast_adjust',
            'saturation_adjust',
            'hue_adjust'
        ]
        
        # Add custom preprocessing functions if any
        if hasattr(self, '_custom_preprocessing_functions'):
            custom_options = list(self._custom_preprocessing_functions.keys())
            return base_options + custom_options
        
        return base_options
    
    def load_custom_preprocessing(self, file_path):
        """Load custom preprocessing functions from a file."""
        try:
            custom_functions = self._extract_preprocessing_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid preprocessing functions found in the selected file.\n\n"
                    "Functions should accept image/data arrays and return processed arrays."
                )
                return False
            
            # Add custom functions to the available preprocessing options
            for func_name, func_info in custom_functions.items():
                self._add_custom_preprocessing_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom preprocessing function(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese functions are now available in the preprocessing chain."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom preprocessing functions from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_preprocessing_option(self, func_name, func_info):
        """Add a custom preprocessing function as an option."""
        # Store the function with metadata
        self._custom_preprocessing_functions[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_preprocessing_parameters[func_name] = func_info['parameters']
        
        # Add to preprocessing chain with default parameters
        self.config['Preprocessing Chain'][func_name] = {
            'enabled': False
        }
        
        # Add function-specific parameters
        if 'parameters' in func_info:
            for param in func_info['parameters']:
                self.config['Preprocessing Chain'][func_name][param['name']] = param.get('default', 0.0)
    
    def _extract_preprocessing_functions(self, file_path):
        """Extract preprocessing function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely a preprocessing function
                    if self._is_preprocessing_function(node):
                        func_info = self._analyze_preprocessing_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's a preprocessing class
                    if self._is_preprocessing_class(node):
                        class_info = self._analyze_preprocessing_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_preprocessing_function(self, node):
        """Check if a function node is likely a preprocessing function."""
        # Check function name for preprocessing indicators
        preprocessing_keywords = ['preprocess', 'normalize', 'resize', 'augment', 'transform', 'process']
        has_preprocessing_name = any(keyword in node.name.lower() for keyword in preprocessing_keywords)
        
        # Check if function has common preprocessing parameters
        args = [arg.arg for arg in node.args.args]
        preprocessing_args = ['image', 'data', 'array', 'x', 'input', 'tensor']
        has_preprocessing_args = any(arg in args for arg in preprocessing_args)
        
        return has_preprocessing_name or has_preprocessing_args
    
    def _is_preprocessing_class(self, node):
        """Check if a class node is likely a preprocessing class."""
        # Check class name for preprocessing indicators
        preprocessing_keywords = ['preprocess', 'transform', 'augment', 'normalize']
        has_preprocessing_name = any(keyword in node.name.lower() for keyword in preprocessing_keywords)
        
        # Check if class has __call__ method (callable preprocessing)
        has_call = any(isinstance(child, ast.FunctionDef) and child.name == '__call__' 
                      for child in node.body)
        
        return has_preprocessing_name or has_call
    
    def _analyze_preprocessing_function(self, node, file_path):
        """Analyze a preprocessing function and extract its metadata."""
        try:
            # Extract parameters (skip the first parameter which is usually the input data)
            parameters = []
            defaults = node.args.defaults
            default_values = [None] * (len(node.args.args) - len(defaults)) + defaults
            
            for i, arg in enumerate(node.args.args[1:], 1):  # Skip first parameter
                param_info = {
                    'name': arg.arg,
                    'type': self._infer_parameter_type(arg.arg),
                    'default': self._extract_default_value(default_values[i])
                }
                parameters.append(param_info)
            
            # Extract docstring
            docstring = ast.get_docstring(node) or f"Custom preprocessing function: {node.name}"
            
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
    
    def _analyze_preprocessing_class(self, node, file_path):
        """Analyze a preprocessing class and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom preprocessing class: {node.name}"
            
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
        
        if any(word in param_name_lower for word in ['size', 'width', 'height', 'steps', 'num', 'count']):
            return 'int'
        elif any(word in param_name_lower for word in ['rate', 'factor', 'scale', 'range', 'mean', 'std', 'alpha']):
            return 'float'
        elif any(word in param_name_lower for word in ['flip', 'enable', 'use', 'apply']):
            return 'bool'
        elif any(word in param_name_lower for word in ['method', 'mode', 'interpolation', 'format']):
            return 'str'
        else:
            return 'float'  # Default to float
    
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
    
    def get_preprocessing_config(self):
        """Get the current preprocessing configuration for training."""
        config = {}
        
        # Get enabled preprocessing steps
        preprocessing_chain = []
        for method_name, method_config in self.config['Preprocessing Chain'].items():
            if method_config.get('enabled', False):
                step_config = {
                    'method': method_name,
                    'parameters': {k: v for k, v in method_config.items() if k != 'enabled'}
                }
                preprocessing_chain.append(step_config)
        
        config['preprocessing_chain'] = preprocessing_chain
        config['options'] = dict(self.config['Preprocessing Options'])
        
        # Add custom preprocessing functions if any
        if preprocessing_chain:
            custom_methods = []
            for step in preprocessing_chain:
                if step['method'] in self._custom_preprocessing_functions:
                    custom_methods.append({
                        'method': step['method'],
                        'function_info': self._custom_preprocessing_functions[step['method']],
                        'parameters': step['parameters']
                    })
            
            if custom_methods:
                config['custom_preprocessing'] = custom_methods
        
        return config
    
    def load_custom_preprocessing_from_metadata(self, metadata):
        """Load a custom preprocessing function from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_preprocessing", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_preprocessing_functions[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom preprocessing function: {str(e)}")
            return False
