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

# Custom augmentation group that includes preset augmentations and allows adding custom augmentations from files

class AugmentationGroup:
    def __init__(self, **opts):
        self.config = {
            'Augmentation Options': {
                'enable_augmentation': True,
                'augmentation_probability': 0.8,
                'apply_to_validation': False
            },
            'Image Augmentation': {
                'rotation_range': 0.0,
                'width_shift_range': 0.0,
                'height_shift_range': 0.0,
                'shear_range': 0.0,
                'zoom_range': 0.0,
                'horizontal_flip': False,
                'vertical_flip': False,
                'brightness_range': [1.0, 1.0],
                'channel_shift_range': 0.0
            },
            'Advanced Augmentation': {
                'mixup_alpha': 0.0,
                'cutmix_alpha': 0.0,
                'cutout_probability': 0.0,
                'cutout_size': [16, 16],
                'mosaic_probability': 0.0,
                'elastic_transform': False,
                'gaussian_noise': 0.0
            },
            'Custom Parameters': {}
        }
        
        # Initialize custom augmentation storage
        self._custom_augmentations = {}
        self._custom_augmentation_parameters = {}
        
    def get_config(self):
        """Get the current augmentation configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the augmentation configuration."""
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
    
    def _get_augmentation_options(self):
        """Get list of available augmentation methods including custom ones."""
        base_options = [
            'rotation',
            'width_shift',
            'height_shift',
            'shear',
            'zoom',
            'horizontal_flip',
            'vertical_flip',
            'brightness',
            'channel_shift',
            'mixup',
            'cutmix',
            'cutout',
            'mosaic',
            'elastic_transform',
            'gaussian_noise'
        ]
        
        # Add custom augmentations if any
        if hasattr(self, '_custom_augmentations'):
            custom_options = list(self._custom_augmentations.keys())
            return base_options + custom_options
        
        return base_options
    
    def load_custom_augmentations(self, file_path):
        """Load custom augmentation functions from a file."""
        try:
            custom_functions = self._extract_augmentation_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid augmentation functions found in the selected file.\n\n"
                    "Functions should accept image/data arrays and return augmented arrays."
                )
                return False
            
            # Add custom functions to the available augmentation options
            for func_name, func_info in custom_functions.items():
                self._add_custom_augmentation_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom augmentation(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese augmentations are now available for use."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom augmentations from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_augmentation_option(self, func_name, func_info):
        """Add a custom augmentation as an option."""
        # Store the function with metadata
        self._custom_augmentations[func_name] = func_info
        
        # Store custom parameters if any
        if 'parameters' in func_info:
            self._custom_augmentation_parameters[func_name] = func_info['parameters']
        
        # Add augmentation configuration section
        self.config['Custom Parameters'][func_name] = {
            'enabled': False,
            'probability': 0.5
        }
        
        # Add augmentation-specific parameters
        if 'parameters' in func_info:
            for param in func_info['parameters']:
                self.config['Custom Parameters'][func_name][param['name']] = param.get('default', 0.0)
    
    def _extract_augmentation_functions(self, file_path):
        """Extract augmentation function definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's likely an augmentation function
                    if self._is_augmentation_function(node):
                        func_info = self._analyze_augmentation_function(node, file_path)
                        if func_info:
                            functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    # Check if it's an augmentation class
                    if self._is_augmentation_class(node):
                        class_info = self._analyze_augmentation_class(node, file_path)
                        if class_info:
                            functions[node.name] = class_info
            
            return functions
            
        except Exception as e:
            cli_error("Parse Error", f"Error parsing file: {str(e)}")
            return {}
    
    def _is_augmentation_function(self, node):
        """Check if a function node is likely an augmentation function."""
        # Check function name for augmentation indicators
        augmentation_keywords = ['augment', 'transform', 'flip', 'rotate', 'zoom', 'shift', 'cutout', 'mixup']
        has_augmentation_name = any(keyword in node.name.lower() for keyword in augmentation_keywords)
        
        # Check if function has common augmentation parameters
        args = [arg.arg for arg in node.args.args]
        augmentation_args = ['image', 'data', 'array', 'x', 'input', 'tensor']
        has_augmentation_args = any(arg in args for arg in augmentation_args)
        
        return has_augmentation_name or has_augmentation_args
    
    def _is_augmentation_class(self, node):
        """Check if a class node is likely an augmentation class."""
        # Check class name for augmentation indicators
        augmentation_keywords = ['augment', 'transform', 'flip', 'rotate', 'zoom', 'shift']
        has_augmentation_name = any(keyword in node.name.lower() for keyword in augmentation_keywords)
        
        # Check if class has __call__ method (callable augmentation)
        has_call = any(isinstance(child, ast.FunctionDef) and child.name == '__call__' 
                      for child in node.body)
        
        return has_augmentation_name or has_call
    
    def _analyze_augmentation_function(self, node, file_path):
        """Analyze an augmentation function and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom augmentation function: {node.name}"
            
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
    
    def _analyze_augmentation_class(self, node, file_path):
        """Analyze an augmentation class and extract its metadata."""
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
            docstring = ast.get_docstring(node) or f"Custom augmentation class: {node.name}"
            
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
        
        if any(word in param_name_lower for word in ['range', 'factor', 'rate', 'alpha', 'scale', 'intensity']):
            return 'float'
        elif any(word in param_name_lower for word in ['size', 'width', 'height', 'num', 'count']):
            return 'int'
        elif any(word in param_name_lower for word in ['flip', 'enable', 'use', 'apply']):
            return 'bool'
        elif any(word in param_name_lower for word in ['method', 'mode', 'interpolation']):
            return 'str'
        else:
            return 'float'  # Default to float for augmentation parameters
    
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
    
    def get_augmentation_config(self):
        """Get the current augmentation configuration for training."""
        config = {}
        
        # Basic augmentation options
        config['augmentation_enabled'] = self.config['Augmentation Options']['enable_augmentation']
        config['augmentation_probability'] = self.config['Augmentation Options']['augmentation_probability']
        config['apply_to_validation'] = self.config['Augmentation Options']['apply_to_validation']
        
        # Image augmentation parameters
        image_augmentation = {}
        for key, value in self.config['Image Augmentation'].items():
            if isinstance(value, (int, float)) and value != 0.0:
                image_augmentation[key] = value
            elif isinstance(value, bool) and value:
                image_augmentation[key] = value
            elif isinstance(value, list) and value != [1.0, 1.0]:
                image_augmentation[key] = value
        
        if image_augmentation:
            config['image_augmentation'] = image_augmentation
        
        # Advanced augmentation parameters
        advanced_augmentation = {}
        for key, value in self.config['Advanced Augmentation'].items():
            if isinstance(value, (int, float)) and value != 0.0:
                advanced_augmentation[key] = value
            elif isinstance(value, bool) and value:
                advanced_augmentation[key] = value
            elif isinstance(value, list) and value:
                advanced_augmentation[key] = value
        
        if advanced_augmentation:
            config['advanced_augmentation'] = advanced_augmentation
        
        # Custom augmentations
        custom_augmentations = []
        for aug_name, aug_config in self.config['Custom Parameters'].items():
            if aug_config.get('enabled', False):
                custom_aug = {
                    'name': aug_name,
                    'parameters': {k: v for k, v in aug_config.items() if k not in ['enabled']}
                }
                if aug_name in self._custom_augmentations:
                    custom_aug['function_info'] = self._custom_augmentations[aug_name]
                custom_augmentations.append(custom_aug)
        
        if custom_augmentations:
            config['custom_augmentations'] = custom_augmentations
        
        return config
    
    def load_custom_augmentation_from_metadata(self, metadata):
        """Load a custom augmentation from saved metadata."""
        try:
            file_path = metadata.get('file_path')
            function_name = metadata.get('function_name')
            
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Load the function
            spec = importlib.util.spec_from_file_location("custom_augmentation", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, function_name):
                    func = getattr(module, function_name)
                    
                    # Store the function
                    self._custom_augmentations[function_name] = {
                        'function': func,
                        'metadata': metadata
                    }
                    
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Error loading custom augmentation: {str(e)}")
            return False
