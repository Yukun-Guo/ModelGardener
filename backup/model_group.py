"""
Model configuration group for dynamic model-specific parameters and custom model loading.
"""

import os
import sys
import importlib.util
import inspect
import tensorflow as tf
import keras

# CLI-only message functions
def cli_info(title, message):
    print(f"[INFO] {title}: {message}")

def cli_warning(title, message):
    print(f"[WARNING] {title}: {message}")

def cli_error(title, message):
    print(f"[ERROR] {title}: {message}")

# Try to import pyqtgraph (for GUI functionality) but make it optional
try:
    from pyqtgraph.parametertree import Parameter, ParameterTree
    from pyqtgraph.parametertree.parameterTypes import GroupParameter
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    # Create dummy base class for CLI-only mode
    class GroupParameter:
        def __init__(self, **opts):
            self.opts = opts
            self.name_value = opts.get('name', 'test')
            self._children = []
        
        def name(self):
            return self.name_value
            
        def children(self):
            return self._children
            
        def addChild(self, child):
            self._children.append(child)
            
        def removeChild(self, child):
            if child in self._children:
                self._children.remove(child)
        
        def child(self, name):
            for child in self._children:
                if child.name() == name:
                    return child
            return None
            
        def clearChildren(self):
            self._children.clear()


if not PYQTGRAPH_AVAILABLE:
    # Create Parameter class for CLI mode
    class Parameter:
        def __init__(self, **opts):
            self.opts = opts
            self.name_value = opts.get('name', 'parameter')
            self._value = opts.get('value', None)
            self._children = []
        
        def name(self):
            return self.name_value
            
        def value(self):
            return self._value
            
        def setValue(self, value):
            self._value = value
            
        def children(self):
            return self._children
            
        def addChild(self, child):
            if isinstance(child, dict):
                child = Parameter(**child)
            self._children.append(child)
            
        def child(self, name):
            for child in self._children:
                if child.name() == name:
                    return child
            return None


# Model architectures mapping
MODEL_ARCHITECTURES = {
    'resnet': {
        'ResNet-50': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        },
        'ResNet-101': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        },
        'ResNet-152': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        }
    },
    'vgg': {
        'VGG-16': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        },
        'VGG-19': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        }
    },
    'densenet': {
        'DenseNet-121': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        },
        'DenseNet-169': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        }
    },
    'efficientnet': {
        'EfficientNetB0': {
            'input_shape': {'height': 224, 'width': 224, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        },
        'EfficientNetB1': {
            'input_shape': {'height': 240, 'width': 240, 'channels': 3},
            'include_top': {'type': 'bool', 'default': True},
            'weights': {'type': 'str', 'default': 'imagenet'},
            'classes': {'type': 'int', 'default': 1000}
        }
    }
}


class ModelGroup(GroupParameter):
    """Model configuration group with CLI support."""
    
    def __init__(self, **opts):
        super().__init__(**opts)
        self._custom_models = {}
        self._setup_model_parameters()
    
    def _setup_model_parameters(self):
        """Set up basic model parameters."""
        if PYQTGRAPH_AVAILABLE:
            self._setup_gui_parameters()
        else:
            self._setup_cli_parameters()
    
    def _setup_gui_parameters(self):
        """Set up parameters for GUI mode (original functionality)."""
        # This would contain the original pyqtgraph parameter setup
        pass
    
    def _setup_cli_parameters(self):
        """Set up parameters for CLI mode."""
        # Basic model configuration for CLI
        self.model_family = 'resnet'
        self.model_name = 'ResNet-50'
        self.model_parameters = MODEL_ARCHITECTURES['resnet']['ResNet-50'].copy()
    
    def get_model_config(self):
        """Get current model configuration."""
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'children'):
            # GUI mode - extract from parameter tree
            config = {}
            for child in self.children():
                if child.name() == 'model_family':
                    config['model_family'] = child.value()
                elif child.name() == 'model_name':
                    config['model_name'] = child.value()
                elif child.name() == 'model_parameters':
                    config['model_parameters'] = {}
                    for param_child in child.children():
                        config['model_parameters'][param_child.name()] = param_child.value()
            return config
        else:
            # CLI mode - return stored values
            return {
                'model_family': getattr(self, 'model_family', 'resnet'),
                'model_name': getattr(self, 'model_name', 'ResNet-50'),
                'model_parameters': getattr(self, 'model_parameters', MODEL_ARCHITECTURES['resnet']['ResNet-50'].copy())
            }
    
    def set_model_config(self, config):
        """Set model configuration."""
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'children'):
            # GUI mode - update parameter tree
            for child in self.children():
                if child.name() == 'model_family' and 'model_family' in config:
                    child.setValue(config['model_family'])
                elif child.name() == 'model_name' and 'model_name' in config:
                    child.setValue(config['model_name'])
                elif child.name() == 'model_parameters' and 'model_parameters' in config:
                    for param_name, value in config['model_parameters'].items():
                        param_child = child.child(param_name)
                        if param_child:
                            param_child.setValue(value)
        else:
            # CLI mode - store values
            if 'model_family' in config:
                self.model_family = config['model_family']
            if 'model_name' in config:
                self.model_name = config['model_name']
            if 'model_parameters' in config:
                self.model_parameters = config['model_parameters']
    
    def load_custom_model_from_path(self, file_path):
        """Load custom model from file path (CLI compatible)."""
        try:
            if not os.path.exists(file_path):
                cli_error("File Error", f"Custom model file does not exist: {file_path}")
                return False, None
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_model", file_path)
            if spec is None or spec.loader is None:
                cli_error("Import Error", f"Cannot load module from: {file_path}")
                return False, None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find model functions/classes
            model_functions = []
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith('_'):
                    # Check if it looks like a model function/class
                    if self._is_model_function(obj, name):
                        model_functions.append((name, obj))
            
            if not model_functions:
                cli_warning("No Models Found", "No valid model functions found in the selected file.")
                return False, None
            
            # Store custom models
            for name, obj in model_functions:
                model_info = {
                    'name': name,
                    'file_path': file_path,
                    'type': 'function' if inspect.isfunction(obj) else 'class',
                    'parameters': self._extract_model_parameters(obj)
                }
                self._custom_models[name] = model_info
            
            cli_info("Models Loaded", f"Successfully loaded {len(model_functions)} custom model(s) from {file_path}")
            return True, self._custom_models
            
        except Exception as e:
            cli_error("Load Error", f"Failed to load custom model: {str(e)}")
            return False, None
    
    def _is_model_function(self, obj, name):
        """Check if an object is likely a model function."""
        # Simple heuristics for model detection
        name_lower = name.lower()
        return (
            'model' in name_lower or
            'net' in name_lower or 
            'classifier' in name_lower or
            'encoder' in name_lower or
            'decoder' in name_lower
        )
    
    def _extract_model_parameters(self, obj):
        """Extract parameters from model function/class."""
        parameters = {}
        try:
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                for param_name, param in sig.parameters.items():
                    if param_name != 'self':
                        param_info = {'type': 'str', 'default': None}
                        if param.default != inspect.Parameter.empty:
                            param_info['default'] = param.default
                            # Infer type from default value
                            if isinstance(param.default, bool):
                                param_info['type'] = 'bool'
                            elif isinstance(param.default, int):
                                param_info['type'] = 'int'
                            elif isinstance(param.default, float):
                                param_info['type'] = 'float'
                        parameters[param_name] = param_info
            elif inspect.isclass(obj):
                # Extract from __init__ method
                init_method = getattr(obj, '__init__', None)
                if init_method:
                    sig = inspect.signature(init_method)
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self':
                            param_info = {'type': 'str', 'default': None}
                            if param.default != inspect.Parameter.empty:
                                param_info['default'] = param.default
                                if isinstance(param.default, bool):
                                    param_info['type'] = 'bool'
                                elif isinstance(param.default, int):
                                    param_info['type'] = 'int'
                                elif isinstance(param.default, float):
                                    param_info['type'] = 'float'
                            parameters[param_name] = param_info
        except Exception as e:
            cli_warning("Parameter Extraction", f"Could not extract parameters from {obj}: {e}")
        
        return parameters
    
    def load_custom_model_from_metadata(self, model_info):
        """Load custom model from metadata (for config loading)."""
        try:
            file_path = model_info.get('file_path')
            if not file_path or not os.path.exists(file_path):
                cli_error("Model Load Error", f"Custom model file not found: {file_path}")
                return False
            
            success, loaded_models = self.load_custom_model_from_path(file_path)
            if success:
                # Update current model selection to the loaded custom model
                model_name = model_info.get('function_name') or model_info.get('class_name')
                if model_name and model_name in loaded_models:
                    self.model_family = 'custom'
                    self.model_name = model_name
                    self.model_parameters = loaded_models[model_name].get('parameters', {})
                    cli_info("Model Loaded", f"Loaded custom model: {model_name}")
                    return True
            
            return False
            
        except Exception as e:
            cli_error("Load Error", f"Failed to load custom model from metadata: {str(e)}")
            return False
    
    def get_available_models(self):
        """Get list of available models including custom ones."""
        available = {}
        
        # Add built-in models
        for family, models in MODEL_ARCHITECTURES.items():
            available[family] = list(models.keys())
        
        # Add custom models
        if self._custom_models:
            available['custom'] = list(self._custom_models.keys())
        
        return available
    
    def get_model_parameters(self, model_family, model_name):
        """Get parameters for a specific model."""
        if model_family == 'custom' and model_name in self._custom_models:
            return self._custom_models[model_name]['parameters']
        elif model_family in MODEL_ARCHITECTURES and model_name in MODEL_ARCHITECTURES[model_family]:
            return MODEL_ARCHITECTURES[model_family][model_name].copy()
        else:
            return {}


# For backwards compatibility, create a simple function that returns configuration
def create_model_config(model_family='resnet', model_name='ResNet-50', **kwargs):
    """Create a model configuration dictionary."""
    model_group = ModelGroup()
    model_group.model_family = model_family
    model_group.model_name = model_name
    
    # Update parameters if provided
    if model_family in MODEL_ARCHITECTURES and model_name in MODEL_ARCHITECTURES[model_family]:
        model_group.model_parameters = MODEL_ARCHITECTURES[model_family][model_name].copy()
        model_group.model_parameters.update(kwargs)
    else:
        model_group.model_parameters = kwargs
    
    return model_group.get_model_config()
