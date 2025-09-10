
import os
import inspect
import importlib.util
from typing import Any, Dict, List, Tuple
# Import script generator
try:
    from .script_generator import ScriptGenerator
except ImportError:
    print("Warning: ScriptGenerator not available")
    ScriptGenerator = None

def is_model_function(obj, name: str) -> bool:
    """Check if an object is a model function based on example_funcs pattern."""
    try:
        if inspect.isfunction(obj):
            # Check function signature for model parameters
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Model functions should have input_shape and/or num_classes parameters
            has_input_shape = 'input_shape' in params
            has_num_classes = 'num_classes' in params or 'classes' in params
            
            # Check function name patterns
            name_lower = name.lower()
            model_name_patterns = ['model', 'net', 'network', 'cnn', 'resnet', 'efficientnet', 'mobilenet', 'unet']
            has_model_name = any(pattern in name_lower for pattern in model_name_patterns)
            
            # Check docstring for model-related keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            model_keywords = ['model', 'architecture', 'neural network', 'keras.model', 'returns', 'keras model']
            has_model_keywords = any(keyword in docstring_lower for keyword in model_keywords)
            
            # Must have model parameters OR model name/keywords
            return (has_input_shape or has_num_classes) or (has_model_name and has_model_keywords)
            
        elif inspect.isclass(obj):
            # Check if class looks like a model class
            name_lower = name.lower()
            class_patterns = ['model', 'net', 'network', 'cnn', 'classifier']
            return any(pattern in name_lower for pattern in class_patterns)
                
    except Exception:
        pass
    
    return False

def is_data_loader_function(obj, name: str) -> bool:
    """
    Check if an object is a data loader function based on example_funcs wrapper pattern.
    
    Pattern: outer function with config params, returns wrapper function that takes (train_dir, val_dir)
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        bool: True if it's a valid data loader function
    """
    import inspect
    
    # Skip private functions, imports, and common utilities
    if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'os', 'sys', 'train_test_split', 'pd', 'pandas']:
        return False
    
    # Skip objects from imported modules, but allow dynamically loaded modules and custom modules
    if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
        # Skip standard library and common third-party modules
        skip_modules = ['builtins', 'numpy', 'tensorflow', 'keras', 'pandas', 'sklearn', 'torch', 'torchvision']
        if any(obj.__module__.startswith(mod) for mod in skip_modules):
            return False
        
    try:
        if inspect.isfunction(obj):
            # Check function name patterns
            name_lower = name.lower()
            data_loader_patterns = ['loader', 'data', 'dataset', 'load', 'reader', 'importer']
            has_data_loader_name = any(pattern in name_lower for pattern in data_loader_patterns)
            
            # Check docstring for data loader keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            data_loader_keywords = ['data loader', 'dataset', 'load data', 'wrapper', 'tf.data', 'training and validation']
            has_data_loader_keywords = any(keyword in docstring_lower for keyword in data_loader_keywords)
            
            # Check function signature for data loader configuration parameters
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Data loader wrapper pattern: has config params like batch_size, shuffle, etc.
            # Should NOT have train_dir/val_dir (those are in the inner wrapper)
            config_indicators = ['batch_size', 'shuffle', 'buffer_size', 'validation_split', 'epochs', 'seed']
            has_config_params = any(indicator in param.lower() for param in params for indicator in config_indicators)
            
            # Should not have data path params (those are in inner wrapper)
            path_indicators = ['train_dir', 'val_dir', 'data_dir', 'path']
            has_no_path_params = not any(indicator in param.lower() for param in params for indicator in path_indicators)
            
            # Data loader functions must have name/keywords AND follow wrapper pattern (config params OR no path params)
            return (has_data_loader_name or has_data_loader_keywords) and (has_config_params and has_no_path_params)
            
        elif inspect.isclass(obj):
            # Check if class has data loader-like methods
            methods = [method for method in dir(obj) if not method.startswith('_')]
            data_loader_methods = ['load', 'get_dataset', 'load_data', 'get_data']
            has_loader_methods = any(method.lower() in [m.lower() for m in data_loader_methods] for method in methods)
            
            # Check class docstring
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            class_keywords = ['data loader', 'dataset', 'load data', 'dataloader']
            has_class_keywords = any(keyword in docstring_lower for keyword in class_keywords)
            
            return has_loader_methods or has_class_keywords
            
    except Exception:
        return False
        
    return False

def is_loss_function( obj, name: str) -> bool:
    """
    Check if an object is a loss function based on example_funcs wrapper pattern.
    
    Pattern: outer function with config params, returns wrapper function that takes (y_true, y_pred)
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        bool: True if it's a valid loss function
    """
    import inspect
    
    # Skip private functions, imports, and common utilities
    if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'keras', 'K']:
        return False
    
    # Skip objects from imported modules, but allow dynamically loaded modules and custom modules
    if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
        # Skip standard library and common third-party modules
        skip_modules = ['builtins', 'numpy', 'tensorflow', 'keras', 'pandas', 'sklearn', 'torch', 'torchvision']
        if any(obj.__module__.startswith(mod) for mod in skip_modules):
            return False
        
    try:
        if inspect.isfunction(obj):
            # Check function name for loss patterns (more lenient matching)
            name_lower = name.lower()
            loss_name_patterns = ['loss', 'cost', 'error', 'distance', 'divergence', 'mse', 'mae', 'bce', 'crossentropy', 'example_loss']
            has_loss_name = any(pattern in name_lower for pattern in loss_name_patterns)
            
            # Check docstring for loss function keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            loss_keywords = ['loss', 'cost', 'error', 'distance', 'divergence', 'wrapper', 'y_true', 'y_pred', 'loss_value', 'loss calculation']
            has_loss_keywords = any(keyword in docstring_lower for keyword in loss_keywords)
            
            # Check function signature for wrapper pattern
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Wrapper pattern: should NOT have y_true, y_pred in outer function
            # These should be in the inner wrapper function
            execution_indicators = ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction', 'labels', 'logits']
            has_no_execution_params = not any(indicator in param.lower() for param in params for indicator in execution_indicators)
            
            # Loss functions MUST have appropriate name patterns AND follow wrapper pattern
            return has_loss_name and has_no_execution_params
            
        elif inspect.isclass(obj):
            # Check if class inherits from typical loss classes or has loss-like methods
            methods = [method for method in dir(obj) if not method.startswith('_')]
            loss_methods = ['call', '__call__', 'compute_loss', 'calculate_loss']
            has_loss_methods = any(method.lower() in [m.lower() for m in loss_methods] for method in methods)
            
            # Check class docstring
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            class_keywords = ['loss', 'cost function', 'objective function']
            has_class_keywords = any(keyword in docstring_lower for keyword in class_keywords)
            
            return has_loss_methods or has_class_keywords
            
    except Exception:
        return False
        
    return False

def is_likely_metrics_function( func_name: str, args: List[str], content: str) -> bool:
    """Check if a function is likely a metrics function."""
    func_name_lower = func_name.lower()
    
    # Common metrics function name patterns
    metrics_patterns = [
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'score', 'metric',
        'mse', 'mae', 'rmse', 'loss', 'error', 'iou', 'dice'
    ]
    
    # Check if function name contains metrics patterns
    name_matches = any(pattern in func_name_lower for pattern in metrics_patterns)
    
    # Check for typical metrics function parameters
    typical_params = ['y_true', 'y_pred', 'true', 'pred', 'actual', 'predicted', 'labels', 'outputs']
    param_matches = any(param in ' '.join(args).lower() for param in typical_params)
    
    # Check if function has at least 2 parameters (typical for metrics: y_true, y_pred)
    has_enough_params = len(args) >= 2
    
    return (name_matches or param_matches) and has_enough_params

def is_improved_template_config( config: Dict[str, Any]) -> bool:
    """Check if this is an improved template configuration that needs custom YAML formatting."""
    # Check for custom augmentation/preprocessing/callback options that indicate improved template
    try:
        data_config = config.get('configuration', {}).get('data', {})
        model_config = config.get('configuration', {}).get('model', {})
        
        # Check for custom functions by looking for entries with function_name and file_path
        has_custom_aug = any(
            isinstance(v, dict) and 'function_name' in v and 'file_path' in v 
            for v in data_config.get('augmentation', {}).values()
        )
        has_custom_prep = any(
            isinstance(v, dict) and 'function_name' in v and 'file_path' in v 
            for v in data_config.get('preprocessing', {}).values()
        )
        has_custom_callback = any(
            isinstance(v, dict) and 'function_name' in v and 'file_path' in v 
            for v in model_config.get('callbacks', {}).values()
        )
        
        return has_custom_aug or has_custom_prep or has_custom_callback
    except:
        return False

def is_preprocessing_function( obj, name: str) -> bool:
    """
    Check if an object is a preprocessing function based on example_funcs wrapper pattern.
    
    Pattern: outer function with config params, returns wrapper function that takes (data, label)
    """
    try:
        if inspect.isfunction(obj):
            # Check function name patterns
            name_lower = name.lower()
            name_patterns = [
                'preprocess', 'process', 'transform', 'normalize', 'resize', 'scale', 
                'adjust', 'crop', 'pad', 'enhance', 'filter', 'convert'
            ]
            has_preprocessing_name = any(pattern in name_lower for pattern in name_patterns)
            
            # Check docstring for preprocessing-related keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            preprocessing_keywords = [
                'preprocess', 'transform', 'normalize', 'resize', 'enhance', 'filter', 
                'wrapper', 'data', 'label', 'processed_data', 'processed_label'
            ]
            has_preprocessing_keywords = any(keyword in docstring_lower for keyword in preprocessing_keywords)
            
            # Check function signature for wrapper pattern
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Wrapper pattern: should NOT have data, label in outer function
            # These should be in the inner wrapper function
            execution_indicators = ['data', 'image', 'img', 'input', 'x', 'array', 'tensor', 'label', 'labels']
            has_no_execution_params = not any(indicator in param.lower() for param in params for indicator in execution_indicators)
            
            # Should have configuration parameters (or no parameters for simple cases)
            config_indicators = ['param', 'size', 'scale', 'factor', 'threshold', 'alpha', 'beta', 'gamma']
            has_config_params = any(indicator in param.lower() for param in params for indicator in config_indicators) or len(params) == 0
            
            # Preprocessing functions must have name/keywords AND follow wrapper pattern
            return (has_preprocessing_name or has_preprocessing_keywords) and has_no_execution_params
            
    except Exception:
        pass
    
    return False

def is_augmentation_function( obj, name: str) -> bool:
    """
    Check if an object is an augmentation function based on example_funcs wrapper pattern.
    
    Pattern: outer function with config params, returns wrapper function that takes (data, label)
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        True if it's a valid augmentation function
    """
    
    # Must be callable
    if not callable(obj):
        return False
    
    # Skip private/magic methods
    if name.startswith('_'):
        return False
        
    # Skip common non-augmentation functions
    skip_names = {'main', 'setup', 'init', 'test', 'demo'}
    if name.lower() in skip_names:
        return False
    
    try:
        # Check function name patterns for augmentation
        name_lower = name.lower()
        augmentation_patterns = [
            'augment', 'flip', 'rotate', 'brightness', 'contrast', 'blur', 'noise',
            'crop', 'zoom', 'shift', 'color', 'hue', 'saturation', 'jitter',
            'distort', 'elastic', 'random', 'transform'
        ]
        has_augmentation_name = any(pattern in name_lower for pattern in augmentation_patterns)
        
        # Check docstring for augmentation-related keywords  
        docstring = inspect.getdoc(obj) or ""
        docstring_lower = docstring.lower()
        augmentation_keywords = [
            'augment', 'random', 'flip', 'rotate', 'brightness', 'contrast', 'blur', 'noise', 
            'crop', 'zoom', 'distort', 'transform', 'color', 'hue', 'saturation', 'wrapper',
            'data', 'label', 'modified_data', 'modified_label'
        ]
        has_augmentation_keywords = any(keyword in docstring_lower for keyword in augmentation_keywords)
        
        # Check function signature for wrapper pattern
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Wrapper pattern: should NOT have data, label in outer function
            # These should be in the inner wrapper function
            execution_indicators = ['data', 'image', 'img', 'input', 'x', 'array', 'tensor', 'label', 'labels']
            has_no_execution_params = not any(indicator in param.lower() for param in params for indicator in execution_indicators)
            
            # Should have configuration parameters (or no parameters for simple cases)
            config_indicators = ['param', 'probability', 'rate', 'factor', 'range', 'alpha', 'beta', 'gamma', 'angle', 'scale']
            has_config_params = any(indicator in param.lower() for param in params for indicator in config_indicators) or len(params) == 0
            
            # Augmentation functions must have name/keywords AND follow wrapper pattern
            return (has_augmentation_name or has_augmentation_keywords) and has_no_execution_params
            
    except Exception:
        return False
    
    return False

def is_callback_function(obj, name: str) -> bool:
    """
    Check if an object is a callback function or class based on example_funcs pattern.
    
    Pattern: classes inheriting from keras.callbacks.Callback
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        bool: True if it's a valid callback function/class
    """
    # Skip private functions
    if name.startswith('_'):
        return False
    
    # Skip common non-callback functions
    skip_names = {'main', 'setup', 'init', 'test', 'demo', 'load', 'save', 'print'}
    if name.lower() in skip_names:
        return False
    
    try:
        if inspect.isclass(obj):
            # Check function/class name patterns for callbacks
            name_lower = name.lower()
            callback_patterns = [
                'callback', 'early', 'stopping', 'checkpoint', 'tensorboard', 'csv',
                'logger', 'monitor', 'scheduler', 'plateau', 'reduce', 'lr', 'rate',
                'decay', 'custom', 'history', 'progress', 'bar', 'metric', 'epoch',
                'batch', 'terminate', 'nan', 'backup', 'remote'
            ]
            
            has_callback_name = any(pattern in name_lower for pattern in callback_patterns)
            
            # Check docstring for callback-related keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            callback_keywords = [
                'callback', 'epoch', 'batch', 'training', 'monitor', 'metric',
                'checkpoint', 'early stopping', 'learning rate', 'tensorboard',
                'csv logger', 'progress', 'history', 'validation'
            ]
            has_callback_keywords = any(keyword in docstring_lower for keyword in callback_keywords)
            
            # For classes, check if it inherits from Keras callback
            try:
                import tensorflow as tf
                # Check inheritance from tf.keras.callbacks.Callback
                if issubclass(obj, tf.keras.callbacks.Callback):
                    return True
            except (ImportError, TypeError):
                pass
            
            try:
                import keras
                # Check inheritance from keras.callbacks.Callback  
                if issubclass(obj, keras.callbacks.Callback):
                    return True
            except (ImportError, TypeError):
                pass
            
            # Check if class has typical callback methods
            methods = [method for method in dir(obj) if not method.startswith('_')]
            callback_methods = ['on_epoch_end', 'on_batch_end', 'on_train_begin', 'on_train_end', 'on_epoch_begin', 'on_batch_begin']
            has_callback_methods = any(method in methods for method in callback_methods)
            
            return has_callback_name or has_callback_keywords or has_callback_methods
            
        elif inspect.isfunction(obj):
            # For functions, check if it returns a callback-like object
            name_lower = name.lower()
            callback_patterns = ['callback', 'early', 'stopping', 'checkpoint', 'monitor']
            has_callback_name = any(pattern in name_lower for pattern in callback_patterns)
            
            try:
                sig = inspect.signature(obj)
                return_annotation = sig.return_annotation
                if return_annotation != inspect.Signature.empty:
                    return_type_str = str(return_annotation)
                    if 'callback' in return_type_str.lower():
                        return True
            except Exception:
                pass
                
            return has_callback_name
                
    except Exception:
        pass
    
    return False

def extract_model_parameters(obj) -> Dict[str, Any]:
    """Extract parameters from model function/class."""
    parameters = {}
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            for param_name, param in sig.parameters.items():
                if param_name not in ['self', 'input_shape', 'num_classes']:
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
                    if param_name not in ['self', 'input_shape', 'num_classes']:
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
    except Exception:
        pass
    
    return parameters

def extract_data_loader_parameters(obj) -> Dict[str, Any]:
    """
    Extract parameters from a data loader function or class.
    
    Args:
        obj: The data loader function or class
        
    Returns:
        Dict containing parameter information
    """
    import inspect
    
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            params = {}
            
            for param_name, param in sig.parameters.items():
                # Skip common fixed parameters
                if param_name in ['data_dir', 'train_dir', 'val_dir', 'split']:
                    continue
                    
                param_info = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }
                
                # Infer parameter type
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = str(param.annotation)
                elif param.default is not None:
                    param_info['type'] = type(param.default).__name__
                else:
                    param_info['type'] = 'Any'
                
                params[param_name] = param_info
            
            return {
                'type': 'function',
                'parameters': params,
                'signature': str(sig),
                'description': inspect.getdoc(obj) or f"Data loader function: {obj.__name__}"
            }
            
        elif inspect.isclass(obj):
            # Get constructor parameters
            init_method = getattr(obj, '__init__', None)
            params = {}
            
            if init_method:
                sig = inspect.signature(init_method)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    if param_name in ['data_dir', 'train_dir', 'val_dir']:
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
            
            return {
                'type': 'class',
                'parameters': params,
                'signature': f"class {obj.__name__}",
                'description': inspect.getdoc(obj) or f"Data loader class: {obj.__name__}"
            }
            
    except Exception:
        return {
            'type': 'unknown',
            'parameters': {},
            'signature': '',
            'description': f"Data loader: {getattr(obj, '__name__', 'Unknown')}"
        }
    
    return {}

def extract_loss_parameters(obj) -> Dict[str, Any]:
    """
    Extract parameters from a loss function (supports wrapper pattern).
    
    Args:
        obj: The loss function or class
        
    Returns:
        Dict containing parameter information
    """
    import inspect
    
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            params = {}
            
            for param_name, param in sig.parameters.items():
                # For wrapper pattern, we want the configuration parameters (outer function)
                # Skip y_true, y_pred parameters if they exist (traditional pattern)
                if param_name.lower() in ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction']:
                    continue
                    
                param_info = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }
                
                # Infer parameter type
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = str(param.annotation)
                elif param.default is not None:
                    param_info['type'] = type(param.default).__name__
                else:
                    param_info['type'] = 'Any'
                
                params[param_name] = param_info
            
            return {
                'type': 'function',
                'parameters': params,
                'signature': str(sig),
                'description': inspect.getdoc(obj) or f"Loss function: {obj.__name__}"
            }
            
        elif inspect.isclass(obj):
            # Get constructor parameters
            init_method = getattr(obj, '__init__', None)
            params = {}
            
            if init_method:
                sig = inspect.signature(init_method)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
            
            return {
                'type': 'class',
                'parameters': params,
                'signature': f"class {obj.__name__}",
                'description': inspect.getdoc(obj) or f"Loss class: {obj.__name__}"
            }
            
    except Exception:
        return {
            'type': 'unknown',
            'parameters': {},
            'signature': '',
            'description': f"Loss function: {getattr(obj, '__name__', 'Unknown')}"
        }
    
    return {}

def is_metrics_function(obj, name: str) -> bool:
    """
    Check if an object is a metrics function based on example_funcs wrapper pattern.
    
    Pattern: outer function with config params, returns wrapper function that takes (y_true, y_pred)
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        bool: True if it's a valid metrics function
    """
    import inspect
    
    # Skip private functions, imports, and common utilities
    if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'keras', 'K']:
        return False
    
    # Skip objects from imported modules, but allow dynamically loaded modules and custom modules
    if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
        # Skip standard library and common third-party modules
        skip_modules = ['builtins', 'numpy', 'tensorflow', 'keras', 'pandas', 'sklearn', 'torch', 'torchvision']
        if any(obj.__module__.startswith(mod) for mod in skip_modules):
            return False
        
    try:
        if inspect.isfunction(obj):
            # Check function name for metrics patterns (more lenient matching)
            name_lower = name.lower()
            metrics_name_patterns = ['metric', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'score', 'mse', 'mae', 'rmse', 'iou', 'dice', 'example_metric']
            has_metrics_name = any(pattern in name_lower for pattern in metrics_name_patterns)
            
            # Check docstring for metrics function keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            metrics_keywords = ['metric', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'score', 'measure', 'evaluation', 'wrapper', 'y_true', 'y_pred', 'metric_value', 'metric calculation']
            has_metrics_keywords = any(keyword in docstring_lower for keyword in metrics_keywords)
            
            # Check function signature for wrapper pattern
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Wrapper pattern: should NOT have y_true, y_pred in outer function
            # These should be in the inner wrapper function
            execution_indicators = ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction', 'labels', 'logits']
            has_no_execution_params = not any(indicator in param.lower() for param in params for indicator in execution_indicators)
            
            # Metrics functions MUST have appropriate name patterns AND follow wrapper pattern
            return has_metrics_name and has_no_execution_params
            
        elif inspect.isclass(obj):
            # Check if class inherits from typical metrics classes or has metrics-like methods
            methods = [method for method in dir(obj) if not method.startswith('_')]
            metrics_methods = ['call', '__call__', 'compute_metric', 'calculate_metric', 'evaluate', 'score']
            has_metrics_methods = any(method.lower() in [m.lower() for m in metrics_methods] for method in methods)
            
            # Check class docstring
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            class_keywords = ['metric', 'accuracy', 'precision', 'recall', 'evaluation', 'measure']
            has_class_keywords = any(keyword in docstring_lower for keyword in class_keywords)
            
            return has_metrics_methods or has_class_keywords
            
    except Exception:
        return False
        
    return False

def extract_metrics_parameters(obj) -> Dict[str, Any]:
    """
    Extract parameters from a metrics function (supports wrapper pattern).
    
    Args:
        obj: The metrics function or class
        
    Returns:
        Dict containing parameter information
    """
    import inspect
    
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            params = {}
            
            for param_name, param in sig.parameters.items():
                # For wrapper pattern, we want the configuration parameters (outer function)
                # Skip y_true, y_pred parameters if they exist (traditional pattern)
                if param_name.lower() in ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction']:
                    continue
                    
                param_info = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }
                
                # Infer parameter type
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = str(param.annotation)
                elif param.default is not None:
                    param_info['type'] = type(param.default).__name__
                else:
                    param_info['type'] = 'Any'
                
                params[param_name] = param_info
            
            return {
                'type': 'function',
                'parameters': params,
                'signature': str(sig),
                'description': inspect.getdoc(obj) or f"Metrics function: {obj.__name__}"
            }
            
        elif inspect.isclass(obj):
            # Get constructor parameters
            init_method = getattr(obj, '__init__', None)
            params = {}
            
            if init_method:
                sig = inspect.signature(init_method)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
            
            return {
                'type': 'class',
                'parameters': params,
                'signature': f"class {obj.__name__}",
                'description': inspect.getdoc(obj) or f"Metrics class: {obj.__name__}"
            }
            
    except Exception:
        return {
            'type': 'unknown',
            'parameters': {},
            'signature': '',
            'description': f"Metrics function: {getattr(obj, '__name__', 'Unknown')}"
        }
    
    return {}


def analyze_custom_model_outputs(file_path: str, function_name: str, 
                                model_config: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Analyze a custom model function to determine its outputs.
    
    Args:
        file_path: Path to the Python file containing the model
        function_name: Name of the model function
        model_config: Model configuration parameters
        
    Returns:
        Tuple of (num_outputs, output_names)
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("custom_model", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the model function
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in {file_path}")
    
    model_func = getattr(module, function_name)
    
    # Try to build the model to analyze its structure
    try:
        # Complete suppression of TensorFlow warnings during model building
        import sys
        import contextlib
        from io import StringIO
        
        # Create context manager to suppress all output
        @contextlib.contextmanager
        def suppress_output():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        # Prepare model parameters
        model_params = model_config.get('model_parameters', {})
        input_shape = (
            model_params.get('input_shape', {}).get('height', 224),
            model_params.get('input_shape', {}).get('width', 224), 
            model_params.get('input_shape', {}).get('channels', 3)
        )
        num_classes = model_params.get('classes', 10)
        
        # Build the model with complete output suppression
        with suppress_output():
            import keras
            
            if inspect.isclass(model_func):
                # If it's a class, instantiate it
                model = model_func(input_shape=input_shape, num_classes=num_classes)
            else:
                # If it's a function, call it
                model = model_func(input_shape=input_shape, num_classes=num_classes)
        
        if hasattr(model, 'outputs') and hasattr(model.outputs, '__len__'):
            num_outputs = len(model.outputs)
            output_names = []
            
            for i, output in enumerate(model.outputs):
                output_name = None
                
                if hasattr(output, 'name') and output.name:
                    # Extract clean name from tensor name (remove :0 suffix and path)
                    clean_name = output.name.split(':')[0].split('/')[-1]
                    
                    # Check if it's a meaningful name (not generic tensor names)
                    if clean_name and not any(generic in clean_name.lower() for generic in 
                                            ['keras_tensor', 'dense_', 'sequential_', 'functional_']):
                        output_name = clean_name
                    
                    # Special case: look for aux/auxiliary patterns
                    if 'aux' in clean_name.lower() or 'auxiliary' in clean_name.lower():
                        output_name = clean_name
                
                # If no meaningful name found, generate a sensible default
                if not output_name:
                    if i == 0:
                        output_name = 'main_output'
                    else:
                        output_name = f'aux_output_{i}' if i == 1 else f'output_{i+1}'
                
                output_names.append(output_name)
            
            return num_outputs, output_names
        else:
            return 1, ['main_output']
            
    except Exception:
        # Fall back to source code analysis
        return analyze_model_source_code(model_func)

def analyze_model_source_code(model_func) -> Tuple[int, List[str]]:
    """
    Analyze model function source code to detect multiple outputs.
    
    Args:
        model_func: The model function to analyze
        
    Returns:
        Tuple of (num_outputs, output_names)
    """

    try:
        source = inspect.getsource(model_func)
        source_lower = source.lower()
        
        # Look for multiple outputs patterns
        multiple_output_patterns = [
            'model(inputs, [',  # keras.Model(inputs, [output1, output2])
            'return [',         # return [output1, output2]
            ', name=',         # multiple named outputs
            'outputs = [',     # outputs = [...]
            'aux_output',      # auxiliary outputs
        ]
        
        pattern_count = sum(1 for pattern in multiple_output_patterns if pattern in source_lower)
        
        if pattern_count >= 2 or 'aux_output' in source_lower:
            # Likely multiple outputs - try to extract names
            output_names = []
            
            # Look for name= patterns in layer definitions
            import re
            name_patterns = re.findall(r'name=[\'"]([^\'\"]+)[\'"]', source)
            for name in name_patterns:
                if any(keyword in name.lower() for keyword in ['output', 'aux', 'auxiliary']):
                    output_names.append(name)
            
            # Look for variable names that suggest outputs
            variable_patterns = re.findall(r'(\w*(?:output|aux)\w*)\s*=', source_lower)
            for var_name in variable_patterns:
                if var_name and var_name not in output_names:
                    output_names.append(var_name)
            
            # Clean up and validate output names
            clean_names = []
            for name in output_names:
                if name and len(name) > 0:
                    clean_names.append(name)
            
            if not clean_names:
                clean_names = ['main_output', 'aux_output']
            elif len(clean_names) == 1:
                clean_names = ['main_output', clean_names[0]]
            
            # Limit to reasonable number of outputs
            if len(clean_names) > 5:
                clean_names = clean_names[:5]
            
            return len(clean_names), clean_names
        
        return 1, ['main_output']
        
    except Exception:
        return 1, ['main_output']



def generate_python_scripts(config: Dict[str, Any], config_file_path: str):
    """
    Generate Python scripts (train.py, evaluation.py, prediction.py, deploy.py) 
    and custom modules templates in the same directory as the config file.
    
    Args:
        config: The configuration dictionary
        config_file_path: Path to the saved configuration file
    """
    if ScriptGenerator is None:
        print("⚠️  ScriptGenerator not available, skipping script generation")
        return
    
    try:
        # Get the directory where the config file is saved
        config_dir = os.path.dirname(config_file_path)
        if not config_dir:
            config_dir = '.'
        config_filename = os.path.basename(config_file_path)
        
        # Create script generator
        generator = ScriptGenerator()
        
        # Generate scripts
        print("\n🐍 Generating Python scripts...")
        success = generator.generate_scripts(config, config_dir, config_filename)
        
        # Generate custom modules templates
        print("📁 Generating custom modules templates...")
        custom_modules_success = generator.generate_custom_modules_templates(config_dir)
        
        if success:
            print("✅ Python scripts generated successfully!")
            print(f"📁 Location: {os.path.abspath(config_dir)}")
            print("📄 Generated files:")
            print("   • train.py - Training script")
            print("   • evaluation.py - Evaluation script") 
            print("   • prediction.py - Prediction script")
            print("   • deploy.py - Deployment script")
            print("   • requirements.txt - Python dependencies")
            print("   • README.md - Usage instructions")
            
            if custom_modules_success:
                print("   • custom_modules/ - Custom function templates")
        else:
            print("❌ Failed to generate some Python scripts")
            
    except Exception as e:
        print(f"❌ Error generating Python scripts: {str(e)}")


def copy_example_data(project_dir: str):
    """
    Ensure CIFAR-10 dataset exists and copy it to the project directory.
    Uses caching and automatic generation for robustness.
    Fixed to 500 samples per class.
    
    Args:
        project_dir: Target project directory
    """
    import tempfile
    
    dest_data_dir = os.path.join(project_dir, 'data')
    cifar10_dest = os.path.join(dest_data_dir, 'cifar10.npz')
    
    # Use system temp directory
    temp_dir = os.path.join(tempfile.gettempdir(), 'modelgardener_cache')
    
    try:
        # Create data directory
        os.makedirs(dest_data_dir, exist_ok=True)
        
        # Use robust dataset management with fixed 500 samples per class
        if ensure_cifar10_dataset(
            temp_dir=temp_dir,
            target_path=cifar10_dest,
            samples_per_class=500,
            verbose=True
        ):
            # Verify the copied dataset
            try:
                import numpy as np
                with np.load(cifar10_dest) as data:
                    x_data = data['x']
                    y_data = data['y']
                    print(f"✅ CIFAR-10 dataset copied to: {dest_data_dir}")
                    print(f"📊 CIFAR-10 dataset: {len(x_data)} samples, {x_data.shape[1:]} shape, {len(np.unique(y_data))} classes")
            except Exception as e:
                print(f"📊 CIFAR-10 dataset copied (could not read metadata: {e})")
        else:
            print(f"❌ Failed to prepare CIFAR-10 dataset")
            
    except Exception as e:
        print(f"❌ Error preparing CIFAR-10 data: {str(e)}")
        print("💡 Please check your internet connection for downloading CIFAR-10 dataset")

def create_improved_template_config(config: Dict[str, Any], project_dir: str = '.') -> Dict[str, Any]:
    """
    Create an improved template configuration with user-friendly comments and enhancements.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Enhanced configuration with user-friendly structure
    """
    # Start with the base configuration
    improved_config = config.copy()
    
    # Clean up file_path entries from configuration sections (they should only be in metadata)
    def remove_file_path_from_config_sections(config_dict):
        """Recursively remove file_path entries from configuration sections."""
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # Remove file_path if it exists and this is not a metadata section
                    if 'file_path' in value and key != 'metadata':
                        value.pop('file_path', None)
                    # Recursively process nested dictionaries
                    remove_file_path_from_config_sections(value)
    
    # Apply cleanup to configuration section only (preserve metadata)
    if 'configuration' in improved_config:
        remove_file_path_from_config_sections(improved_config['configuration'])
    
    # Add all available custom augmentation functions
    if 'data' in improved_config['configuration'] and 'augmentation' in improved_config['configuration']['data']:
        augmentation_functions = discover_custom_functions('./example_funcs/example_custom_augmentations.py')
        for func_name, func_info in augmentation_functions.items():
            augmentation_config = {
                'enabled': True,  # Enabled in template
                'function_name': func_name, 
                'probability': 0.5  # Add probability parameter for augmentations
            }
            # Add function-specific parameters
            params = func_info.get('parameters', {})
            if params:
                augmentation_config['parameters'] = params
            # Place functions directly under augmentation (no suffix)
            improved_config['configuration']['data']['augmentation'][func_name] = augmentation_config
    
    # Add all available custom preprocessing functions using direct placement format
    if 'data' in improved_config['configuration'] and 'preprocessing' in improved_config['configuration']['data']:
        preprocessing_functions = discover_custom_functions('./example_funcs/example_custom_preprocessing.py')
        for func_name, func_info in preprocessing_functions.items():
            preprocessing_config = {
                'enabled': True,  # Enabled in template like augmentations
                'function_name': func_name
            }
            # Add function-specific parameters
            params = func_info.get('parameters', {})
            if params:
                preprocessing_config['parameters'] = params
            
            # Place functions directly under preprocessing (consistent with augmentation)
            improved_config['configuration']['data']['preprocessing'][func_name] = preprocessing_config
    
    # Remove custom optimizer from metadata (if present) since it's rarely used
    if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
        if 'optimizers' in improved_config['metadata']['custom_functions']:
            del improved_config['metadata']['custom_functions']['optimizers']
    
    # Remove references to non-existent function files from metadata
    if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
        # Keep only functions that have actual generated files
        existing_functions = {}
        
        # Check which custom modules were actually generated
        from .script_generator import ScriptGenerator
        generator = ScriptGenerator()
        
        # These are the functions we know exist based on generated modules (parameters removed - they exist in configuration section)
        known_functions = {
            'models': [{
                'name': 'example_model',
                'file_path': './custom_modules/custom_models.py', 
                'function_name': 'example_model',
                'type': 'function'
            }],
            'data_loaders': [{
                'name': 'example_data_loader',
                'file_path': './custom_modules/custom_data_loaders.py',
                'function_name': 'example_data_loader', 
                'type': 'function'
            }],
            'loss_functions': [{
                'name': 'example_loss_1',
                'file_path': './custom_modules/custom_loss_functions.py',
                'function_name': 'example_loss_1',
                'type': 'function'
            },{
                'name': 'example_loss_2',
                'file_path': './custom_modules/custom_loss_functions.py',
                'function_name': 'example_loss_2',
                'type': 'function'
            }],
            'metrics': [{
                'name': 'example_metric_1',
                'file_path': './custom_modules/custom_metrics.py',
                'function_name': 'example_metric_1',
                'type': 'function'
            },
            {
                'name': 'example_metric_2',
                'file_path': './custom_modules/custom_metrics.py',
                'function_name': 'example_metric_2',
                'type': 'function'
            }],
            'callbacks': [{
                'name': 'ExampleCallbackClass1',
                'file_path': './custom_modules/custom_callbacks.py',
                'function_name': 'ExampleCallbackClass1',
                'type': 'class'
            },{
                'name': 'ExampleCallbackClass2',
                'file_path': './custom_modules/custom_callbacks.py',
                'function_name': 'ExampleCallbackClass2',
                'type': 'class'
            }],
            'augmentations': [{
                'name': 'example_augmentation_1',
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': 'example_augmentation_1',
                'type': 'function'
            },{
                'name': 'example_augmentation_2',
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': 'example_augmentation_2',
                'type': 'function'
            }],
            'preprocessing': [{
                'name': 'example_preprocessing_1',
                'file_path': './custom_modules/custom_preprocessing.py',
                'function_name': 'example_preprocessing_1',
                'type': 'function'
            },{
                'name': 'example_preprocessing_2',
                'file_path': './custom_modules/custom_preprocessing.py',
                'function_name': 'example_preprocessing_2',
                'type': 'function'
            }],
            'training_loops': [{
                'name': 'example_training_loop',
                'file_path': './custom_modules/custom_training_loops.py',
                'function_name': 'example_training_loop',
                'type': 'function'
            }]
        }
        
    # Update metadata to include custom functions (instead of None)
    if 'metadata' in improved_config:
        improved_config['metadata']['custom_functions'] = known_functions
        
    return improved_config

def extract_function_parameters(function_name: str, file_path: str, project_dir: str = '.', show_warnings: bool = True) -> Dict[str, Any]:
    """
    Extract function parameters from a custom function file.
    
    Args:
        function_name: Name of the function to extract parameters from
        file_path: Path to the file containing the function
        project_dir: Project directory for resolving relative paths
        show_warnings: Whether to show warnings when files are not found
        
    Returns:
        Dictionary of function parameters with default values
    """
    import inspect
    import importlib.util
    import os
    
    try:
        # Convert relative path to absolute using project directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_dir, file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            if show_warnings:
                print(f"⚠️ Function parameter extraction: File {file_path} not found")
            return {}
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        if spec is None or spec.loader is None:
            return {}
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        if not hasattr(module, function_name):
            if show_warnings:
                print(f"⚠️ Function {function_name} not found in {file_path}")
            return {}
        
        func = getattr(module, function_name)
        
        # Extract function signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            # Skip the first parameter (usually 'data' or 'model')
            if param_name in ['data', 'model', 'self', 'cls']:
                continue
            
            # Get default value
            if param.default != inspect.Parameter.empty:
                default_value = param.default
            else:
                # Provide sensible defaults based on parameter name
                default_value = get_parameter_default_value(param_name, param.annotation)
            
            parameters[param_name] = default_value
        
        return parameters
        
    except Exception as e:
        print(f"⚠️ Error extracting parameters from {function_name}: {str(e)}")
        return {}

def discover_custom_functions(file_path: str) -> Dict[str, Any]:
    """
    Dynamically discover and analyze custom functions from a Python file.
    
    Args:
        file_path: Path to the Python file containing custom functions
        
    Returns:
        Dictionary mapping function names to their information
    """
    import ast
    import inspect
    
    custom_functions = {}
    
    if not os.path.exists(file_path):
        return custom_functions
    
    try:
        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                
                # Skip private functions
                if func_name.startswith('_'):
                    continue
                
                # Extract function parameters and their defaults
                params = {}
                
                # Get function arguments
                args = node.args.args
                defaults = node.args.defaults or []
                
                # Skip the first parameter (data/image)
                if args:
                    param_args = args[1:]  # Skip first parameter
                    
                    # Match defaults with parameters (from right to left)
                    num_defaults = len(defaults)
                    num_params = len(param_args)
                    
                    for i, arg in enumerate(param_args):
                        param_name = arg.arg
                        
                        # Determine if this parameter has a default value
                        default_index = i - (num_params - num_defaults)
                        if default_index >= 0:
                            default_node = defaults[default_index]
                            if isinstance(default_node, ast.Constant):
                                default_value = default_node.value
                            elif isinstance(default_node, ast.Num):  # Python < 3.8 compatibility
                                default_value = default_node.n
                            elif isinstance(default_node, ast.Str):  # Python < 3.8 compatibility
                                default_value = default_node.s
                            else:
                                default_value = 0.5  # Fallback default
                            
                            params[param_name] = default_value
                
                # Extract docstring if available
                docstring = ast.get_docstring(node) or f"Custom function: {func_name}"
                
                custom_functions[func_name] = {
                    'parameters': params,
                    'docstring': docstring,
                    'file_path': file_path,
                    'function_name': func_name
                }
                
    except Exception as e:
        print(f"Warning: Error parsing {file_path}: {e}")
    
    return custom_functions

def get_parameter_default_value(param_name: str, param_annotation) -> Any:
    """Get a sensible default value for a parameter based on its name and type annotation."""
    # Common parameter name patterns and their defaults
    default_mappings = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'lr': 0.001,
        'epochs': 100,
        'dropout_rate': 0.5,
        'clip_limit': 2.0,
        'tile_grid_size': 8,
        'buffer_size': 10000,
        'image_size': [224, 224],
        'input_shape': [224, 224, 3],
        'num_classes': 1000,
        'shuffle': True,
        'augment': False,
        'enabled': False,
        'probability': 0.5,
        'patience': 10,
        'monitor': 'val_loss',
        'factor': 0.5,
        'min_lr': 1e-7,
        'initial_resolution': 32,
        'final_resolution': 224,
        'progression_schedule': 'linear'
    }
    
    # Check if parameter name matches known patterns
    for pattern, default in default_mappings.items():
        if pattern in param_name.lower():
            return default
    
    # Fall back to type-based defaults
    if param_annotation == int:
        return 1
    elif param_annotation == float:
        return 0.1
    elif param_annotation == bool:
        return False
    elif param_annotation == str:
        return ""
    elif param_annotation == list:
        return []
    else:
        return None

def generate_improved_yaml(config: Dict[str, Any]) -> str:
    """Generate user-friendly YAML with helpful comments."""
    yaml_lines = []
    
    # Header with instructions and options reference
    yaml_lines.extend([
        "# ModelGardener Configuration Template - Ready to run with custom functions and sample data",
        "",
        "# INSTRUCTIONS:",
        "# 1. Custom functions are configured in metadata section below",
        "# 2. Modify parameters below to customize training behavior",
        "# 3. Run training with: mg train -c config.yaml or python train.py",
        "",
        "# AVAILABLE OPTIONS REFERENCE:",
        "# - Optimizers: [Adam, SGD, RMSprop, Adagrad, AdamW, Adadelta, Adamax, Nadam, FTRL]",
        "# - Loss Functions: [Categorical Crossentropy, Sparse Categorical Crossentropy, Binary Crossentropy, Mean Squared Error, Mean Absolute Error, Focal Loss, Huber Loss]", 
        "# - Metrics: [Accuracy, Categorical Accuracy, Sparse Categorical Accuracy, Top-K Categorical Accuracy, Precision, Recall, F1 Score, AUC, Mean Squared Error, Mean Absolute Error]",
        "# - Training Loops: [Standard Training, Progressive Training, Curriculum Learning, Adversarial Training, Self-Supervised Training]",
        ""
    ])
    
    # Generate configuration section with comments
    configuration = config.get('configuration', {})
    yaml_lines.append("configuration:")
    
    # Task type
    task_type = configuration.get('task_type', 'image_classification')
    yaml_lines.append(f"  task_type: {task_type}")
    
    # Data section
    data_config = configuration.get('data', {})
    yaml_lines.append("  data:")
    yaml_lines.append(f"    train_dir: {data_config.get('train_dir', './data')}")
    yaml_lines.append(f"    val_dir: {data_config.get('val_dir', './data')}")
    
    # Add data loader section
    data_loader = data_config.get('data_loader', {})
    yaml_lines.extend([
        "    data_loader:",
        f"      selected_data_loader: {data_loader.get('selected_data_loader', 'Default')}",
        f"      use_for_train: {str(data_loader.get('use_for_train', True)).lower()}",
        f"      use_for_val: {str(data_loader.get('use_for_val', True)).lower()}",
        "      parameters:"
    ])
    
    params = data_loader.get('parameters', {})
    for key, value in params.items():
        yaml_lines.append(f"        {key}: {value}")
    
    # Preprocessing section with custom options
    preprocessing = data_config.get('preprocessing', {})
    yaml_lines.append("    preprocessing:")
    
    # Add both built-in and custom preprocessing functions
    custom_preprocessing_found = False
    for key, value in preprocessing.items():
        if isinstance(value, dict) and 'function_name' in value and 'file_path' in value:
            # This is a custom function - add comment if first one
            if not custom_preprocessing_found:
                yaml_lines.append("      # Custom preprocessing functions")
                custom_preprocessing_found = True
        else:
            # This is a built-in preprocessing option
            pass
        yaml_lines.append(f"      {key}:")
        add_nested_yaml(yaml_lines, value, 8)
            
    # Augmentation section with custom options
    augmentation = data_config.get('augmentation', {})
    yaml_lines.append("    augmentation:")
    yaml_lines.append("      # Built-in augmentation options")
    
    # Add both built-in and custom augmentation functions  
    custom_augmentation_found = False
    for key, value in augmentation.items():
        if isinstance(value, dict) and 'function_name' in value and 'file_path' in value:
            # This is a custom function - add comment if first one
            if not custom_augmentation_found:
                yaml_lines.append("      # Custom augmentation functions")
                custom_augmentation_found = True
        yaml_lines.append(f"      {key}:")
        add_nested_yaml(yaml_lines, value, 8)
            
    # Model section
    model_config = configuration.get('model', {})
    yaml_lines.append("  model:")
    yaml_lines.append(f"    model_family: {model_config.get('model_family', 'custom_model')}")
    yaml_lines.append(f"    model_name: {model_config.get('model_name', 'create_simple_cnn')}")
    
    # Model parameters
    model_params = model_config.get('model_parameters', {})
    yaml_lines.append("    model_parameters:")
    for key, value in model_params.items():
        if isinstance(value, dict):
            yaml_lines.append(f"      {key}:")
            add_nested_yaml(yaml_lines, value, 8)
        else:
            yaml_lines.append(f"      {key}: {value}")
            
    # Optimizer section with comment
    optimizer = model_config.get('optimizer', {})
    yaml_lines.append("    optimizer:")
    yaml_lines.append("      # Available optimizers: [Adam, SGD, RMSprop, Adagrad, AdamW, Adadelta, Adamax, Nadam, FTRL]")
    for key, value in optimizer.items():
        yaml_lines.append(f"      {key}:")
        add_nested_yaml(yaml_lines, value, 8)
        
    # Loss functions with comment
    loss_functions = model_config.get('loss_functions', {})
    yaml_lines.append("    loss_functions:")
    yaml_lines.append("      # Available loss functions: [Categorical Crossentropy, Sparse Categorical Crossentropy, Binary Crossentropy, Mean Squared Error, Mean Absolute Error, Focal Loss, Huber Loss]")
    for key, value in loss_functions.items():
        yaml_lines.append(f"      {key}:")
        add_nested_yaml(yaml_lines, value, 8)
        
    # Metrics with comment
    metrics = model_config.get('metrics', {})
    yaml_lines.append("    metrics:")
    yaml_lines.append("      # Available metrics: [Accuracy, Categorical Accuracy, Sparse Categorical Accuracy, Top-K Categorical Accuracy, Precision, Recall, F1 Score, AUC, Mean Squared Error, Mean Absolute Error]")
    for key, value in metrics.items():
        yaml_lines.append(f"      {key}:")
        add_nested_yaml(yaml_lines, value, 8)
        
    # Callbacks with custom option
    callbacks = model_config.get('callbacks', {})
    yaml_lines.append("    callbacks:")
    
    for key, value in callbacks.items():
        if key != 'Custom Callback':
            yaml_lines.append(f"      {key}:")
            add_nested_yaml(yaml_lines, value, 8)
    
    # Add custom callback
    if 'Custom Callback' in callbacks:
        yaml_lines.extend([
            "      # Custom callback (disabled - file not included in this template)",
            "      # To add: Create ./custom_modules/custom_callbacks.py with desired callbacks", 
            "      Custom Callback:"
        ])
        custom_callback = callbacks['Custom Callback']
        for key, value in custom_callback.items():
            yaml_lines.append(f"        {key}: {value}")
            
    # Training section
    training = configuration.get('training', {})
    yaml_lines.append("  training:")
    for key, value in training.items():
        if key == 'training_loop':
            yaml_lines.append("    training_loop:")
            yaml_lines.append("      # Available training strategies: [Standard Training, Progressive Training, Curriculum Learning, Adversarial Training, Self-Supervised Training]")
            for sub_key, sub_value in value.items():
                yaml_lines.append(f"      {sub_key}: {sub_value}")
        elif isinstance(value, dict):
            yaml_lines.append(f"    {key}:")
            add_nested_yaml(yaml_lines, value, 6)
        else:
            yaml_lines.append(f"    {key}: {value}")
            
    # Runtime section
    runtime = configuration.get('runtime', {})
    yaml_lines.append("  runtime:")
    for key, value in runtime.items():
        yaml_lines.append(f"    {key}: {value}")
        
    # Metadata section
    metadata = config.get('metadata', {})
    yaml_lines.append("metadata:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            yaml_lines.append(f"  {key}:")
            add_nested_yaml(yaml_lines, value, 4)
        else:
            yaml_lines.append(f"  {key}: {value}")
    
    return '\n'.join(yaml_lines)

def add_nested_yaml(yaml_lines: List[str], value: Any, indent_level: int):
    """Add nested YAML content with proper indentation."""
    indent = " " * indent_level
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                yaml_lines.append(f"{indent}{k}:")
                add_nested_yaml(yaml_lines, v, indent_level + 2)
            elif isinstance(v, list):
                yaml_lines.append(f"{indent}{k}:")
                for item in v:
                    if isinstance(item, dict):
                        yaml_lines.append(f"{indent}- name: {item.get('name', '')}")
                        for sub_k, sub_v in item.items():
                            if sub_k != 'name':  # name already added
                                if sub_k == 'parameters' and isinstance(sub_v, dict):
                                    # Add parameters as nested structure
                                    yaml_lines.append(f"{indent}  {sub_k}:")
                                    for param_k, param_v in sub_v.items():
                                        yaml_lines.append(f"{indent}    {param_k}: {param_v}")
                                else:
                                    yaml_lines.append(f"{indent}  {sub_k}: {sub_v}")
                    else:
                        yaml_lines.append(f"{indent}- {item}")
            else:
                yaml_lines.append(f"{indent}{k}: {v}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yaml_lines.append(f"{indent}- name: {item.get('name', '')}")
                for k, v in item.items():
                    if k != 'name':  # name already added
                        if k == 'parameters' and isinstance(v, dict):
                            # Add parameters as nested structure
                            yaml_lines.append(f"{indent}  {k}:")
                            for param_k, param_v in v.items():
                                yaml_lines.append(f"{indent}    {param_k}: {param_v}")
                        else:
                            yaml_lines.append(f"{indent}  {k}: {v}")
            else:
                yaml_lines.append(f"{indent}- {item}")
    else:
        yaml_lines.append(f"{indent}{value}")

def extract_preprocessing_parameters(obj) -> Dict[str, Any]:
    """Extract parameters from preprocessing function (supports wrapper pattern)."""
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            param_info = {}
            
            for param_name, param in sig.parameters.items():
                # For wrapper pattern, we want the configuration parameters (outer function)
                # Skip data/image parameters if they exist (traditional pattern)
                if param_name in ['data', 'image', 'img', 'input', 'x', 'array', 'tensor', 'label', 'labels']:
                    continue
                    
                param_details = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }
                
                # Infer parameter type
                if param.annotation != inspect.Parameter.empty:
                    param_details['type'] = str(param.annotation)
                elif param.default is not None:
                    param_details['type'] = type(param.default).__name__
                else:
                    param_details['type'] = 'Any'
                    
                param_info[param_name] = param_details
            
            # Extract function metadata
            function_info = {
                'name': obj.__name__,
                'type': 'function',
                'parameters': param_info,
                'signature': str(sig),
                'description': obj.__doc__.strip().split('\n')[0] if obj.__doc__ else f"Preprocessing function: {obj.__name__}"
            }
            
            return function_info
            
    except Exception:
        pass
    
    return {}

def extract_augmentation_parameters(func) -> Dict[str, Any]:
    """
    Extract parameters from an augmentation function (supports wrapper pattern).
    
    Args:
        func: The function to analyze
        
    Returns:
        Dictionary containing function information
    """
    import inspect
    
    try:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "Custom augmentation function"
        
        # Extract parameters
        parameters = {}
        for param_name, param in sig.parameters.items():
            # For wrapper pattern, we want the configuration parameters (outer function)
            # Skip data/image parameters if they exist (traditional pattern)
            if param_name in ['data', 'image', 'img', 'input', 'x', 'array', 'tensor', 'label', 'labels']:
                continue
            
            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
            }
            
            # Infer parameter type
            if param.annotation != inspect.Parameter.empty:
                param_info['type'] = str(param.annotation)
            elif param.default is not None:
                param_info['type'] = type(param.default).__name__
            else:
                param_info['type'] = 'Any'
            
            parameters[param_name] = param_info
        
        return {
            'name': func.__name__,
            'type': 'function',
            'description': doc.split('\n')[0] if doc else f"Custom augmentation: {func.__name__}",
            'parameters': parameters,
            'signature': str(sig)
        }
            
    except Exception:
        pass
    
    return {}

def extract_callback_parameters(obj) -> Dict[str, Any]:
    """
    Extract parameters from a callback class (supports keras.callbacks.Callback).
    
    Args:
        obj: The callback class
        
    Returns:
        Dict containing parameter information
    """
    import inspect
    
    try:
        if inspect.isclass(obj):
            # Get constructor parameters
            init_method = getattr(obj, '__init__', None)
            params = {}
            
            if init_method:
                sig = inspect.signature(init_method)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                        
                    param_info = {
                        'name': param_name,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                    }
                    
                    if param.annotation != inspect.Parameter.empty:
                        param_info['type'] = str(param.annotation)
                    elif param.default is not None:
                        param_info['type'] = type(param.default).__name__
                    else:
                        param_info['type'] = 'Any'
                    
                    params[param_name] = param_info
            
            return {
                'name': obj.__name__,
                'type': 'class',
                'parameters': params,
                'signature': f"class {obj.__name__}",
                'description': inspect.getdoc(obj) or f"Callback class: {obj.__name__}"
            }
            
        elif inspect.isfunction(obj):
            # If it's a function that returns a callback, analyze its parameters
            sig = inspect.signature(obj)
            params = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'required': param.default == inspect.Parameter.empty,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None
                }
                
                if param.annotation != inspect.Parameter.empty:
                    param_info['type'] = str(param.annotation)
                elif param.default is not None:
                    param_info['type'] = type(param.default).__name__
                else:
                    param_info['type'] = 'Any'
                
                params[param_name] = param_info
            
            return {
                'name': obj.__name__,
                'type': 'function',
                'parameters': params,
                'signature': str(sig),
                'description': inspect.getdoc(obj) or f"Callback function: {obj.__name__}"
            }
            
    except Exception:
        return {
            'type': 'unknown',
            'parameters': {},
            'signature': '',
            'description': f"Callback: {getattr(obj, '__name__', 'Unknown')}"
        }
    
    return {}

def generate_cifar10_subset(samples_per_class=500, output_path="example_data/cifar10.npz", 
                          random_seed=42, verbose=True):
    """
    Generate a subset of CIFAR-10 dataset with specified samples per class.
    
    Args:
        samples_per_class (int): Number of samples to include for each class
        output_path (str): Output path for the generated NPZ file
        random_seed (int): Random seed for reproducibility
        verbose (bool): Whether to print detailed information
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import numpy as np
        import tensorflow as tf
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        if verbose:
            print(f"🔄 Generating CIFAR-10 dataset...")
        
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Combine train and test data
        x_data = np.concatenate([x_train, x_test], axis=0)
        y_data = np.concatenate([y_train, y_test], axis=0).flatten()
        
        if verbose:
            print(f"📊 Full CIFAR-10 dataset: {x_data.shape[0]} samples")
            print(f"🎯 Target: {samples_per_class} samples per class (10 classes)")
        
        # Check if we have enough samples for each class
        for class_id in range(10):
            class_count = np.sum(y_data == class_id)
            if class_count < samples_per_class:
                if verbose:
                    print(f"❌ Error: Class {class_id} only has {class_count} samples, but {samples_per_class} requested")
                return False
        
        # Initialize arrays for subset
        subset_x = []
        subset_y = []
        
        # Extract samples for each class
        for class_id in range(10):
            class_indices = np.where(y_data == class_id)[0]
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
            
            subset_x.append(x_data[selected_indices])
            subset_y.append(y_data[selected_indices])
            
            if verbose:
                print(f"✅ Class {class_id}: {len(selected_indices)} samples selected")
        
        # Combine all selected samples
        subset_x = np.concatenate(subset_x, axis=0)
        subset_y = np.concatenate(subset_y, axis=0)
        
        # Shuffle the dataset
        shuffle_indices = np.random.permutation(len(subset_x))
        subset_x = subset_x[shuffle_indices]
        subset_y = subset_y[shuffle_indices]
        
        if verbose:
            print(f"📈 Final dataset shape: {subset_x.shape}")
            print(f"🔢 Final labels shape: {subset_y.shape}")
            print(f"🎨 Unique classes: {len(np.unique(subset_y))}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save as NPZ file
        np.savez(output_path, x=subset_x, y=subset_y)
        
        if verbose:
            print(f"✅ CIFAR-10 subset saved to: {output_path}")
        
        # Verify the saved file
        if verify_dataset(output_path, expected_samples=samples_per_class * 10, verbose=verbose):
            return True
        else:
            if verbose:
                print("❌ Dataset verification failed")
            return False
            
    except Exception as e:
        if verbose:
            print(f"❌ Error generating dataset: {str(e)}")
        return False

def verify_dataset(dataset_path, expected_samples=None, verbose=True):
    """
    Verify a dataset NPZ file.
    
    Args:
        dataset_path (str): Path to the NPZ file
        expected_samples (int): Expected total number of samples
        verbose (bool): Whether to print verification details
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    try:
        import numpy as np
        
        if not os.path.exists(dataset_path):
            if verbose:
                print(f"❌ Dataset file not found: {dataset_path}")
            return False
        
        loaded_data = np.load(dataset_path)
        
        if 'x' not in loaded_data or 'y' not in loaded_data:
            if verbose:
                print("❌ Dataset missing required keys 'x' and 'y'")
            return False
        
        x_data = loaded_data['x']
        y_data = loaded_data['y']
        
        if verbose:
            print(f"🔍 Verification:")
            print(f"   - X shape: {x_data.shape}")
            print(f"   - Y shape: {y_data.shape}")
            print(f"   - Classes: {np.unique(y_data)}")
            print(f"   - Samples per class: {[np.sum(y_data == i) for i in range(10)]}")
        
        # Basic checks
        if len(x_data) != len(y_data):
            if verbose:
                print("❌ Mismatch between X and Y data lengths")
            return False
        
        if expected_samples and len(x_data) != expected_samples:
            if verbose:
                print(f"❌ Expected {expected_samples} samples, got {len(x_data)}")
            return False
        
        # Check if we have all 10 classes
        unique_classes = np.unique(y_data)
        if len(unique_classes) != 10 or not np.array_equal(unique_classes, np.arange(10)):
            if verbose:
                print(f"❌ Expected classes 0-9, got {unique_classes}")
            return False
        
        if verbose:
            print("✅ Dataset verification passed")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Error verifying dataset: {str(e)}")
        return False

def ensure_cifar10_dataset(temp_dir="temp", target_path="example_data/cifar10.npz", 
                          samples_per_class=500, force_regenerate=False, verbose=True):
    """
    Ensure CIFAR-10 dataset exists, using temp cache or generating new one.
    
    Args:
        temp_dir (str): Temporary directory to cache datasets
        target_path (str): Target path for the dataset
        samples_per_class (int): Number of samples per class (fixed to 500)
        force_regenerate (bool): Force regeneration even if cache exists
        verbose (bool): Whether to print detailed information
    
    Returns:
        bool: True if dataset is ready, False otherwise
    """
    import shutil
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define cache path in temp directory (fixed filename for 500 samples)
    cache_filename = "cifar10_500samples_per_class.npz"
    cache_path = os.path.join(temp_dir, cache_filename)
    
    # Check if cached dataset exists and is valid
    if not force_regenerate and os.path.exists(cache_path):
        if verbose:
            print(f"🔍 Checking cached dataset: {cache_path}")
        
        if verify_dataset(cache_path, expected_samples=samples_per_class * 10, verbose=False):
            if verbose:
                print(f"✅ Valid cached dataset found")
            
            # Copy from cache to target
            target_dir = os.path.dirname(target_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            
            shutil.copy2(cache_path, target_path)
            if verbose:
                print(f"📋 Dataset copied to: {target_path}")
            return True
        else:
            if verbose:
                print(f"⚠️ Cached dataset is invalid, will regenerate")
    
    # Generate new dataset
    if verbose:
        print(f"🚀 Generating new CIFAR-10 dataset ({samples_per_class} samples per class)")
    
    # Generate to cache first
    if generate_cifar10_subset(samples_per_class, cache_path, verbose=verbose):
        # Copy to target location
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        
        shutil.copy2(cache_path, target_path)
        if verbose:
            print(f"📋 Dataset copied to: {target_path}")
        return True
    else:
        if verbose:
            print("❌ Failed to generate dataset")
        return False
