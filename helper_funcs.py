
import os
import inspect
import importlib.util
from typing import Any, Dict, List, Tuple
# Import script generator
try:
    from script_generator import ScriptGenerator
except ImportError:
    print("Warning: ScriptGenerator not available")
    ScriptGenerator = None

def is_model_function(obj, name: str) -> bool:
        """Check if an object is likely a model function."""
        try:
            if inspect.isfunction(obj):
                # Check if function signature includes typical model parameters
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                
                # Look for common model function patterns
                model_indicators = [
                    'input_shape', 'num_classes', 'classes', 'inputs', 'outputs',
                    'input_tensor', 'model', 'layers', 'activation'
                ]
                
                # Check if function has model-related parameters
                has_model_params = any(indicator in ' '.join(params).lower() for indicator in model_indicators)
                
                # Check function name patterns
                name_lower = name.lower()
                name_patterns = [
                    'create', 'build', 'get', 'make', 'model', 'net', 'network',
                    'cnn', 'resnet', 'efficientnet', 'mobilenet', 'unet'
                ]
                has_model_name = any(pattern in name_lower for pattern in name_patterns)
                
                return has_model_params or has_model_name
                
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
    Check if an object is a valid data loader function or class.
    
    Args:
        obj: The object to check
        name: Name of the object
        
    Returns:
        bool: True if it's a valid data loader function/class
    """
    import inspect
    
    # Skip private functions, imports, and common utilities
    if name.startswith('_') or name in ['tf', 'tensorflow', 'np', 'numpy', 'os', 'sys', 'train_test_split', 'pd', 'pandas']:
        return False
    
    # Skip objects from imported modules
    if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
        if not obj.__module__.startswith('custom') and 'custom' not in obj.__module__:
            return False
        
    try:
        if inspect.isfunction(obj):
            # Check function signature for data loader patterns
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Must have data-related parameters
            data_loader_indicators = [
                'data_dir', 'batch_size', 'split', 'train_dir', 'val_dir',
                'dataset', 'images', 'labels', 'data_path', 'file_path',
                'csv_path', 'npz_path', 'tfrecord_path'
            ]
            
            # Check if function has data loader-like parameters
            has_data_params = any(indicator in param.lower() for param in params for indicator in data_loader_indicators)
            
            # Must return tf.data.Dataset or similar
            return_annotation = sig.return_annotation
            valid_return_type = False
            if return_annotation != inspect.Signature.empty:
                return_type_str = str(return_annotation)
                if 'tf.data.Dataset' in return_type_str or 'Dataset' in return_type_str or 'DatasetV2' in return_type_str:
                    valid_return_type = True
            
            # Check docstring for data loader keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            data_loader_keywords = ['data loader', 'dataset', 'load data', 'data loading']
            has_data_keywords = any(keyword in docstring_lower for keyword in data_loader_keywords)
            
            # Exclude simple utility functions like invalid_data_function
            if len(params) < 2:
                return False
            
            # Must have either data params + valid return type OR data keywords + data params
            return (has_data_params and valid_return_type) or (has_data_keywords and has_data_params)
            
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
            
            # Check constructor parameters
            init_method = getattr(obj, '__init__', None)
            has_data_params = False
            if init_method:
                try:
                    sig = inspect.signature(init_method)
                    params = list(sig.parameters.keys())
                    data_indicators = ['data_dir', 'batch_size', 'data_path', 'npz_path', 'csv_path']
                    has_data_params = any(indicator in param.lower() for param in params for indicator in data_indicators)
                except:
                    pass
            
            return (has_loader_methods or has_class_keywords) and has_data_params
            
    except Exception:
        return False
        
    return False

def is_loss_function( obj, name: str) -> bool:
    """
    Check if an object is a valid loss function.
    
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
    
    # Skip objects from imported modules (except custom ones)
    if hasattr(obj, '__module__') and obj.__module__ not in [None, '__main__']:
        if not obj.__module__.startswith('custom') and 'custom' not in obj.__module__:
            return False
        
    try:
        if inspect.isfunction(obj):
            # Check function signature for loss function patterns
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())
            
            # Must have typical loss function parameters
            loss_indicators = ['y_true', 'y_pred', 'true', 'pred', 'target', 'prediction', 'labels', 'logits']
            has_loss_params = len(params) >= 2 and any(indicator in param.lower() for param in params for indicator in loss_indicators)
            
            # Check docstring for loss function keywords
            docstring = inspect.getdoc(obj) or ""
            docstring_lower = docstring.lower()
            loss_keywords = ['loss', 'cost', 'error', 'distance', 'divergence']
            has_loss_keywords = any(keyword in docstring_lower for keyword in loss_keywords)
            
            return has_loss_params or has_loss_keywords
            
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
        
        has_custom_aug = 'Custom Augmentation' in data_config.get('augmentation', {})
        has_custom_prep = 'Custom Preprocessing' in data_config.get('preprocessing', {})
        has_custom_callback = 'Custom Callback' in model_config.get('callbacks', {})
        
        return has_custom_aug or has_custom_prep or has_custom_callback
    except:
        return False

def is_preprocessing_function( obj, name: str) -> bool:
    """Check if an object is likely a preprocessing function (including wrapper pattern)."""
    try:
        if inspect.isfunction(obj):
            # Check function name patterns
            name_lower = name.lower()
            name_patterns = [
                'preprocess', 'process', 'transform', 'normalize', 'resize', 'augment',
                'enhance', 'filter', 'convert', 'scale', 'adjust', 'crop', 'pad', 'tf_', 'cv_',
                'gamma', 'histogram', 'edge', 'adaptive', 'enhancement', 'correction'
            ]
            has_preprocessing_name = any(pattern in name_lower for pattern in name_patterns)
            
            # Check docstring for preprocessing-related keywords
            has_preprocessing_keywords = False
            if obj.__doc__:
                docstring_lower = obj.__doc__.lower()
                preprocessing_keywords = ['preprocess', 'transform', 'normalize', 'resize', 'enhance', 'filter', 'gamma', 'histogram', 'edge', 'contrast', 'brightness', 'correction', 'adaptive']
                has_preprocessing_keywords = any(keyword in docstring_lower for keyword in preprocessing_keywords)
            
            # Accept functions with preprocessing names/keywords regardless of signature
            # This supports both wrapper pattern and legacy pattern
            if has_preprocessing_name or has_preprocessing_keywords:
                return True
            
    except Exception:
        pass
    
    return False

def is_augmentation_function( obj, name: str) -> bool:
    """
    Check if an object is a valid augmentation function (including wrapper pattern).
    
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
            'distort', 'elastic', 'tf_', 'cv_', 'random', 'transform'
        ]
        has_augmentation_name = any(pattern in name_lower for pattern in augmentation_patterns)
        
        # Check docstring for augmentation-related keywords  
        has_augmentation_keywords = False
        if obj.__doc__:
            docstring_lower = obj.__doc__.lower()
            augmentation_keywords = ['augment', 'random', 'flip', 'rotate', 'brightness', 'contrast', 'blur', 'noise', 'crop', 'zoom', 'distort', 'transform', 'color', 'hue', 'saturation']
            has_augmentation_keywords = any(keyword in docstring_lower for keyword in augmentation_keywords)
        
        # Accept functions with augmentation names/keywords regardless of signature
        # This supports both wrapper pattern and legacy pattern
        if has_augmentation_name or has_augmentation_keywords:
            return True
            
    except Exception:
        return False
    
    return False

def is_callback_function(obj, name: str) -> bool:
    """
    Check if an object is a valid callback function or class.
    
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
        if inspect.isfunction(obj) or inspect.isclass(obj):
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
            has_callback_keywords = False
            if hasattr(obj, '__doc__') and obj.__doc__:
                docstring_lower = obj.__doc__.lower()
                callback_keywords = [
                    'callback', 'epoch', 'batch', 'training', 'monitor', 'metric',
                    'checkpoint', 'early stopping', 'learning rate', 'tensorboard',
                    'csv logger', 'progress', 'history', 'validation'
                ]
                has_callback_keywords = any(keyword in docstring_lower for keyword in callback_keywords)
            
            # For classes, check if it inherits from Keras callback
            if inspect.isclass(obj):
                try:
                    import tensorflow as tf
                    if issubclass(obj, tf.keras.callbacks.Callback):
                        return True
                except (ImportError, TypeError):
                    pass
            
            # For functions, check if it returns a callback-like object
            if inspect.isfunction(obj):
                try:
                    sig = inspect.signature(obj)
                    return_annotation = sig.return_annotation
                    if return_annotation != inspect.Signature.empty:
                        return_type_str = str(return_annotation)
                        if 'callback' in return_type_str.lower():
                            return True
                except Exception:
                    pass
            
            return has_callback_name or has_callback_keywords
                
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
    Extract parameters from a loss function.
    
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
                # Skip y_true, y_pred parameters as they are provided during training
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
    Copy CIFAR-10 NPZ dataset to the project directory.
    
    Args:
        project_dir: Target project directory
    """
    import shutil
    
    # Define source and destination paths
    source_data_dir = os.path.join(os.path.dirname(__file__), 'example_data')
    dest_data_dir = os.path.join(project_dir, 'data')
    cifar10_source = os.path.join(source_data_dir, 'cifar10.npz')
    cifar10_dest = os.path.join(dest_data_dir, 'cifar10.npz')
    
    try:
        # Create data directory
        os.makedirs(dest_data_dir, exist_ok=True)
        
        # Copy CIFAR-10 NPZ file
        if os.path.exists(cifar10_source):
            shutil.copy2(cifar10_source, cifar10_dest)
            print(f"✅ CIFAR-10 dataset copied to: {dest_data_dir}")
            
            # Load and show dataset info
            try:
                import numpy as np
                with np.load(cifar10_source) as data:
                    x_data = data['x']
                    y_data = data['y']
                    print(f"📊 CIFAR-10 dataset: {len(x_data)} samples, {x_data.shape[1:]} shape, {len(np.unique(y_data))} classes")
            except Exception as e:
                print(f"📊 CIFAR-10 dataset copied (could not read metadata: {e})")
                
        else:
            print(f"⚠️ Warning: CIFAR-10 dataset not found at {cifar10_source}")
            print("� Please run test_generate_subset.py to generate the CIFAR-10 dataset")
            
    except Exception as e:
        print(f"❌ Error copying CIFAR-10 data: {str(e)}")
        print("💡 Please ensure CIFAR-10 dataset is available in example_data/cifar10.npz")

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
    
    # Add all available custom augmentation functions
    if 'data' in improved_config['configuration'] and 'augmentation' in improved_config['configuration']['data']:
        augmentation_functions = discover_custom_functions('./example_funcs/example_custom_augmentations.py')
        for func_name, func_info in augmentation_functions.items():
            display_name = f"{func_name.replace('_', ' ').title()} (custom)"
            augmentation_config = {
                'enabled': False,
                'function_name': func_name, 
                'file_path': './custom_modules/custom_augmentations.py'
            }
            # Add function-specific parameters
            augmentation_config.update(func_info.get('parameters', {}))
            improved_config['configuration']['data']['augmentation'][display_name] = augmentation_config
    
    # Add all available custom preprocessing functions
    if 'data' in improved_config['configuration'] and 'preprocessing' in improved_config['configuration']['data']:
        preprocessing_functions = discover_custom_functions('./example_funcs/example_custom_preprocessing.py')
        for func_name, func_info in preprocessing_functions.items():
            display_name = f"{func_name.replace('_', ' ').title()} (custom)"
            preprocessing_config = {
                'enabled': False,
                'function_name': func_name,
                'file_path': './custom_modules/custom_preprocessing.py'
            }
            # Add function-specific parameters
            preprocessing_config.update(func_info.get('parameters', {}))
            improved_config['configuration']['data']['preprocessing'][display_name] = preprocessing_config
    
    # Add custom callback option
    if 'model' in improved_config['configuration'] and 'callbacks' in improved_config['configuration']['model']:
        improved_config['configuration']['model']['callbacks']['Custom Callback'] = {
            'enabled': False,
            'callback_name': 'custom_callback_name',
            'file_path': './custom_modules/custom_callbacks.py'
        }
    
    # Remove custom optimizer from metadata (if present) since it's rarely used
    if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
        if 'optimizers' in improved_config['metadata']['custom_functions']:
            del improved_config['metadata']['custom_functions']['optimizers']
    
    # Remove references to non-existent function files from metadata
    if 'metadata' in improved_config and 'custom_functions' in improved_config['metadata']:
        # Keep only functions that have actual generated files
        existing_functions = {}
        
        # Check which custom modules were actually generated
        from script_generator import ScriptGenerator
        generator = ScriptGenerator()
        
        # These are the functions we know exist based on generated modules with their parameters
        known_functions = {
            'models': [{
                'name': 'create_simple_cnn',
                'file_path': './custom_modules/custom_models.py', 
                'function_name': 'create_simple_cnn',
                'type': 'function',
                'parameters': extract_function_parameters('create_simple_cnn', './custom_modules/custom_models.py', project_dir, show_warnings=False)
            }],
            'data_loaders': [{
                'name': 'Custom_load_cifar10_npz_data',
                'file_path': './custom_modules/custom_data_loaders.py',
                'function_name': 'Custom_load_cifar10_npz_data', 
                'type': 'function',
                'parameters': extract_function_parameters('Custom_load_cifar10_npz_data', './custom_modules/custom_data_loaders.py', project_dir, show_warnings=False)
            }],
            'loss_functions': [{
                'name': 'dice_loss',
                'file_path': './custom_modules/custom_loss_functions.py',
                'function_name': 'dice_loss',
                'type': 'function',
                'parameters': extract_function_parameters('dice_loss', './custom_modules/custom_loss_functions.py', project_dir, show_warnings=False)
            }],
            'optimizers': [{
                'name': 'adaptive_adam',
                'file_path': './custom_modules/custom_optimizers.py',
                'function_name': 'adaptive_adam',
                'type': 'function',
                'parameters': extract_function_parameters('adaptive_adam', './custom_modules/custom_optimizers.py', project_dir, show_warnings=False)
            }],
            'metrics': [{
                'name': 'balanced_accuracy',
                'file_path': './custom_modules/custom_metrics.py',
                'function_name': 'balanced_accuracy',
                'type': 'function',
                'parameters': extract_function_parameters('balanced_accuracy', './custom_modules/custom_metrics.py', project_dir, show_warnings=False)
            }],
            'callbacks': [{
                'name': 'MemoryUsageMonitor',
                'file_path': './custom_modules/custom_callbacks.py',
                'function_name': 'MemoryUsageMonitor',
                'type': 'class',
                'parameters': extract_function_parameters('MemoryUsageMonitor', './custom_modules/custom_callbacks.py', project_dir, show_warnings=False)
            }],
            'augmentations': [{
                'name': 'color_shift',
                'file_path': './custom_modules/custom_augmentations.py',
                'function_name': 'color_shift',
                'type': 'function',
                'parameters': extract_function_parameters('color_shift', './custom_modules/custom_augmentations.py', project_dir, show_warnings=False)
            }],
            'preprocessing': [{
                'name': 'adaptive_histogram_equalization',
                'file_path': './custom_modules/custom_preprocessing.py',
                'function_name': 'adaptive_histogram_equalization',
                'type': 'function',
                'parameters': extract_function_parameters('adaptive_histogram_equalization', './custom_modules/custom_preprocessing.py', project_dir, show_warnings=False)
            }],
            'training_loops': [{
                'name': 'progressive_training_loop',
                'file_path': './custom_modules/custom_training_loops.py',
                'function_name': 'progressive_training_loop',
                'type': 'function',
                'parameters': extract_function_parameters('progressive_training_loop', './custom_modules/custom_training_loops.py', project_dir, show_warnings=False)
            }]
        }
        
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
        "# 1. Sample data has been copied to ./data/ directory with 3 classes", 
        "# 2. Custom functions are configured in metadata section below",
        "# 3. Modify parameters below to customize training behavior",
        "# 4. Run training with: python train.py",
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
    
    # Standard preprocessing options (non-custom)
    for key, value in preprocessing.items():
        if not key.endswith('(custom)'):
            yaml_lines.append(f"      {key}:")
            add_nested_yaml(yaml_lines, value, 8)
    
    # Add custom preprocessing functions
    custom_preprocessing_found = False
    for key, value in preprocessing.items():
        if key.endswith('(custom)'):
            if not custom_preprocessing_found:
                yaml_lines.append("      # Custom preprocessing functions (disabled by default)")
                custom_preprocessing_found = True
            yaml_lines.append(f"      {key}:")
            for sub_key, sub_value in value.items():
                yaml_lines.append(f"        {sub_key}: {sub_value}")
            
    # Augmentation section with custom options
    augmentation = data_config.get('augmentation', {})
    yaml_lines.append("    augmentation:")
    yaml_lines.append("      # Built-in augmentation options")
    
    # Standard augmentation options (non-custom)
    for key, value in augmentation.items():
        if not key.endswith('(custom)'):
            yaml_lines.append(f"      {key}:")
            add_nested_yaml(yaml_lines, value, 8)
    
    # Add custom augmentation functions
    custom_augmentation_found = False
    for key, value in augmentation.items():
        if key.endswith('(custom)'):
            if not custom_augmentation_found:
                yaml_lines.append("      # Custom augmentation functions (disabled by default)")
                custom_augmentation_found = True
            yaml_lines.append(f"      {key}:")
            for sub_key, sub_value in value.items():
                yaml_lines.append(f"        {sub_key}: {sub_value}")
            
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
    """Extract parameters from preprocessing function."""
    try:
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            param_info = {}
            
            for param_name, param in sig.parameters.items():
                # Skip the first parameter (usually 'data', 'image', etc.)
                if param_name in ['data', 'image', 'img', 'input', 'x', 'array', 'tensor']:
                    continue
                    
                param_details = {'type': 'str', 'default': None}
                if param.default != inspect.Parameter.empty:
                    param_details['default'] = param.default
                    # Infer type from default value
                    if isinstance(param.default, bool):
                        param_details['type'] = 'bool'
                    elif isinstance(param.default, int):
                        param_details['type'] = 'int'
                    elif isinstance(param.default, float):
                        param_details['type'] = 'float'
                    elif isinstance(param.default, (list, tuple)):
                        param_details['type'] = 'list'
                param_info[param_name] = param_details
            
            # Extract function metadata
            function_info = {
                'name': obj.__name__,
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
    Extract parameters from an augmentation function.
    
    Args:
        func: The function to analyze
        
    Returns:
        Dictionary containing function information
    """
    import inspect
    import ast
    
    try:
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "Custom augmentation function"
        
        # Extract parameters (skip first parameter - image/data)
        parameters = {}
        param_names = list(sig.parameters.keys())[1:]  # Skip first parameter
        
        for param_name in param_names:
            param = sig.parameters[param_name]
            param_info = {'name': param_name}
            
            # Get type annotation if available
            if param.annotation != inspect.Parameter.empty:
                param_info['type'] = param.annotation.__name__ if hasattr(param.annotation, '__name__') else 'str'
            else:
                # Try to infer type from default value
                if param.default != inspect.Parameter.empty:
                    param_info['type'] = type(param.default).__name__
                else:
                    param_info['type'] = 'str'
            
            # Get default value
            if param.default != inspect.Parameter.empty:
                param_info['default'] = param.default
            else:
                # Set sensible defaults based on type and name
                if param_info['type'] == 'bool':
                    param_info['default'] = True
                elif param_info['type'] in ['int', 'float']:
                    if 'probability' in param_name.lower():
                        param_info['default'] = 0.5
                    else:
                        param_info['default'] = 1.0 if param_info['type'] == 'float' else 1
                else:
                    param_info['default'] = ""
            
            parameters[param_name] = param_info
        
        return {
            'description': doc.split('\n')[0] if doc else f"Custom augmentation: {func.__name__}",
            'parameters': parameters,
            'signature': str(sig)
        }
            
    except Exception:
        pass
    
    return {}
