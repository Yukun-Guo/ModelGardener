"""
Model configuration module for ModelGardener CLI.
"""

import inspect
import importlib.util
import os
from typing import Dict, Any, List, Tuple, Optional
from .base_config import BaseConfig


class ModelConfig(BaseConfig):
    """Model configuration handler."""
    
    def __init__(self):
        super().__init__()
        self.available_models = {
            'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
            'efficientnet': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 
                           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7'],
            'mobilenet': ['MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large'],
            'vgg': ['VGG16', 'VGG19'],
            'densenet': ['DenseNet121', 'DenseNet169', 'DenseNet201'],
            'inception': ['InceptionV3', 'InceptionResNetV2'],
            'xception': ['Xception'],
            'unet': ['UNet','ResUNet'],
            'custom': ['CustomModel']
        }

    def _is_model_function(self, obj, name: str) -> bool:
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

    def _extract_model_parameters(self, obj) -> Dict[str, Any]:
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

    def analyze_custom_model_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Analyze a Python file to extract custom model functions."""
        try:
            if not os.path.exists(file_path):
                return False, {"error": f"File does not exist: {file_path}"}
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_model", file_path)
            if spec is None or spec.loader is None:
                return False, {"error": f"Cannot load module from: {file_path}"}
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find model functions/classes
            found_models = {}
            for name in dir(module):
                if name.startswith('_'):
                    continue
                    
                obj = getattr(module, name)
                if self._is_model_function(obj, name):
                    model_info = {
                        'name': name,
                        'type': 'function' if inspect.isfunction(obj) else 'class',
                        'file_path': file_path,
                        'parameters': self._extract_model_parameters(obj)
                    }
                    
                    # Add signature info for functions
                    if inspect.isfunction(obj):
                        try:
                            sig = inspect.signature(obj)
                            model_info['signature'] = str(sig)
                        except Exception:
                            model_info['signature'] = 'N/A'
                    
                    # Add docstring if available
                    if obj.__doc__:
                        model_info['description'] = obj.__doc__.strip().split('\n')[0]
                    else:
                        model_info['description'] = f"Custom {model_info['type']}: {name}"
                    
                    found_models[name] = model_info
            
            if not found_models:
                return False, {"error": "No valid model functions found in the file"}
            
            return True, found_models
            
        except Exception as e:
            return False, {"error": f"Failed to analyze file: {str(e)}"}

    def interactive_custom_model_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Interactive selection of custom model from analyzed file."""
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return None, {}
            
        success, analysis_result = self.analyze_custom_model_file(file_path)
        
        if not success:
            self.print_error(f"Error analyzing file: {analysis_result.get('error', 'Unknown error')}")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} custom model(s) in {os.path.basename(file_path)}")
        
        # Create choices for inquirer
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} (function)"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = f"{name}"
            choices.append((choice_text, name))
        
        # Let user select the model
        selected = inquirer.list_input(
            "Select custom model to use",
            choices=[choice[0] for choice in choices],
            default=choices[0][0] if choices else None
        )
        
        # Find the selected model name
        selected_name = None
        for choice_text, name in choices:
            if choice_text == selected:
                selected_name = name
                break
        
        if selected_name and selected_name in analysis_result:
            return selected_name, analysis_result[selected_name]
        
        return None, {}

    def analyze_model_outputs(self, config: Dict[str, Any]) -> Tuple[int, List[str]]:
        """
        Analyze model configuration to determine number of outputs and their names.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Tuple of (number_of_outputs, list_of_output_names)
        """
        try:
            model_config = config.get('configuration', {}).get('model', {})
            model_family = model_config.get('model_family', 'resnet')
            
            if model_family == 'custom':
                # Custom model - need to analyze the actual model
                model_name = model_config.get('model_name', '')
                model_params = model_config.get('model_parameters', {})
                file_path = model_params.get('file_path', '')
                
                if file_path and os.path.exists(file_path):
                    return self._analyze_custom_model_outputs(file_path, model_name, model_params)
                else:
                    return 1, ['output']
            else:
                # Standard models typically have single output
                return 1, ['predictions']
                
        except Exception as e:
            self.print_warning(f"Error analyzing model outputs: {e}")
            return 1, ['output']

    def _analyze_custom_model_outputs(self, file_path: str, function_name: str, 
                                    model_params: Dict[str, Any]) -> Tuple[int, List[str]]:
        """
        Analyze custom model file to determine outputs.
        
        Args:
            file_path: Path to custom model file
            function_name: Name of the model function
            model_params: Model parameters
            
        Returns:
            Tuple of (number_of_outputs, list_of_output_names)
        """
        try:
            # Load the module
            module = self.load_custom_module(file_path, 'custom_model_analysis')
            if module is None:
                return 1, ['output']
            
            # Get the model function
            model_func = getattr(module, function_name, None)
            if model_func is None:
                return 1, ['output']
            
            # Try to analyze the function source code
            return self._analyze_model_source_code(model_func)
            
        except Exception as e:
            self.print_warning(f"Error analyzing custom model: {e}")
            return 1, ['output']

    def _analyze_model_source_code(self, model_func) -> Tuple[int, List[str]]:
        """
        Analyze model function source code to find output layers.
        
        Args:
            model_func: Model function to analyze
            
        Returns:
            Tuple of (number_of_outputs, list_of_output_names)
        """
        try:
            source = inspect.getsource(model_func)
            lines = source.split('\n')
            
            output_names = []
            
            # Look for patterns that indicate multiple outputs
            for line in lines:
                line_stripped = line.strip().lower()
                
                # Look for output layer definitions
                if any(pattern in line_stripped for pattern in [
                    'dense(', 'output', 'predictions', 'classifier', 'logits'
                ]):
                    # Try to extract variable name
                    if '=' in line:
                        var_name = line.split('=')[0].strip()
                        if var_name and not var_name.startswith('#'):
                            # Clean up variable name
                            var_name = var_name.split()[-1]
                            if var_name.isidentifier():
                                output_names.append(var_name)
                
                # Look for Model() constructor with multiple outputs
                if 'model(' in line_stripped and 'outputs=' in line_stripped:
                    # Try to extract output list
                    outputs_part = line_stripped.split('outputs=')[1]
                    if '[' in outputs_part and ']' in outputs_part:
                        # Multiple outputs in list format
                        outputs_str = outputs_part.split('[')[1].split(']')[0]
                        output_vars = [name.strip() for name in outputs_str.split(',')]
                        output_names.extend([name for name in output_vars if name and name.isidentifier()])
            
            # Remove duplicates while preserving order
            unique_outputs = []
            for name in output_names:
                if name not in unique_outputs:
                    unique_outputs.append(name)
            
            # Default names if nothing found or only one output
            if len(unique_outputs) == 0:
                return 1, ['output']
            elif len(unique_outputs) == 1:
                return 1, ['main_output']
            else:
                # Multiple outputs found
                return len(unique_outputs), unique_outputs
                
        except Exception as e:
            self.print_warning(f"Error analyzing model source code: {e}")
            return 1, ['output']
