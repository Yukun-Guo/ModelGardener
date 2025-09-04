"""
Loss and metrics configuration module for ModelGardener CLI.
"""

import ast
import inspect
import importlib.util
import os
from typing import Dict, Any, List, Tuple, Optional
from .base_config import BaseConfig


class LossMetricsConfig(BaseConfig):
    """Loss function and metrics configuration handler."""
    
    def __init__(self):
        super().__init__()
        self.available_losses = [
            'Categorical Crossentropy', 'Sparse Categorical Crossentropy', 'Binary Crossentropy',
            'Mean Squared Error', 'Mean Absolute Error', 'Huber Loss', 'Focal Loss'
        ]
        self.available_metrics = [
            'Accuracy', 'Categorical Accuracy', 'Sparse Categorical Accuracy', 'Top K Categorical Accuracy',
            'Precision', 'Recall', 'F1 Score', 'AUC', 'Mean Squared Error', 'Mean Absolute Error'
        ]

    def _is_loss_function(self, obj, name: str) -> bool:
        """
        Check if an object is a valid loss function.
        
        Args:
            obj: The object to check
            name: Name of the object
            
        Returns:
            bool: True if it's a valid loss function
        """
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

    def _extract_loss_parameters(self, obj) -> Dict[str, Any]:
        """
        Extract parameters from a loss function.
        
        Args:
            obj: The loss function or class
            
        Returns:
            Dict containing parameter information
        """
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

    def analyze_custom_loss_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract loss functions.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (success, loss_info)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            module = self.load_custom_module(file_path, "custom_losses")
            if module is None:
                return False, {}
            
            loss_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if self._is_loss_function(obj, name):
                    info = self._extract_loss_parameters(obj)
                    if info:
                        loss_info[name] = info
            
            return len(loss_info) > 0, loss_info
            
        except Exception as e:
            self.print_error(f"Error analyzing loss function file: {str(e)}")
            return False, {}

    def interactive_custom_loss_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom loss function from analyzed file.
        
        Args:
            file_path: Path to the custom loss function file
            
        Returns:
            Tuple of (selected_loss_name, loss_info)
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return None, {}
            
        success, analysis_result = self.analyze_custom_loss_file(file_path)
        
        if not success or not analysis_result:
            self.print_error("No valid loss functions found in the file")
            return None, {}
        
        print(f"\n✅ Found {len(analysis_result)} loss function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user
        choices = []
        for name, info in analysis_result.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} {info['signature']}"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = f"{name} ({info['type']})"
            
            choices.append(choice_text)
        
        # Let user select
        selected_choice = inquirer.list_input(
            "Select custom loss function to use",
            choices=choices
        )
        
        # Extract the name from the choice
        selected_name = selected_choice.split(' ')[0] if ' ' in selected_choice else selected_choice
        
        if selected_name in analysis_result:
            info = analysis_result[selected_name]
            self.print_success(f"Selected custom loss function: {selected_name}")
            print(f"   Type: {info['type']}")
            
            # Ask for parameters if any
            parameters = {}
            if 'parameters' in info and info['parameters']:
                param_count = len([p for p in info['parameters'].values() if not p['required']])
                if param_count > 0:
                    print(f"\n⚙️  Custom loss function parameters found: {param_count}")
                    
                    for param_name, param_info in info['parameters'].items():
                        if not param_info['required']:  # Only ask for optional parameters
                            default_val = param_info.get('default', '')
                            user_value = inquirer.text(
                                f"Enter {param_name} (default: {default_val})",
                                default=str(default_val) if default_val is not None else ""
                            )
                            
                            # Convert to appropriate type
                            if user_value:
                                try:
                                    if param_info['type'] == 'int':
                                        parameters[param_name] = int(user_value)
                                    elif param_info['type'] == 'float':
                                        parameters[param_name] = float(user_value)
                                    elif param_info['type'] == 'bool':
                                        parameters[param_name] = user_value.lower() in ['true', '1', 'yes', 'on']
                                    else:
                                        parameters[param_name] = user_value
                                except ValueError:
                                    parameters[param_name] = user_value
            
            result_info = info.copy()
            if parameters:
                result_info['user_parameters'] = parameters
            
            return selected_name, result_info
        
        return None, {}

    def analyze_custom_metrics_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom metrics functions.
        
        Args:
            file_path: Path to the Python file containing custom metrics
            
        Returns:
            Tuple of (success, analysis_result)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use AST to parse the file
            tree = ast.parse(content)
            
            functions_info = {}
            
            # Look for function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Skip private functions
                    if func_name.startswith('_'):
                        continue
                    
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    
                    # Look for typical metrics function patterns
                    if self._is_likely_metrics_function(func_name, args, content):
                        # Extract parameters
                        parameters = self._extract_function_parameters_from_ast(node, content)
                        
                        # Get signature string
                        signature = f"({', '.join(args)})"
                        
                        functions_info[func_name] = {
                            'type': 'function',
                            'function_name': func_name,
                            'signature': signature,
                            'parameters': parameters,
                            'file_path': file_path
                        }
                
                # Look for class definitions that might be custom metrics
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Skip private classes
                    if class_name.startswith('_'):
                        continue
                    
                    # Look for classes that might be metrics
                    has_call = any(isinstance(n, ast.FunctionDef) and n.name == '__call__' 
                                 for n in node.body)
                    has_compute = any(isinstance(n, ast.FunctionDef) and n.name == 'compute' 
                                    for n in node.body)
                    
                    if has_call or has_compute or 'metric' in class_name.lower():
                        functions_info[class_name] = {
                            'type': 'class',
                            'function_name': class_name,
                            'signature': '(class)',
                            'parameters': {},
                            'file_path': file_path
                        }
            
            return len(functions_info) > 0, functions_info
            
        except Exception as e:
            self.print_error(f"Error analyzing metrics file {file_path}: {e}")
            return False, {}

    def _is_likely_metrics_function(self, func_name: str, args: List[str], content: str) -> bool:
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

    def _extract_function_parameters_from_ast(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract function parameters from AST node."""
        parameters = {}
        try:
            for arg in node.args.args:
                if arg.arg in ['y_true', 'y_pred', 'true', 'pred', 'actual', 'predicted']:
                    continue  # Skip the main input parameters
                    
                param_info = {
                    'name': arg.arg,
                    'required': True,  # Will be updated if default found
                    'default': None,
                    'type': 'Any'
                }
                parameters[arg.arg] = param_info
            
            # Update with defaults if available
            if node.args.defaults:
                defaults_start = len(node.args.args) - len(node.args.defaults)
                for i, default in enumerate(node.args.defaults):
                    arg_index = defaults_start + i
                    if arg_index < len(node.args.args):
                        arg_name = node.args.args[arg_index].arg
                        if arg_name in parameters:
                            parameters[arg_name]['required'] = False
                            # Try to extract default value
                            if isinstance(default, ast.Constant):
                                parameters[arg_name]['default'] = default.value
                                parameters[arg_name]['type'] = type(default.value).__name__
                            elif isinstance(default, ast.Num):  # Python < 3.8
                                parameters[arg_name]['default'] = default.n
                                parameters[arg_name]['type'] = type(default.n).__name__
                            elif isinstance(default, ast.Str):  # Python < 3.8
                                parameters[arg_name]['default'] = default.s
                                parameters[arg_name]['type'] = 'str'
                                
        except Exception:
            pass
            
        return parameters

    def interactive_custom_metrics_selection(self, file_path: str, metrics_info: Dict[str, Any]) -> List[str]:
        """
        Interactive selection of custom metrics functions from analyzed file.
        
        Args:
            file_path: Path to the custom metrics function file
            metrics_info: Analysis result from analyze_custom_metrics_file
            
        Returns:
            List of selected metrics names
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return []
            
        if not metrics_info:
            self.print_error("No metrics functions found in the file")
            return []
        
        print(f"\n✅ Found {len(metrics_info)} metrics function(s) in {os.path.basename(file_path)}")
        
        # Create choices for the user
        choices = []
        for name, info in metrics_info.items():
            if info['type'] == 'function' and 'signature' in info:
                choice_text = f"{name} (function)"
            elif info['type'] == 'class':
                choice_text = f"{name} (class)"
            else:
                choice_text = name
            choices.append(choice_text)
        
        # Let user select multiple metrics
        if len(choices) == 1:
            # Only one function available
            selected_choices = inquirer.checkbox(
                "Select metrics functions to load",
                choices=choices,
                default=choices
            )
        else:
            selected_choices = inquirer.checkbox(
                "Select metrics functions to load (use space to select, enter to confirm)",
                choices=choices
            )
        
        # Extract actual function names from choices
        selected_metrics = []
        for choice in selected_choices:
            # Extract function name
            actual_name = choice.split(' ')[0] if ' ' in choice else choice.split('(')[0]
            selected_metrics.append(f"{actual_name} (custom)")
        
        return selected_metrics
