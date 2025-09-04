"""
Preprocessing configuration module for ModelGardener CLI.
"""

import ast
import inspect
import importlib.util
import os
from typing import Dict, Any, List, Tuple, Optional
from .base_config import BaseConfig


class PreprocessingConfig(BaseConfig):
    """Preprocessing configuration handler."""
    
    def __init__(self):
        super().__init__()

    def _is_preprocessing_function(self, obj, name: str) -> bool:
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

    def _extract_preprocessing_parameters(self, obj) -> Dict[str, Any]:
        """Extract parameters from preprocessing function."""
        parameters = {}
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

    def analyze_custom_preprocessing_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom preprocessing functions.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (success, preprocessing_info)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            module = self.load_custom_module(file_path, "custom_preprocessing")
            if module is None:
                return False, {}
            
            preprocessing_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if self._is_preprocessing_function(obj, name):
                    info = self._extract_preprocessing_parameters(obj)
                    if info:
                        preprocessing_info[name] = info
            
            return len(preprocessing_info) > 0, preprocessing_info
            
        except Exception as e:
            self.print_error(f"Error analyzing preprocessing file: {str(e)}")
            return False, {}

    def interactive_custom_preprocessing_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom preprocessing functions from analyzed file.
        
        Args:
            file_path: Path to the Python file containing preprocessing functions
            
        Returns:
            Tuple of (selected_function_name, function_info_with_user_params)
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return None, {}
            
        success, analysis_result = self.analyze_custom_preprocessing_file(file_path)
        
        if not success:
            self.print_error("No valid preprocessing functions found in the file")
            return None, {}
        
        if not analysis_result:
            self.print_error("No preprocessing functions found in the file")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} preprocessing function(s):")
        for name, info in analysis_result.items():
            print(f"   • {name}: {info.get('description', 'No description')}")
        
        # Let user select function
        function_names = list(analysis_result.keys())
        selected_function = inquirer.list_input(
            "Select preprocessing function",
            choices=function_names
        )
        
        if not selected_function:
            return None, {}
        
        function_info = analysis_result[selected_function]
        user_parameters = {}
        
        # Configure parameters for selected function
        if function_info.get('parameters'):
            print(f"\n⚙️  Configuring parameters for {selected_function}:")
            
            for param_name, param_info in function_info['parameters'].items():
                param_type = param_info.get('type', 'str')
                default_val = param_info.get('default')
                
                if param_type == 'bool':
                    value = inquirer.confirm(
                        f"Set {param_name}",
                        default=bool(default_val) if default_val is not None else False
                    )
                else:
                    prompt_text = f"Enter {param_name}"
                    if default_val is not None:
                        prompt_text += f" (default: {default_val})"
                    
                    value_str = inquirer.text(prompt_text, default=str(default_val) if default_val is not None else "")
                    
                    # Convert to appropriate type
                    try:
                        if param_type == 'int':
                            value = int(value_str) if value_str else (default_val if default_val is not None else 0)
                        elif param_type == 'float':
                            value = float(value_str) if value_str else (default_val if default_val is not None else 0.0)
                        elif param_type == 'list':
                            if value_str:
                                # Try to parse as list
                                try:
                                    value = ast.literal_eval(value_str)
                                except:
                                    value = value_str.split(',') if ',' in value_str else [value_str]
                            else:
                                value = default_val if default_val is not None else []
                        else:
                            value = value_str if value_str else (default_val if default_val is not None else "")
                    except ValueError:
                        value = default_val if default_val is not None else ""
                        self.print_warning(f"Invalid value for {param_name}, using default: {value}")
                
                user_parameters[param_name] = value
        
        # Return function info with user parameters
        result_info = function_info.copy()
        result_info['user_parameters'] = user_parameters
        result_info['file_path'] = file_path
        
        return selected_function, result_info

    def configure_preprocessing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive configuration of preprocessing settings.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Updated preprocessing configuration
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return {}
            
        print("\n🔧 Preprocessing Configuration")
        print("=" * 40)
        
        preprocessing_config = {
            "Resizing": {
                "enabled": False,
                "target_size": {"width": 224, "height": 224, "depth": 1},
                "interpolation": "bilinear",
                "preserve_aspect_ratio": True,
                "data_format": "2D"
            },
            "Normalization": {
                "enabled": True,
                "method": "zero-center",
                "min_value": 0.0,
                "max_value": 1.0,
                "mean": {"r": 0.485, "g": 0.456, "b": 0.406},
                "std": {"r": 0.229, "g": 0.224, "b": 0.225},
                "axis": -1,
                "epsilon": 1e-07
            }
        }
        
        # 1. Configure Resizing
        print("\n📏 Step 1: Resizing Configuration")
        resizing_strategies = ['None', 'scaling', 'crop-padding']
        resizing_strategy = inquirer.list_input(
            "Select resizing strategy",
            choices=resizing_strategies,
            default='scaling'
        )
        
        if resizing_strategy != 'None':
            preprocessing_config["Resizing"]["enabled"] = True
            
            # Select method for the chosen strategy
            if resizing_strategy == 'scaling':
                scaling_methods = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']
                method = inquirer.list_input(
                    "Select scaling method",
                    choices=scaling_methods,
                    default='bilinear'
                )
                preprocessing_config["Resizing"]["interpolation"] = method
                
                # Ask about preserving aspect ratio
                preserve_aspect = inquirer.confirm(
                    "Preserve aspect ratio?",
                    default=True
                )
                preprocessing_config["Resizing"]["preserve_aspect_ratio"] = preserve_aspect
                
            elif resizing_strategy == 'crop-padding':
                crop_methods = ['central_cropping', 'random_cropping']
                method = inquirer.list_input(
                    "Select crop-padding method",
                    choices=crop_methods,
                    default='central_cropping'
                )
                preprocessing_config["Resizing"]["crop_method"] = method
                preprocessing_config["Resizing"]["interpolation"] = "nearest"  # Default for cropping
            
            # Set target size
            print("\n📐 Configure target dimensions:")
            
            # Ask about data format first
            data_format = inquirer.list_input(
                "Select data format",
                choices=['2D (images)', '3D (volumes/sequences)'],
                default='2D (images)'
            )
            
            width = inquirer.text("Enter target width", default="224")
            height = inquirer.text("Enter target height", default="224")
            
            if data_format == '3D (volumes/sequences)':
                depth = inquirer.text("Enter depth (for 3D data)", default="16")
                preprocessing_config["Resizing"]["data_format"] = "3D"
            else:
                depth = "1"  # For 2D data, depth is 1
                preprocessing_config["Resizing"]["data_format"] = "2D"
            
            try:
                preprocessing_config["Resizing"]["target_size"] = {
                    "width": int(width),
                    "height": int(height),
                    "depth": int(depth)
                }
            except ValueError:
                self.print_warning("Invalid dimensions, using defaults")
        
        # 2. Configure Normalization
        print("\n📊 Step 2: Normalization Configuration")
        enable_normalization = inquirer.confirm("Enable normalization?", default=True)
        
        if enable_normalization:
            preprocessing_config["Normalization"]["enabled"] = True
            
            # Select normalization method
            normalization_methods = [
                'zero-center',  # (x - mean) / std
                'min-max',      # (x - min) / (max - min)
                'unit-norm',    # x / |x|
                'standard',     # (x - mean) / std (same as zero-center)
                'robust'        # (x - median) / IQR
            ]
            
            norm_method = inquirer.list_input(
                "Select normalization method",
                choices=normalization_methods,
                default='zero-center'
            )
            preprocessing_config["Normalization"]["method"] = norm_method
            
            # Configure parameters based on selected method
            if norm_method in ['zero-center', 'standard']:
                print("\n⚙️  Configure normalization parameters:")
                
                use_imagenet = inquirer.confirm(
                    "Use ImageNet statistics? (recommended for pre-trained models)",
                    default=True
                )
                
                if not use_imagenet:
                    # Custom mean and std
                    mean_r = inquirer.text("Enter mean for R channel", default="0.485")
                    mean_g = inquirer.text("Enter mean for G channel", default="0.456")
                    mean_b = inquirer.text("Enter mean for B channel", default="0.406")
                    
                    std_r = inquirer.text("Enter std for R channel", default="0.229")
                    std_g = inquirer.text("Enter std for G channel", default="0.224")
                    std_b = inquirer.text("Enter std for B channel", default="0.225")
                    
                    try:
                        preprocessing_config["Normalization"]["mean"] = {
                            "r": float(mean_r),
                            "g": float(mean_g),
                            "b": float(mean_b)
                        }
                        preprocessing_config["Normalization"]["std"] = {
                            "r": float(std_r),
                            "g": float(std_g),
                            "b": float(std_b)
                        }
                    except ValueError:
                        self.print_warning("Invalid values, using ImageNet defaults")
            
            elif norm_method == 'min-max':
                min_val = inquirer.text("Enter minimum value", default="0.0")
                max_val = inquirer.text("Enter maximum value", default="1.0")
                
                try:
                    preprocessing_config["Normalization"]["min_value"] = float(min_val)
                    preprocessing_config["Normalization"]["max_value"] = float(max_val)
                except ValueError:
                    self.print_warning("Invalid values, using defaults")
            
            # Common parameters
            axis = inquirer.text("Enter normalization axis (-1 for last axis)", default="-1")
            epsilon = inquirer.text("Enter epsilon value", default="1e-07")
            
            try:
                preprocessing_config["Normalization"]["axis"] = int(axis)
                preprocessing_config["Normalization"]["epsilon"] = float(epsilon)
            except ValueError:
                self.print_warning("Invalid values, using defaults")
        else:
            preprocessing_config["Normalization"]["enabled"] = False
        
        # 3. Custom Preprocessing
        print("\n🛠️  Step 3: Custom Preprocessing")
        add_custom = inquirer.confirm("Add custom preprocessing functions?", default=False)
        
        custom_preprocessing = []
        
        if add_custom:
            custom_preprocessing_path = inquirer.text(
                "Enter path to Python file containing custom preprocessing functions",
                default="./example_funcs/example_custom_preprocessing.py"
            )
            
            if custom_preprocessing_path and os.path.exists(custom_preprocessing_path):
                # Analyze and select custom preprocessing functions
                success, preprocessing_info = self.analyze_custom_preprocessing_file(custom_preprocessing_path)
                
                if success and preprocessing_info:
                    print(f"\n✅ Found {len(preprocessing_info)} preprocessing function(s)")
                    
                    # Allow multiple selections
                    add_more = True
                    while add_more:
                        func_name, func_info = self.interactive_custom_preprocessing_selection(custom_preprocessing_path)
                        
                        if func_name and func_info:
                            custom_func_config = {
                                "function_name": func_name,
                                "enabled": True,
                                "file_path": custom_preprocessing_path,
                                "parameters": func_info.get('user_parameters', {})
                            }
                            custom_preprocessing.append(custom_func_config)
                            self.print_success(f"Added custom preprocessing: {func_name}")
                        
                        add_more = inquirer.confirm("Add another custom preprocessing function?", default=False)
                else:
                    self.print_error("No valid preprocessing functions found in the file")
            else:
                self.print_error("Invalid file path or file does not exist")
        
        # Add custom preprocessing to config if any were selected
        if custom_preprocessing:
            preprocessing_config["Custom"] = custom_preprocessing
        
        return preprocessing_config
