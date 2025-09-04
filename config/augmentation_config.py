"""
Augmentation configuration module for ModelGardener CLI.
"""

import ast
import inspect
import importlib.util
import os
from typing import Dict, Any, List, Tuple, Optional
from .base_config import BaseConfig


class AugmentationConfig(BaseConfig):
    """Augmentation configuration handler."""
    
    def __init__(self):
        super().__init__()

    def _is_augmentation_function(self, obj, name: str) -> bool:
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

    def _extract_augmentation_parameters(self, func) -> Dict[str, Any]:
        """
        Extract parameters from an augmentation function.
        
        Args:
            func: The function to analyze
            
        Returns:
            Dictionary containing function information
        """
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

    def analyze_custom_augmentation_file(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze a Python file to extract custom augmentation functions.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (success, augmentation_info)
        """
        try:
            if not os.path.exists(file_path):
                return False, {}
            
            # Load the module
            module = self.load_custom_module(file_path, "custom_augmentation")
            if module is None:
                return False, {}
            
            augmentation_info = {}
            
            # Analyze module contents
            for name, obj in inspect.getmembers(module):
                if self._is_augmentation_function(obj, name):
                    info = self._extract_augmentation_parameters(obj)
                    if info:
                        augmentation_info[name] = info
            
            return len(augmentation_info) > 0, augmentation_info
            
        except Exception as e:
            self.print_error(f"Error analyzing augmentation file: {str(e)}")
            return False, {}

    def interactive_custom_augmentation_selection(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Interactive selection of custom augmentation functions from analyzed file.
        
        Args:
            file_path: Path to the Python file containing augmentation functions
            
        Returns:
            Tuple of (selected_function_name, function_info_with_user_params)
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return None, {}
            
        success, analysis_result = self.analyze_custom_augmentation_file(file_path)
        
        if not success:
            self.print_error("No valid augmentation functions found in the file")
            return None, {}
        
        if not analysis_result:
            self.print_error("No augmentation functions found in the file")
            return None, {}
        
        print(f"\n🔍 Found {len(analysis_result)} augmentation function(s):")
        for name, info in analysis_result.items():
            print(f"   • {name}: {info.get('description', 'No description')}")
        
        # Let user select function
        function_names = list(analysis_result.keys())
        selected_function = inquirer.list_input(
            "Select augmentation function",
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

    def configure_augmentation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive configuration of data augmentation settings.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Updated augmentation configuration
        """
        try:
            import inquirer
        except ImportError:
            self.print_error("inquirer library is required for interactive mode")
            return {}
            
        print("\n🔄 Data Augmentation Configuration")
        print("=" * 45)
        
        # Default augmentation configuration with common augmentations
        augmentation_config = {
            "Horizontal Flip": {
                "enabled": False,
                "probability": 0.5
            },
            "Vertical Flip": {
                "enabled": False,
                "probability": 0.5
            },
            "Rotation": {
                "enabled": False,
                "angle_range": 15.0,
                "probability": 0.5
            },
            "Gaussian Noise": {
                "enabled": False,
                "variance_limit": 0.01,
                "probability": 0.2
            },
            "Brightness Adjustment": {
                "enabled": False,
                "brightness_limit": 0.2,
                "probability": 0.5
            },
            "Contrast Adjustment": {
                "enabled": False,
                "contrast_limit": 0.2,
                "probability": 0.5
            },
            "Color Jittering": {
                "enabled": False,
                "hue_shift_limit": 20,
                "sat_shift_limit": 30,
                "val_shift_limit": 20,
                "probability": 0.5
            },
            "Random Cropping": {
                "enabled": False,
                "crop_area_min": 0.08,
                "crop_area_max": 1.0,
                "aspect_ratio_min": 0.75,
                "aspect_ratio_max": 1.33,
                "probability": 1.0
            }
        }
        
        # 1. Configure Preset Augmentations
        print("\n🎯 Step 1: Preset Augmentation Selection")
        
        # Present augmentations as categories
        geometric_augs = ["Horizontal Flip", "Vertical Flip", "Rotation", "Random Cropping"]
        color_augs = ["Brightness Adjustment", "Contrast Adjustment", "Color Jittering"]
        noise_augs = ["Gaussian Noise"]
        
        print("\n📐 Geometric Augmentations:")
        for aug_name in geometric_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                # Configure specific parameters for each augmentation
                if aug_name == "Horizontal Flip" or aug_name == "Vertical Flip":
                    prob = inquirer.text(f"Probability for {aug_name}", default="0.5")
                    try:
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid probability, using default")
                        
                elif aug_name == "Rotation":
                    angle = inquirer.text("Maximum rotation angle (degrees)", default="15.0")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["angle_range"] = float(angle)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
                        
                elif aug_name == "Random Cropping":
                    crop_min = inquirer.text("Minimum crop area (0.01-1.0)", default="0.08")
                    crop_max = inquirer.text("Maximum crop area (0.01-1.0)", default="1.0")
                    aspect_min = inquirer.text("Minimum aspect ratio", default="0.75")
                    aspect_max = inquirer.text("Maximum aspect ratio", default="1.33")
                    try:
                        augmentation_config[aug_name]["crop_area_min"] = float(crop_min)
                        augmentation_config[aug_name]["crop_area_max"] = float(crop_max)
                        augmentation_config[aug_name]["aspect_ratio_min"] = float(aspect_min)
                        augmentation_config[aug_name]["aspect_ratio_max"] = float(aspect_max)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
        
        print("\n🎨 Color Augmentations:")
        for aug_name in color_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                if aug_name == "Brightness Adjustment":
                    limit = inquirer.text("Brightness change limit (±)", default="0.2")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["brightness_limit"] = float(limit)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
                        
                elif aug_name == "Contrast Adjustment":
                    limit = inquirer.text("Contrast change limit (±)", default="0.2")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["contrast_limit"] = float(limit)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
                        
                elif aug_name == "Color Jittering":
                    hue = inquirer.text("Hue shift limit", default="20")
                    sat = inquirer.text("Saturation shift limit", default="30")
                    val = inquirer.text("Value shift limit", default="20")
                    prob = inquirer.text("Probability", default="0.5")
                    try:
                        augmentation_config[aug_name]["hue_shift_limit"] = int(hue)
                        augmentation_config[aug_name]["sat_shift_limit"] = int(sat)
                        augmentation_config[aug_name]["val_shift_limit"] = int(val)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
        
        print("\n🔊 Noise Augmentations:")
        for aug_name in noise_augs:
            enable_aug = inquirer.confirm(
                f"Enable {aug_name}?",
                default=False
            )
            
            if enable_aug:
                augmentation_config[aug_name]["enabled"] = True
                
                if aug_name == "Gaussian Noise":
                    variance = inquirer.text("Noise variance limit", default="0.01")
                    prob = inquirer.text("Probability", default="0.2")
                    try:
                        augmentation_config[aug_name]["variance_limit"] = float(variance)
                        augmentation_config[aug_name]["probability"] = float(prob)
                    except ValueError:
                        self.print_warning("Invalid values, using defaults")
        
        # 2. Custom Augmentation Functions
        print("\n🛠️  Step 2: Custom Augmentation Functions")
        add_custom = inquirer.confirm("Add custom augmentation functions?", default=False)
        
        if add_custom:
            custom_augmentation_path = inquirer.text(
                "Enter path to Python file containing custom augmentation functions",
                default="./example_funcs/example_custom_augmentations.py"
            )
            
            if custom_augmentation_path and os.path.exists(custom_augmentation_path):
                # Analyze and select custom augmentation functions
                success, augmentation_info = self.analyze_custom_augmentation_file(custom_augmentation_path)
                
                if success and augmentation_info:
                    print(f"\n✅ Found {len(augmentation_info)} augmentation function(s)")
                    
                    # Allow multiple selections
                    add_more = True
                    while add_more:
                        func_name, func_info = self.interactive_custom_augmentation_selection(custom_augmentation_path)
                        
                        if func_name and func_info:
                            # Set probability for this augmentation
                            prob = inquirer.text("Set probability for this augmentation", default="0.5")
                            try:
                                probability = float(prob)
                            except ValueError:
                                probability = 0.5
                                self.print_warning("Invalid probability, using 0.5")
                            
                            custom_func_config = {
                                "enabled": True,
                                "function_name": func_name,
                                "file_path": custom_augmentation_path,
                                "probability": probability,
                                "parameters": func_info.get('user_parameters', {})
                            }
                            
                            # Add to config with custom naming
                            custom_key = f"{func_name} (custom)"
                            augmentation_config[custom_key] = custom_func_config
                            self.print_success(f"Added custom augmentation: {func_name}")
                        
                        add_more = inquirer.confirm("Add another custom augmentation function?", default=False)
                else:
                    self.print_error("No valid augmentation functions found in the file")
            else:
                self.print_error("Invalid file path or file does not exist")
        
        return augmentation_config
