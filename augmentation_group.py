import ast
import pyqtgraph.parametertree.parameterTypes as pTypes
from PySide6.QtWidgets import QFileDialog, QMessageBox

# Custom augmentation group that includes preset methods and allows adding custom methods from files

class AugmentationGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset augmentation methods
        self._add_preset_augmentations()
        
        # Add custom augmentation button
        self._add_custom_button()
    
    def _add_preset_augmentations(self):
        """Add preset augmentation methods with their parameters."""
        preset_methods = [
            {
                'name': 'Horizontal Flip',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable horizontal flip augmentation'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of applying horizontal flip'}
                ],
                'tip': 'Randomly flip images horizontally'
            },
            {
                'name': 'Vertical Flip',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable vertical flip augmentation'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of applying vertical flip'}
                ],
                'tip': 'Randomly flip images vertically'
            },
            {
                'name': 'Rotation',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable rotation augmentation'},
                    {'name': 'angle_range', 'type': 'float', 'value': 15.0, 'limits': (0.0, 180.0), 'suffix': '°', 'tip': 'Maximum rotation angle in degrees'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of applying rotation'}
                ],
                'tip': 'Randomly rotate images by specified angle range'
            },
            {
                'name': 'Gaussian Noise',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable Gaussian noise augmentation'},
                    {'name': 'variance_limit', 'type': 'float', 'value': 0.01, 'limits': (0.0, 0.1), 'tip': 'Maximum variance of Gaussian noise'},
                    {'name': 'probability', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Probability of adding noise'}
                ],
                'tip': 'Add random Gaussian noise to images'
            },
            {
                'name': 'Brightness Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable brightness adjustment'},
                    {'name': 'brightness_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum brightness change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of brightness adjustment'}
                ],
                'tip': 'Randomly adjust image brightness'
            },
            {
                'name': 'Contrast Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable contrast adjustment'},
                    {'name': 'contrast_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum contrast change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of contrast adjustment'}
                ],
                'tip': 'Randomly adjust image contrast'
            },
            {
                'name': 'Color Jittering',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable color jittering'},
                    {'name': 'hue_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 50), 'tip': 'Maximum hue shift'},
                    {'name': 'sat_shift_limit', 'type': 'int', 'value': 30, 'limits': (0, 100), 'tip': 'Maximum saturation shift'},
                    {'name': 'val_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 100), 'tip': 'Maximum value shift'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of color jittering'}
                ],
                'tip': 'Randomly adjust hue, saturation, and value'
            },
            {
                'name': 'Random Cropping',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable random cropping'},
                    {'name': 'crop_area_min', 'type': 'float', 'value': 0.08, 'limits': (0.01, 1.0), 'tip': 'Minimum crop area as fraction of original'},
                    {'name': 'crop_area_max', 'type': 'float', 'value': 1.0, 'limits': (0.01, 1.0), 'tip': 'Maximum crop area as fraction of original'},
                    {'name': 'aspect_ratio_min', 'type': 'float', 'value': 0.75, 'limits': (0.1, 2.0), 'tip': 'Minimum aspect ratio for cropping'},
                    {'name': 'aspect_ratio_max', 'type': 'float', 'value': 1.33, 'limits': (0.1, 2.0), 'tip': 'Maximum aspect ratio for cropping'},
                    {'name': 'probability', 'type': 'float', 'value': 1.0, 'limits': (0.0, 1.0), 'tip': 'Probability of random cropping'}
                ],
                'tip': 'Randomly crop parts of the image with specified area and aspect ratio constraints'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom augmentation functions from files."""
        self.addChild({
            'name': 'Load Custom Augmentations',
            'type': 'action',
            'tip': 'Click to load custom augmentation functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Augmentations')
        custom_button.sigActivated.connect(self._load_custom_augmentations)
    
    def _load_custom_augmentations(self):
        """Load custom augmentation functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom augmentation functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_augmentation_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid augmentation functions found in the selected file.\n\n"
                    "Functions should accept 'image' parameter and return modified image."
                )
                return
            
            # Add each found function as a custom augmentation
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom augmentation function(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom augmentations from file:\n{str(e)}"
            )
    
    def _extract_augmentation_functions(self, file_path):
        """Extract valid augmentation functions from a Python file."""
        custom_functions = {}
        
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
                    
                    # Check if it's a valid augmentation function
                    if self._is_valid_augmentation_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom augmentation function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_augmentation_function(self, func_node):
        """Check if a function is a valid augmentation function."""
        # Check if function has at least one parameter (should be 'image')
        if not func_node.args.args:
            return False
        
        # Check if first parameter is likely an image parameter
        first_param = func_node.args.args[0].arg
        if first_param not in ['image', 'img', 'x', 'data']:
            return False
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        if not has_return:
            return False
        
        return True
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'image' parameter)."""
        params = []
        
        # Skip the first parameter (image) and extract others
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 0.5,   # Default value
                'limits': (0.0, 1.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'angle' in param_name.lower():
                param_info.update({'type': 'float', 'default': 15.0, 'limits': (0.0, 180.0), 'suffix': '°'})
            elif 'prob' in param_name.lower() or 'p' == param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'strength' in param_name.lower() or 'intensity' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 5.0)})
            elif 'size' in param_name.lower() or 'kernel' in param_name.lower():
                param_info.update({'type': 'int', 'default': 3, 'limits': (1, 15)})
            elif 'enable' in param_name.lower():
                param_info.update({'type': 'bool', 'default': True})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 1  # -1 to skip image param
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as an augmentation method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists (check both original and display names)
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} augmentation'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            children.append({
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'limits': param_info.get('limits'),
                'suffix': param_info.get('suffix', ''),
                'tip': param_info['tip']
            })
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
        ])
        
        # Create the augmentation method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,  # Keep original function name
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Augmentations" button
        # Find the button's index and insert before it
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Augmentations':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            # Fallback: add at the end if button not found
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        # This method is called by the parameter tree system but we use the button instead
        pass

