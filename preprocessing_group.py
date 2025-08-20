import ast
import pyqtgraph.parametertree.parameterTypes as pTypes
from PySide6.QtWidgets import QFileDialog, QMessageBox

# Custom preprocessing group that includes preset methods and allows adding custom methods from files

class PreprocessingGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset preprocessing methods
        self._add_preset_preprocessing()
        
        # Add custom preprocessing button
        self._add_custom_button()
    
    def _add_preset_preprocessing(self):
        """Add preset preprocessing methods with their parameters."""
        preset_methods = [
            {
                'name': 'Resizing',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable image resizing'},
                    {'name': 'target_size', 'type': 'group', 'children': [
                        {'name': 'width', 'type': 'int', 'value': 224, 'limits': (1, 2048), 'tip': 'Target width in pixels'},
                        {'name': 'height', 'type': 'int', 'value': 224, 'limits': (1, 2048), 'tip': 'Target height in pixels'},
                        {'name': 'depth', 'type': 'int', 'value': 1, 'limits': (1, 512), 'tip': 'Target depth for 3D data (1 for 2D)'}
                    ], 'tip': 'Target dimensions for resizing'},
                    {'name': 'interpolation', 'type': 'list', 'limits': ['bilinear', 'nearest', 'bicubic', 'area'], 'value': 'bilinear', 'tip': 'Interpolation method for resizing'},
                    {'name': 'preserve_aspect_ratio', 'type': 'bool', 'value': True, 'tip': 'Whether to preserve aspect ratio during resize'},
                    {'name': 'data_format', 'type': 'list', 'limits': ['2D', '3D'], 'value': '2D', 'tip': 'Data format (2D for images, 3D for volumes)'}
                ],
                'tip': 'Resize images to target dimensions with support for 2D and 3D data'
            },
            {
                'name': 'Normalization',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable data normalization'},
                    {'name': 'method', 'type': 'list', 'limits': ['min-max', 'zero-center', 'standardization', 'unit-norm', 'robust'], 'value': 'zero-center', 'tip': 'Normalization method'},
                    {'name': 'min_value', 'type': 'float', 'value': 0.0, 'limits': (-10.0, 10.0), 'tip': 'Minimum value for min-max normalization'},
                    {'name': 'max_value', 'type': 'float', 'value': 1.0, 'limits': (-10.0, 10.0), 'tip': 'Maximum value for min-max normalization'},
                    {'name': 'mean', 'type': 'group', 'children': [
                        {'name': 'r', 'type': 'float', 'value': 0.485, 'limits': (0.0, 1.0), 'tip': 'Mean value for red channel'},
                        {'name': 'g', 'type': 'float', 'value': 0.456, 'limits': (0.0, 1.0), 'tip': 'Mean value for green channel'},
                        {'name': 'b', 'type': 'float', 'value': 0.406, 'limits': (0.0, 1.0), 'tip': 'Mean value for blue channel'}
                    ], 'tip': 'Mean values for zero-center normalization (ImageNet defaults)'},
                    {'name': 'std', 'type': 'group', 'children': [
                        {'name': 'r', 'type': 'float', 'value': 0.229, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for red channel'},
                        {'name': 'g', 'type': 'float', 'value': 0.224, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for green channel'},
                        {'name': 'b', 'type': 'float', 'value': 0.225, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for blue channel'}
                    ], 'tip': 'Standard deviation values for standardization (ImageNet defaults)'},
                    {'name': 'axis', 'type': 'int', 'value': -1, 'limits': (-3, 3), 'tip': 'Axis along which to normalize (-1 for all)'},
                    {'name': 'epsilon', 'type': 'float', 'value': 1e-7, 'limits': (1e-10, 1e-3), 'tip': 'Small constant to avoid division by zero'}
                ],
                'tip': 'Normalize data using various methods (min-max, zero-center, standardization, etc.)'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom preprocessing functions from files."""
        self.addChild({
            'name': 'Load Custom Preprocessing',
            'type': 'action',
            'tip': 'Click to load custom preprocessing functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Preprocessing')
        custom_button.sigActivated.connect(self._load_custom_preprocessing)
    
    def _load_custom_preprocessing(self):
        """Load custom preprocessing functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom preprocessing functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_preprocessing_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid preprocessing functions found in the selected file.\n\n"
                    "Functions should accept 'data' parameter and return processed data."
                )
                return
            
            # Add each found function as a custom preprocessing method
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom preprocessing function(s):\n" +
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
                f"Failed to load custom preprocessing from file:\n{str(e)}"
            )
    
    def _extract_preprocessing_functions(self, file_path):
        """Extract valid preprocessing functions from a Python file."""
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
                    
                    # Check if it's a valid preprocessing function
                    if self._is_valid_preprocessing_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom preprocessing function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_preprocessing_function(self, func_node):
        """Check if a function is a valid preprocessing function."""
        # Check if function has at least one parameter (should be 'data' or similar)
        if not func_node.args.args:
            return False
        
        # Check if first parameter is likely a data parameter
        first_param = func_node.args.args[0].arg
        if first_param not in ['data', 'x', 'input', 'array', 'tensor']:
            return False
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        if not has_return:
            return False
        
        return True
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'data' parameter)."""
        params = []
        
        # Skip the first parameter (data) and extract others
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 1.0,   # Default value
                'limits': (0.0, 10.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'size' in param_name.lower() or 'dim' in param_name.lower():
                param_info.update({'type': 'int', 'default': 224, 'limits': (1, 2048)})
            elif 'scale' in param_name.lower() or 'factor' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
            elif 'mean' in param_name.lower() or 'center' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'std' in param_name.lower() or 'deviation' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.25, 'limits': (0.001, 1.0)})
            elif 'enable' in param_name.lower():
                param_info.update({'type': 'bool', 'default': True})
            elif 'method' in param_name.lower() or 'mode' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'bilinear'})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 1  # -1 to skip data param
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
                        elif isinstance(default.value, str):
                            params[param_index]['type'] = 'str'
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as a preprocessing method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists (check both original and display names)
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} preprocessing'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            # Add limits for numeric types
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            children.append(param_config)
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
        ])
        
        # Create the preprocessing method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Preprocessing" button
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Preprocessing':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

