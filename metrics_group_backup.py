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

# Custom metrics group that includes preset metrics and allows adding custom metrics from files  

class MetricsGroup:
    def __init__(self, **opts):
        self.config = {
            'Metrics Selection': {
                'accuracy': True,
                'precision': False,
                'recall': False,
                'f1_score': False
            },
            'Custom Parameters': {}
        }
        self._custom_metrics = {}
        
        # Add model output configuration first
        self._add_output_configuration()
        
        # Add metrics selection
        self._add_metrics_selection()
        
        # Add custom metrics button
        self._add_custom_button()
    
    def _add_output_configuration(self):
        """Add model output configuration for metrics."""
        self.addChild({
            'name': 'Model Output Configuration',
            'type': 'group',
            'children': [
                {'name': 'num_outputs', 'type': 'int', 'value': 1, 'limits': (1, 10), 'tip': 'Number of model outputs (1 for single output, >1 for multiple outputs)'},
                {'name': 'output_names', 'type': 'str', 'value': 'main_output', 'tip': 'Comma-separated names for multiple outputs (e.g., "main_output,aux_output")'},
                {'name': 'metrics_strategy', 'type': 'list', 'limits': ['shared_metrics_all_outputs', 'different_metrics_per_output'], 'value': 'shared_metrics_all_outputs', 'tip': 'Metrics strategy: same metrics for all outputs or different metrics per output'}
            ],
            'tip': 'Configure model outputs and metrics assignment strategy'
        })
        
        # Connect output configuration change to update metrics selection
        output_config = self.child('Model Output Configuration')
        output_config.child('num_outputs').sigValueChanged.connect(self._update_metrics_selection)
        output_config.child('num_outputs').sigValueChanged.connect(self._update_output_names)
        output_config.child('metrics_strategy').sigValueChanged.connect(self._update_metrics_selection)
    
    def _add_metrics_selection(self):
        """Add metrics selection based on output configuration."""
        # Initially add single metrics selection
        self._update_metrics_selection()
    
    def _update_output_names(self):
        """Update output names based on the number of outputs."""
        output_config = self.child('Model Output Configuration')
        num_outputs = output_config.child('num_outputs').value()
        output_names_param = output_config.child('output_names')
        
        # Generate default names based on number of outputs
        if num_outputs == 1:
            output_names_param.setValue('main_output')
        else:
            # Generate names like "output_1, output_2, output_3"
            names = [f'output_{i+1}' for i in range(num_outputs)]
            output_names_param.setValue(', '.join(names))
        
        # Update metrics selection after changing output names
        self._update_metrics_selection()
    
    def _update_metrics_selection(self):
        """Update metrics selection based on output configuration."""
        # Remove existing metrics selection if any
        existing_groups = []
        for child in self.children():
            if child.name().startswith('Metrics Selection') or child.name().startswith('Output'):
                existing_groups.append(child)
        
        for group in existing_groups:
            self.removeChild(group)
        
        # Get current configuration
        output_config = self.child('Model Output Configuration')
        num_outputs = output_config.child('num_outputs').value()
        metrics_strategy = output_config.child('metrics_strategy').value()
        output_names = output_config.child('output_names').value().split(',')
        output_names = [name.strip() for name in output_names if name.strip()]
        
        # Ensure we have enough output names for the number of outputs
        while len(output_names) < num_outputs:
            output_names.append(f'output_{len(output_names) + 1}')
        
        if num_outputs == 1:
            # Single output - always use shared metrics
            self._add_shared_metrics_selection()
        else:
            # Multiple outputs - check strategy
            if metrics_strategy == 'shared_metrics_all_outputs':
                self._add_shared_metrics_selection()
            else:
                # Different metrics per output
                self._add_multiple_metrics_selection(num_outputs, output_names)
    
    def _add_shared_metrics_selection(self):
        """Add shared metrics selection for all outputs."""
        metric_options = self._get_metric_options()
        
        self.addChild({
            'name': 'Metrics Selection',
            'type': 'group',
            'children': [
                {'name': 'selected_metrics', 'type': 'str', 'value': 'Accuracy', 'tip': 'Comma-separated list of metrics to use (e.g., "Accuracy,Top-K Categorical Accuracy")'},
                {'name': 'available_metrics', 'type': 'list', 'limits': metric_options, 'value': 'Accuracy', 'tip': 'Available metrics - select one to add to the list above'}
            ],
            'tip': 'Select metrics to apply to all model outputs'
        })
        
        # Connect the available metrics dropdown to add metrics to the text list
        metrics_selection = self.child('Metrics Selection')
        if metrics_selection and metrics_selection.child('available_metrics'):
            metrics_selection.child('available_metrics').sigValueChanged.connect(
                lambda: self._add_metric_to_selection('Metrics Selection')
            )
        
        # Add metric configuration for each selected metric
        self._add_selected_metrics_parameters('Metrics Selection')
    
    def _add_multiple_metrics_selection(self, num_outputs, output_names):
        """Add multiple metrics selections for different outputs."""        
        metric_options = self._get_metric_options()
        
        for i in range(num_outputs):
            output_name = output_names[i] if i < len(output_names) else f'output_{i+1}'
            
            self.addChild({
                'name': f'Output {i+1}: {output_name}',
                'type': 'group',
                'children': [
                    {'name': 'selected_metrics', 'type': 'str', 'value': 'Accuracy', 'tip': f'Comma-separated list of metrics for {output_name} (e.g., "Accuracy,Precision")'},
                    {'name': 'available_metrics', 'type': 'list', 'limits': metric_options, 'value': 'Accuracy', 'tip': f'Available metrics for {output_name} - select one to add to the list above'}
                ],
                'tip': f'Metrics configuration for output: {output_name}'
            })
            
            # Connect the available metrics dropdown to add metrics to the text list
            output_group = self.child(f'Output {i+1}: {output_name}')
            if output_group and output_group.child('available_metrics'):
                # Create a proper closure to capture the current parent_name value
                def create_callback(pname):
                    return lambda: self._add_metric_to_selection(pname)
                
                output_group.child('available_metrics').sigValueChanged.connect(
                    create_callback(f'Output {i+1}: {output_name}')
                )
            
            # Add metric parameters for this output
            self._add_selected_metrics_parameters(f'Output {i+1}: {output_name}')
    
    def _get_metric_options(self):
        """Get list of available metric names including custom ones."""
        base_options = [
            'Accuracy',
            'Categorical Accuracy',
            'Sparse Categorical Accuracy',
            'Top-K Categorical Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC',
            'Mean Squared Error',
            'Mean Absolute Error'
        ]
        
        # Add custom metrics if any
        if hasattr(self, '_custom_metric_functions'):
            custom_options = list(self._custom_metric_functions.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_metric_to_selection(self, parent_name):
        """Add selected metric from dropdown to the metrics list."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        available_metrics = parent.child('available_metrics')
        selected_metrics = parent.child('selected_metrics')
        
        if not available_metrics or not selected_metrics:
            return
            
        selected_metric = available_metrics.value()
        current_metrics = selected_metrics.value()
        
        # Parse current metrics
        metrics_list = [m.strip() for m in current_metrics.split(',') if m.strip()]
        
        # Add new metric if not already in list
        if selected_metric not in metrics_list:
            metrics_list.append(selected_metric)
            new_metrics_str = ', '.join(metrics_list)
            selected_metrics.setValue(new_metrics_str)
            
            # Update metric parameters
            self._update_metrics_parameters(parent_name)
    
    def _add_selected_metrics_parameters(self, parent_name):
        """Add configuration for selected metrics."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        # Connect selection change to parameter update
        if parent.child('selected_metrics'):
            parent.child('selected_metrics').sigValueChanged.connect(
                lambda: self._update_metrics_parameters(parent_name)
            )
        
        # Add initial parameters
        self._update_metrics_parameters(parent_name)
    
    def _update_metrics_parameters(self, parent_name):
        """Update metrics parameters based on selection."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        selected_metrics_str = parent.child('selected_metrics').value()
        selected_metrics = [m.strip() for m in selected_metrics_str.split(',') if m.strip()]
        
        # Remove existing metric configurations (except selected_metrics and available_metrics)
        existing_configs = []
        for child in parent.children():
            if child.name() not in ['selected_metrics', 'available_metrics']:
                existing_configs.append(child)
        
        for config in existing_configs:
            parent.removeChild(config)
        
        # Add configuration for each selected metric
        for metric_name in selected_metrics:
            metric_params = self._get_metric_parameters(metric_name)
            if metric_params:
                parent.addChild({
                    'name': f'{metric_name} Config',
                    'type': 'group',
                    'children': metric_params,
                    'tip': f'Configuration for {metric_name} metric'
                })
    
    def _get_metric_parameters(self, metric_name):
        """Get parameters for a specific metric."""
        # Check if it's a custom metric
        if hasattr(self, '_custom_metric_parameters') and metric_name in self._custom_metric_parameters:
            return self._custom_metric_parameters[metric_name]
        
        # Return built-in metric parameters
        metric_parameters = {
            'Accuracy': [
                {'name': 'name', 'type': 'str', 'value': 'accuracy', 'tip': 'Name for this metric'}
            ],
            'Categorical Accuracy': [
                {'name': 'name', 'type': 'str', 'value': 'categorical_accuracy', 'tip': 'Name for this metric'}
            ],
            'Sparse Categorical Accuracy': [
                {'name': 'name', 'type': 'str', 'value': 'sparse_categorical_accuracy', 'tip': 'Name for this metric'}
            ],
            'Top-K Categorical Accuracy': [
                {'name': 'name', 'type': 'str', 'value': 'top_5_accuracy', 'tip': 'Name for this metric'},
                {'name': 'k', 'type': 'int', 'value': 5, 'limits': (1, 100), 'tip': 'Number of top predictions to consider'}
            ],
            'Precision': [
                {'name': 'name', 'type': 'str', 'value': 'precision', 'tip': 'Name for this metric'},
                {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary precision (0 for first class, None for multiclass)'}
            ],
            'Recall': [
                {'name': 'name', 'type': 'str', 'value': 'recall', 'tip': 'Name for this metric'},
                {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary recall (0 for first class, None for multiclass)'}
            ],
            'F1 Score': [
                {'name': 'name', 'type': 'str', 'value': 'f1_score', 'tip': 'Name for this metric'},
                {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary F1 (0 for first class, None for multiclass)'}
            ],
            'AUC': [
                {'name': 'name', 'type': 'str', 'value': 'auc', 'tip': 'Name for this metric'},
                {'name': 'curve', 'type': 'list', 'limits': ['ROC', 'PR'], 'value': 'ROC', 'tip': 'Curve type (ROC or Precision-Recall)'},
                {'name': 'multi_class', 'type': 'list', 'limits': ['ovr', 'ovo'], 'value': 'ovr', 'tip': 'Multiclass strategy (one-vs-rest or one-vs-one)'}
            ],
            'Mean Squared Error': [
                {'name': 'name', 'type': 'str', 'value': 'mse', 'tip': 'Name for this metric'}
            ],
            'Mean Absolute Error': [
                {'name': 'name', 'type': 'str', 'value': 'mae', 'tip': 'Name for this metric'}
            ]
        }
        
        return metric_parameters.get(metric_name, [])
    
    def _add_preset_metrics(self):
        """Add preset metrics with their parameters."""
        preset_metrics = [
            {
                'name': 'Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Standard accuracy metric for classification tasks'
            },
            {
                'name': 'Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'categorical_accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Categorical accuracy metric for multi-class classification'
            },
            {
                'name': 'Sparse Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable sparse categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'sparse_categorical_accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Sparse categorical accuracy for integer label classification'
            },
            {
                'name': 'Top-K Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable top-k categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'top_5_accuracy', 'tip': 'Name for this metric'},
                    {'name': 'k', 'type': 'int', 'value': 5, 'limits': (1, 100), 'tip': 'Number of top predictions to consider'}
                ],
                'tip': 'Top-K accuracy metric (e.g., top-5 accuracy)'
            },
            {
                'name': 'Precision',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable precision metric'},
                    {'name': 'name', 'type': 'str', 'value': 'precision', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary precision (0 for first class, None for multiclass)'}
                ],
                'tip': 'Precision metric for classification tasks'
            },
            {
                'name': 'Recall',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable recall metric'},
                    {'name': 'name', 'type': 'str', 'value': 'recall', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary recall (0 for first class, None for multiclass)'}
                ],
                'tip': 'Recall metric for classification tasks'
            },
            {
                'name': 'F1 Score',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable F1 score metric'},
                    {'name': 'name', 'type': 'str', 'value': 'f1_score', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary F1 (0 for first class, None for multiclass)'}
                ],
                'tip': 'F1 score metric (harmonic mean of precision and recall)'
            },
            {
                'name': 'AUC',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable AUC metric'},
                    {'name': 'name', 'type': 'str', 'value': 'auc', 'tip': 'Name for this metric'},
                    {'name': 'curve', 'type': 'list', 'limits': ['ROC', 'PR'], 'value': 'ROC', 'tip': 'Curve type (ROC or Precision-Recall)'},
                    {'name': 'multi_class', 'type': 'list', 'limits': ['ovr', 'ovo'], 'value': 'ovr', 'tip': 'Multiclass strategy (one-vs-rest or one-vs-one)'}
                ],
                'tip': 'Area Under the Curve (AUC) metric'
            },
            {
                'name': 'Mean Squared Error',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable mean squared error metric'},
                    {'name': 'name', 'type': 'str', 'value': 'mse', 'tip': 'Name for this metric'}
                ],
                'tip': 'Mean squared error metric for regression tasks'
            },
            {
                'name': 'Mean Absolute Error',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable mean absolute error metric'},
                    {'name': 'name', 'type': 'str', 'value': 'mae', 'tip': 'Name for this metric'}
                ],
                'tip': 'Mean absolute error metric for regression tasks'
            }
        ]
        
        # Add all preset metrics
        for metric in preset_metrics:
            self.addChild(metric)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom metrics from files."""
        self.addChild({
            'name': 'Load Custom Metrics',
            'type': 'action',
            'tip': 'Click to load custom metrics from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Metrics')
        custom_button.sigActivated.connect(self._load_custom_metrics)
    
    def _load_custom_metrics(self):
        """Load custom metrics from a selected Python file."""
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


class MetricsGroup:
    def __init__(self, **opts):
        self.config = {
            'Metrics Selection': {
                'accuracy': True,
                'precision': False,
                'recall': False,
                'f1_score': False
            },
            'Custom Parameters': {}
        }
        self._custom_metrics = {}
        self._custom_metric_functions = {}
        
        # Add model output configuration first
        self._add_output_configuration()
        
    def _add_output_configuration(self):
        """Add output configuration for metrics."""
        # In CLI mode, output configuration is handled through the config system
        pass
    
    def load_custom_metrics(self, file_path):
        """Load custom metric functions from a file."""
        try:
            custom_functions = self._extract_metric_functions(file_path)
            
            if not custom_functions:
                cli_warning(
                    "No Functions Found",
                    "No valid metrics found in the selected file.\n\n"
                    "Functions should accept 'y_true' and 'y_pred' parameters or be TensorFlow metric classes."
                )
                return False
            
            # Add custom functions to the available metric options
            for func_name, func_info in custom_functions.items():
                self._add_custom_metric_option(func_name, func_info)
            
            cli_info(
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom metric(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese metrics are now available in the selection dropdowns."
            )
            return True
                
        except (OSError, SyntaxError) as e:
            cli_error(
                "Error Loading File",
                f"Failed to load custom metrics from file:\n{str(e)}"
            )
            return False
    
    def _add_custom_metric_option(self, func_name, func_info):
        """Add a custom metric as an option in dropdowns."""
        # Store custom metric function info for later use
        if not hasattr(self, '_custom_metric_functions'):
            self._custom_metric_functions = {}
        
        display_name = f"{func_name} (custom)"
        self._custom_metric_functions[display_name] = func_info
        
        # Add parameters for this custom metric
        params = []
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
            elif param_info['type'] == 'list' and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            params.append(param_config)
        
        # Add metadata parameters
        params.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'metric_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of metric (function or class)'}
        ])
        
        # Store parameters for this custom metric
        if not hasattr(self, '_custom_metric_parameters'):
            self._custom_metric_parameters = {}
        self._custom_metric_parameters[display_name] = params
    
    def _update_all_metrics_selections(self):
        """Update all metrics selection dropdowns with custom metrics."""
        # Get updated metric options
        metric_options = self._get_metric_options()
        
        # Find all metrics selection parameters and update their options
        for child in self.children():
            if child.name().startswith('Metrics Selection') or child.name().startswith('Output'):
                available_metrics = child.child('available_metrics')
                if available_metrics:
                    available_metrics.setLimits(metric_options)
    
    def _extract_metric_functions(self, file_path):
        """Extract valid metric functions from a Python file."""
        import ast
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if self._is_valid_metric_function(node):
                        func_info = {
                            'type': 'function',
                            'parameters': self._extract_function_parameters_metrics(node),
                            'docstring': ast.get_docstring(node) or f'Custom metric function: {node.name}',
                            'file_path': file_path,
                            'function_name': node.name
                        }
                        custom_functions[node.name] = func_info
                elif isinstance(node, ast.ClassDef):
                    if self._is_valid_metric_class(node):
                        func_info = {
                            'type': 'class',
                            'parameters': self._extract_class_parameters_metrics(node),
                            'docstring': ast.get_docstring(node) or f'Custom metric class: {node.name}',
                            'file_path': file_path,
                            'function_name': node.name
                        }
                        custom_functions[node.name] = func_info
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_metric_function(self, func_node):
        """Check if a function is a valid metric function."""
        # Check if function has at least two parameters (should be 'y_true', 'y_pred')
        if len(func_node.args.args) < 2:
            return False
        
        # Check if parameters are likely metric function parameters
        param_names = [arg.arg for arg in func_node.args.args]
        
        # Common metric function parameter names
        valid_patterns = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits'],
            ['ground_truth', 'predictions']
        ]
        
        for pattern in valid_patterns:
            if all(any(p in param.lower() for p in pattern) for param in param_names[:2]):
                return True
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return
    
    def _is_valid_metric_class(self, class_node):
        """Check if a class is a valid metric class."""
        class_name = class_node.name.lower()
        
        # Check class name for metric indicators
        if 'metric' in class_name or 'accuracy' in class_name or 'precision' in class_name or 'recall' in class_name or 'f1' in class_name:
            return True
        
        # Check if class has common TensorFlow metric methods
        metric_methods = ['update_state', 'result', 'reset_states', '__call__']
        class_method_names = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                class_method_names.append(node.name)
        
        # If it has at least 2 of the common metric methods, consider it a metric class
        method_matches = sum(1 for method in metric_methods if method in class_method_names)
        return method_matches >= 2
    
    def _extract_class_parameters_metrics(self, class_node):
        """Extract parameters from class __init__ method."""
        import ast
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
                    param_name = arg.arg
                    
                    # Try to infer parameter type and default values
                    param_info = {
                        'name': param_name,
                        'type': 'str',  # Default type
                        'default': '',   # Default value
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference based on parameter name
                    if 'num' in param_name.lower() or 'k' in param_name.lower() or 'classes' in param_name.lower():
                        param_info.update({'type': 'int', 'default': 1, 'limits': (1, 100)})
                    elif 'threshold' in param_name.lower() or 'alpha' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
                    elif 'name' in param_name.lower():
                        param_info.update({'type': 'str', 'default': class_node.name.lower()})
                    elif 'dtype' in param_name.lower():
                        param_info.update({'type': 'str', 'default': 'float32'})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters_metrics(self, func_node):
        """Extract parameters from metric function definition (excluding 'y_true', 'y_pred' parameters)."""
        import ast
        params = []
        
        # Skip the first two parameters (y_true, y_pred) and extract others
        for arg in func_node.args.args[2:]:
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
            if 'average' in param_name.lower():
                param_info.update({'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'default': 'macro'})
            elif 'threshold' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'k' in param_name.lower():
                param_info.update({'type': 'int', 'default': 5, 'limits': (1, 100)})
            elif 'class' in param_name.lower():
                param_info.update({'type': 'int', 'default': 0, 'limits': (0, 100)})
            elif 'name' in param_name.lower():
                param_info.update({'type': 'str', 'default': func_node.name})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 2  # -2 to skip y_true, y_pred
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
                
        # except Exception as e:
        #     cli_error(
        #         None,
        #         "Error Loading File",
        #         f"Failed to load custom metrics from file:\n{str(e)}"
        #     )
    
    def _add_custom_metric_option(self, func_name, func_info):
        """Add a custom metric as an option."""
        # Store custom metric function info for later use
        if not hasattr(self, '_custom_metric_functions'):
            self._custom_metric_functions = {}
        
        display_name = f"{func_name} (custom)"
        self._custom_metric_functions[display_name] = func_info
        
        # Add parameters for this custom metric
        params = []
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
            elif param_info['type'] == 'list' and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            params.append(param_config)
        
        # Add metadata parameters
        params.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'metric_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of metric (function or class)'}
        ])
        
        # Store parameters for this custom metric
        if not hasattr(self, '_custom_metric_parameters'):
            self._custom_metric_parameters = {}
        self._custom_metric_parameters[display_name] = params
    
    def _extract_metric_functions(self, file_path):
        """Extract valid metric functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid metric function
                    if self._is_valid_metric_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom metric function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a valid metric class
                    if self._is_valid_metric_class(node):
                        # Extract class parameters from __init__ method
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom metric class: {class_name}"
                        
                        custom_functions[class_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': class_name,
                            'type': 'class'
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_metric_function(self, func_node):
        """Check if a function is a valid metric function."""
        # Check if function has at least two parameters (should be 'y_true', 'y_pred')
        if len(func_node.args.args) < 2:
            return False
        
        # Check if parameters are likely metric function parameters
        param_names = [arg.arg for arg in func_node.args.args]
        
        # Common metric function parameter names
        valid_patterns = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits'],
            ['ground_truth', 'predictions']
        ]
        
        for pattern in valid_patterns:
            if all(any(p in param.lower() for p in pattern) for param in param_names[:2]):
                return True
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return
    
    def _is_valid_metric_class(self, class_node):
        """Check if a class is a valid metric class."""
        class_name = class_node.name.lower()
        
        # Check class name for metric indicators
        metric_indicators = ['metric', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'score']
        if any(indicator in class_name for indicator in metric_indicators):
            return True
        
        # Check if class has call method or update_state method (TensorFlow metric pattern)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name in ['__call__', 'update_state', 'result']:
                return True
        
        return False
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
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
                    if 'name' in param_name.lower():
                        param_info.update({'type': 'str', 'default': 'custom_metric'})
                    elif 'k' in param_name.lower() and len(param_name) <= 2:
                        param_info.update({'type': 'int', 'default': 5, 'limits': (1, 100)})
                    elif 'threshold' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
                    elif 'average' in param_name.lower():
                        param_info.update({'type': 'list', 'limits': ['micro', 'macro', 'weighted'], 'default': 'macro'})
                    elif 'class_id' in param_name.lower():
                        param_info.update({'type': 'int', 'default': None})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'y_true', 'y_pred' parameters)."""
        params = []
        
        # Skip the first two parameters (y_true, y_pred) and extract others
        for arg in func_node.args.args[2:]:
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
            if 'name' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'custom_metric'})
            elif 'k' in param_name.lower() and len(param_name) <= 2:
                param_info.update({'type': 'int', 'default': 5, 'limits': (1, 100)})
            elif 'threshold' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'average' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'macro'})
            elif 'class_id' in param_name.lower():
                param_info.update({'type': 'int', 'default': None})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 2  # -2 to skip y_true, y_pred
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
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

# Register the custom parameter types
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
    
    # def _add_custom_button(self):
    #     """Add a button parameter for loading custom augmentation functions from files."""
    #     self.addChild({
    #         'name': 'Load Custom Augmentations',
    #         'type': 'action',
    #         'tip': 'Click to load custom augmentation functions from a Python file'
    #     })
        
    #     # Connect the action to the file loading function
    #     custom_button = self.child('Load Custom Augmentations')
    #     custom_button.sigActivated.connect(self._load_custom_augmentations)
    
    # def _load_custom_augmentations(self):
    #     """Load custom augmentation functions from a selected Python file."""
    #     # CLI-only message functions
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
        
    #     # Open file dialog to select Python file
    #     file_path, _ = cli_get_file_path(
    #         None,
    #         "Select Python file with custom augmentation functions",
    #         "",
    #         "Python Files (*.py)"
    #     )
        
    #     if not file_path:
    #         return
        
    #     try:
    #         # Load and parse the Python file
    #         custom_functions = self._extract_augmentation_functions(file_path)
            
    #         if not custom_functions:
    #             cli_warning(
    #                 None,
    #                 "No Functions Found",
    #                 "No valid augmentation functions found in the selected file.\n\n"
    #                 "Functions should accept 'image' parameter and return modified image."
    #             )
    #             return
            
    #         # Add each found function as a custom augmentation
    #         added_count = 0
    #         for func_name, func_info in custom_functions.items():
    #             if self._add_custom_function(func_name, func_info):
    #                 added_count += 1
            
    #         if added_count > 0:
    #             cli_info(
    #                 None,
    #                 "Functions Loaded",
    #                 f"Successfully loaded {added_count} custom augmentation function(s):\n" +
    #                 "\n".join(custom_functions.keys())
    #             )
    #         else:
    #             cli_warning(
    #                 None,
    #                 "No New Functions",
    #                 "All functions from the file are already loaded or invalid."
    #             )
                
    #     except Exception as e:
    #         cli_error(
    #             None,
    #             "Error Loading File",
    #             f"Failed to load custom augmentations from file:\n{str(e)}"
    #         )
    
    # def _extract_augmentation_functions(self, file_path):
    #     """Extract valid augmentation functions from a Python file."""
    #     custom_functions = {}
        
    #     try:
    #         # Read and parse the file
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             content = f.read()
            
    #         # Parse the AST
    #         tree = ast.parse(content)
            
    #         # Find function definitions
    #         for node in ast.walk(tree):
    #             if isinstance(node, ast.FunctionDef):
    #                 func_name = node.name
                    
    #                 # Check if it's a valid augmentation function
    #                 if self._is_valid_augmentation_function(node):
    #                     # Extract function parameters
    #                     params = self._extract_function_parameters(node)
                        
    #                     # Extract docstring if available
    #                     docstring = ast.get_docstring(node) or f"Custom augmentation function: {func_name}"
                        
    #                     custom_functions[func_name] = {
    #                         'parameters': params,
    #                         'docstring': docstring,
    #                         'file_path': file_path,
    #                         'function_name': func_name
    #                     }
            
    #     except Exception as e:
    #         print(f"Error parsing file {file_path}: {e}")
        
    #     return custom_functions
    
    # def _is_valid_augmentation_function(self, func_node):
    #     """Check if a function is a valid augmentation function."""
    #     # Check if function has at least one parameter (should be 'image')
    #     if not func_node.args.args:
    #         return False
        
    #     # Check if first parameter is likely an image parameter
    #     first_param = func_node.args.args[0].arg
    #     if first_param not in ['image', 'img', 'x', 'data']:
    #         return False
        
    #     # Function should return something (basic check)
    #     has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
    #     if not has_return:
    #         return False
        
    #     return True
    
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
    
    # def _add_custom_function(self, func_name, func_info):
    #     """Add a custom function as an augmentation method."""
    #     # Add (custom) suffix to distinguish from presets
    #     display_name = f"{func_name} (custom)"
        
    #     # Check if function already exists (check both original and display names)
    #     existing_names = [child.name() for child in self.children()]
    #     if func_name in existing_names or display_name in existing_names:
    #         return False
        
    #     # Create parameters list
    #     children = [
    #         {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} augmentation'}
    #     ]
        
    #     # Add function-specific parameters
    #     for param_info in func_info['parameters']:
    #         children.append({
    #             'name': param_info['name'],
    #             'type': param_info['type'],
    #             'value': param_info['default'],
    #             'limits': param_info.get('limits'),
    #             'suffix': param_info.get('suffix', ''),
    #             'tip': param_info['tip']
    #         })
        
    #     # Add metadata parameters
    #     children.extend([
    #         {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
    #         {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
    #     ])
        
    #     # Create the augmentation method
    #     method_config = {
    #         'name': display_name,
    #         'type': 'group',
    #         'children': children,
    #         'removable': True,
    #         'renamable': False,  # Keep original function name
    #         'tip': func_info['docstring']
    #     }
        
    #     # Insert before the "Load Custom Augmentations" button
    #     # Find the button's index and insert before it
    #     button_index = None
    #     for i, child in enumerate(self.children()):
    #         if child.name() == 'Load Custom Augmentations':
    #             button_index = i
    #             break
        
    #     if button_index is not None:
    #         self.insertChild(button_index, method_config)
    #     else:
    #         # Fallback: add at the end if button not found
    #         self.addChild(method_config)
        
    #     return True
    
    def set_metrics_config(self, config):
        """Set the metrics configuration from loaded config data."""
        if not config or not isinstance(config, dict):
            return
        
        try:
            # Set Model Output Configuration
            output_config = config.get('Model Output Configuration', {})
            if output_config:
                model_output_group = self.child('Model Output Configuration')
                if model_output_group:
                    for param_name, param_value in output_config.items():
                        try:
                            param = model_output_group.child(param_name)
                            if param:
                                try:
                                    param.setValue(param_value)
                                except Exception as e:
                                    print(f"Warning: Could not set metrics output config parameter '{param_name}' to '{param_value}': {e}")
                            else:
                                print(f"Warning: Metrics output config parameter '{param_name}' not found")
                        except KeyError as e:
                            print(f"Warning: Metrics output config parameter '{param_name}' not found: {e}")
                        except Exception as e:
                            print(f"Warning: Error accessing metrics output config parameter '{param_name}': {e}")
                    
                    # Update metrics selection after setting output config
                    self._update_metrics_selection()
            
            # Set Metrics Selection configuration
            metrics_selection_config = config.get('Metrics Selection', {})
            if metrics_selection_config:
                metrics_selection_group = self.child('Metrics Selection')
                if metrics_selection_group:
                    # Set selected metrics
                    selected_metrics = metrics_selection_config.get('selected_metrics')
                    if selected_metrics:
                        metrics_selector = metrics_selection_group.child('selected_metrics')
                        if metrics_selector:
                            try:
                                metrics_selector.setValue(selected_metrics)
                                # Update available metrics and configs after setting
                                self._update_metrics_selection()
                            except Exception as e:
                                print(f"Warning: Could not set selected metrics to '{selected_metrics}': {e}")
                    
                    # Set available metrics
                    available_metrics = metrics_selection_config.get('available_metrics')
                    if available_metrics:
                        available_selector = metrics_selection_group.child('available_metrics')
                        if available_selector:
                            try:
                                available_selector.setValue(available_metrics)
                            except Exception as e:
                                print(f"Warning: Could not set available metrics to '{available_metrics}': {e}")
                    
                    # Set metric-specific configurations
                    for param_name, param_value in metrics_selection_config.items():
                        if param_name not in ['selected_metrics', 'available_metrics'] and param_name.endswith(' Config'):
                            # This is a metric configuration group
                            try:
                                metric_config_group = metrics_selection_group.child(param_name)
                                if metric_config_group and isinstance(param_value, dict):
                                    for sub_param_name, sub_param_value in param_value.items():
                                        try:
                                            sub_param = metric_config_group.child(sub_param_name)
                                            if sub_param:
                                                try:
                                                    sub_param.setValue(sub_param_value)
                                                except Exception as e:
                                                    print(f"Warning: Could not set metric config '{param_name}.{sub_param_name}' to '{sub_param_value}': {e}")
                                            else:
                                                print(f"Warning: Metric config parameter '{param_name}.{sub_param_name}' not found")
                                        except KeyError as e:
                                            print(f"Warning: Metric config parameter '{param_name}.{sub_param_name}' not found in group: {e}")
                                        except Exception as e:
                                            print(f"Warning: Error accessing metric config parameter '{param_name}.{sub_param_name}': {e}")
                                else:
                                    print(f"Warning: Metric config group '{param_name}' not found or invalid format")
                            except KeyError as e:
                                print(f"Warning: Metric config group '{param_name}' not found: {e}")
                            except Exception as e:
                                print(f"Warning: Error accessing metric config group '{param_name}': {e}")
            
            # Handle multiple outputs if they exist
            for child in self.children():
                if child.name().startswith('Output'):
                    # This is a per-output metrics configuration
                    output_config_data = config.get(child.name(), {})
                    if output_config_data:
                        # Set selected metrics
                        selected_metrics = output_config_data.get('selected_metrics')
                        if selected_metrics:
                            metrics_selector = child.child('selected_metrics')
                            if metrics_selector:
                                try:
                                    metrics_selector.setValue(selected_metrics)
                                except Exception as e:
                                    print(f"Warning: Could not set metrics for {child.name()}: {e}")
                        
                        # Set other parameters including metric configs
                        for param_name, param_value in output_config_data.items():
                            if param_name not in ['selected_metrics']:
                                if param_name.endswith(' Config') and isinstance(param_value, dict):
                                    try:
                                        metric_config_group = child.child(param_name)
                                        if metric_config_group:
                                            for sub_param_name, sub_param_value in param_value.items():
                                                try:
                                                    sub_param = metric_config_group.child(sub_param_name)
                                                    if sub_param:
                                                        try:
                                                            sub_param.setValue(sub_param_value)
                                                        except Exception as e:
                                                            print(f"Warning: Could not set {child.name()} config '{param_name}.{sub_param_name}': {e}")
                                                    else:
                                                        print(f"Warning: Config parameter '{param_name}.{sub_param_name}' not found for {child.name()} (metric may have changed)")
                                                except KeyError:
                                                    print(f"Warning: Config parameter '{param_name}.{sub_param_name}' not found for {child.name()} (metric may have changed)")
                                        else:
                                            print(f"Warning: Config group '{param_name}' not found for {child.name()} (metric may have changed)")
                                    except KeyError:
                                        print(f"Warning: Config group '{param_name}' not found for {child.name()} (metric may have changed)")
                                else:
                                    try:
                                        param = child.child(param_name)
                                        if param:
                                            try:
                                                param.setValue(param_value)
                                            except Exception as e:
                                                print(f"Warning: Could not set {child.name()} parameter '{param_name}': {e}")
                                        else:
                                            print(f"Warning: Parameter '{param_name}' not found for {child.name()} (metrics may have changed)")
                                    except KeyError:
                                        print(f"Warning: Parameter '{param_name}' not found for {child.name()} (metrics may have changed)")
                        
        except Exception as e:
            print(f"Error setting metrics configuration: {e}")
            import traceback
            traceback.print_exc()
    
    def load_custom_metric_from_metadata(self, metric_info):
        """Load custom metric from metadata info."""
        try:
            file_path = metric_info.get('file_path', '')
            function_name = metric_info.get('function_name', '') or metric_info.get('original_name', '')
            metric_type = metric_info.get('type', 'function')
            
            # Check for empty function name
            if not function_name:
                print(f"Warning: Empty function name in custom metric metadata for {file_path}")
                return False
            
            if not os.path.exists(file_path):
                print(f"Warning: Custom metric file not found: {file_path}")
                return False
            
            # Extract metrics from the file
            custom_functions = self._extract_metric_functions(file_path)
            
            # Find the specific function we need
            target_function = None
            for func_name, func_info in custom_functions.items():
                if func_info['function_name'] == function_name:
                    target_function = func_info
                    break
            
            if not target_function:
                print(f"Warning: Function '{function_name}' not found in {file_path}")
                return False
            
            # Add the custom metric function
            self._add_custom_metric_option(function_name, target_function)
            
            # Update all metric selection dropdowns
            self._update_all_metrics_selections()
            
            print(f"Successfully loaded custom metric: {function_name}")
            return True
            
        except Exception as e:
            print(f"Error loading custom metric from metadata: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # def addNew(self, typ=None):
    #     """Legacy method - no longer used since we load from files."""
    #     # This method is called by the parameter tree system but we use the button instead
    #     pass

