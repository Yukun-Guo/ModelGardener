import ast
import os

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
            'Output Configuration': {
                'num_outputs': 1,
                'output_names': ['output']
            },
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
        
    def get_config(self):
        """Get the current metrics configuration."""
        return dict(self.config)
        
    def set_config(self, config):
        """Set the metrics configuration."""
        self.config.update(config)
        
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
        """Add a custom metric as an option."""
        display_name = f"{func_name} (custom)"
        self._custom_metric_functions[display_name] = func_info
        
    def _extract_metric_functions(self, file_path):
        """Extract metric functions from a Python file."""
        custom_functions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            tree = ast.parse(file_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    if self._is_valid_metric_function(node):
                        func_info = {
                            'name': func_name,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function',
                            'parameters': self._extract_function_parameters(node),
                            'description': ast.get_docstring(node) or f'Custom metric: {func_name}'
                        }
                        custom_functions[func_name] = func_info
                        
        except (OSError, SyntaxError) as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_metric_function(self, func_node):
        """Check if a function is a valid metric function."""
        # Check if function has y_true and y_pred parameters
        arg_names = [arg.arg for arg in func_node.args.args]
        return 'y_true' in arg_names and 'y_pred' in arg_names
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition."""
        params = []
        
        for arg in func_node.args.args:
            if arg.arg not in ['y_true', 'y_pred', 'self']:
                param_info = {
                    'name': arg.arg,
                    'type': 'str',
                    'default': None,
                    'tip': f'Parameter for custom metric: {arg.arg}'
                }
                params.append(param_info)
                
        return params
