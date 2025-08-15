#!/usr/bin/env python3

"""
Simple test to verify the ParameterTree functionality
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes

# Test the custom DirectoryParameter - simpler approach using GroupParameter
class DirectoryParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        # Store the directory value
        self.directory_value = opts.get('value', '')
        
        # Create as a group parameter
        opts['type'] = 'group' 
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add path display (read-only text)
        self.addChild({
            'name': 'path', 
            'type': 'str', 
            'value': self.directory_value,
            'readonly': True
        })
        
        # Add browse action button
        self.addChild({
            'name': 'browse',
            'type': 'action',
            'title': '...'
        })
        
        # Connect the action
        self.param('browse').sigActivated.connect(self._browse_directory)
        
    def _browse_directory(self):
        """Open directory browser dialog."""
        from PySide6.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.name()}")
        if directory:
            self.directory_value = directory
            self.param('path').setValue(directory)
    
    def value(self):
        """Return the current directory value."""
        return self.directory_value
    
    def setValue(self, value):
        """Set the directory value."""
        self.directory_value = value
        if self.param('path'):
            self.param('path').setValue(value)

# Register the custom parameter type
pTypes.registerParameterType('directory', DirectoryParameter, override=True)

def dict_to_params(data, name="Config"):
    """Convert a nested dictionary to Parameter tree structure."""
    if isinstance(data, dict):
        children = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary - create a group
                children.append(dict_to_params(value, key))
            else:
                # Check for special directory parameters
                if key in ['train_dir', 'val_dir'] and isinstance(value, str):
                    # Create directory browser parameter with clean name
                    children.append({
                        'name': key,
                        'type': 'directory',
                        'value': value
                    })
                else:
                    # Leaf value - determine type and create parameter
                    param_type = 'str'  # default
                    if isinstance(value, bool):
                        param_type = 'bool'
                    elif isinstance(value, int):
                        param_type = 'int'
                    elif isinstance(value, float):
                        param_type = 'float'
                    
                    children.append({
                        'name': key,
                        'type': param_type,
                        'value': value
                    })
        
        return {
            'name': name,
            'type': 'group',
            'children': children
        }
    else:
        # Single value
        param_type = 'str'
        if isinstance(data, bool):
            param_type = 'bool'
        elif isinstance(data, int):
            param_type = 'int'
        elif isinstance(data, float):
            param_type = 'float'
        
        return {
            'name': name,
            'type': param_type,
            'value': data
        }

def params_to_dict(param):
    """Convert Parameter tree back to dictionary."""
    if param.hasChildren():
        result = {}
        for child in param.children():
            child_name = child.name()
            
            # Handle directory parameters (they have 'path' and 'browse' children)
            if isinstance(child, DirectoryParameter):
                result[child_name] = child.value()
            else:
                result[child_name] = params_to_dict(child)
        return result
    else:
        return param.value()

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameter Tree Test")
        self.resize(800, 600)
        
        # Test config
        test_config = {
            "data": {"train_dir": "/path/to/train", "val_dir": "/path/to/val"},
            "input": {"image_size": 224, "batch_size": 16, "shuffle": True},
            "train": {"epochs": 5, "learning_rate": 1e-3, "model_dir": "./model_dir"}
        }
        
        # Create parameter tree
        self.params = Parameter.create(**dict_to_params(test_config, "Test Config"))
        self.tree = ParameterTree()
        self.tree.setParameters(self.params, showTop=False)
        
        # Set up UI
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tree)
        self.setCentralWidget(central_widget)

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
