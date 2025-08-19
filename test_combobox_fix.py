#!/usr/bin/env python3
"""
Test script to verify that combobox parameters now show option items correctly.
This script creates a simple parameter tree with list parameters to verify the fix.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from pyqtgraph.parametertree import Parameter, ParameterTree

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combobox Parameter Test")
        self.setGeometry(100, 100, 400, 600)
        
        # Create test parameters
        params = [
            {
                'name': 'Model Configuration',
                'type': 'group',
                'children': [
                    {
                        'name': 'backbone_type',
                        'type': 'list',
                        'limits': ['resnet', 'efficientnet', 'mobilenet', 'vit', 'densenet'],  # Using 'limits' (correct)
                        'value': 'resnet'
                    },
                    {
                        'name': 'activation',
                        'type': 'list',
                        'limits': ['relu', 'swish', 'gelu', 'leaky_relu', 'tanh'],  # Using 'limits' (correct)
                        'value': 'relu'
                    },
                    {
                        'name': 'optimizer_type',
                        'type': 'list',
                        'limits': ['sgd', 'adam', 'adamw', 'rmsprop', 'lars'],  # Using 'limits' (correct)
                        'value': 'adam'
                    },
                    {
                        'name': 'mixed_precision',
                        'type': 'list',
                        'limits': ['None', 'float16', 'bfloat16'],  # Using 'limits' (correct)
                        'value': 'None'
                    }
                ]
            },
            {
                'name': 'Test Wrong Usage',
                'type': 'group',
                'children': [
                    {
                        'name': 'wrong_param',
                        'type': 'list',
                        'values': ['option1', 'option2', 'option3'],  # Using 'values' (wrong) - should be empty
                        'value': 'option1'
                    }
                ]
            }
        ]
        
        # Create parameter tree
        self.param = Parameter.create(name='Test Parameters', type='group', children=params)
        self.tree = ParameterTree()
        self.tree.setParameters(self.param, showTop=False)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Test: Combobox Parameters"))
        layout.addWidget(QLabel("The first group should show dropdown options."))
        layout.addWidget(QLabel("The second group (wrong usage) should NOT show options."))
        layout.addWidget(self.tree)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Connect to parameter changes to print selected values
        self.param.sigTreeStateChanged.connect(self.on_param_changed)
    
    def on_param_changed(self, param, changes):
        """Print parameter changes to verify functionality."""
        print("Parameter changes:")
        for param_obj, change, data in changes:
            if change == 'value':
                print(f"  {param_obj.name()}: {data}")

def main():
    app = QApplication(sys.argv)
    
    window = TestWindow()
    window.show()
    
    print("Test Window Created!")
    print("Expected behavior:")
    print("1. Parameters in 'Model Configuration' should show dropdown options")
    print("2. Parameter in 'Test Wrong Usage' should NOT show dropdown options")
    print("3. Selecting different options should print the changes to console")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
