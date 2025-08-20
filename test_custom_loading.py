#!/usr/bin/env python3
"""
Test the custom augmentation file loading functionality.
"""

import sys
import ast
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QPlainTextEdit
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes

# Test the AugmentationGroup functionality
class TestAugmentationGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add a simple test to verify file parsing works
        self._test_file_parsing()
    
    def _test_file_parsing(self):
        """Test the file parsing functionality."""
        import os
        
        # Test with the example file we created
        test_file = "example_custom_augmentations.py"
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                functions_found = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if self._is_valid_augmentation_function(node):
                            functions_found.append(func_name)
                
                print(f"Found {len(functions_found)} valid augmentation functions:")
                for func in functions_found:
                    print(f"  - {func}")
                    
            except Exception as e:
                print(f"Error parsing test file: {e}")
        else:
            print("Test file not found")
    
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

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Custom Augmentation Loading")
        self.resize(600, 400)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Register the test parameter type
        pTypes.registerParameterType('test_augmentation_group', TestAugmentationGroup, override=True)
        
        # Create augmentation parameter
        self.aug_param = Parameter.create(
            name='Test Augmentation Loading',
            type='test_augmentation_group'
        )
        
        # Create parameter tree
        self.tree = ParameterTree()
        self.tree.setParameters(self.aug_param, showTop=False)
        layout.addWidget(self.tree)
        
        # Add output text area
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        
        # Show initial test results
        self.output.appendPlainText("Custom augmentation file loading test completed.")
        self.output.appendPlainText("Check console output for results.")

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
