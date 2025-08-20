#!/usr/bin/env python3
"""
Test script for the new augmentation configuration system.
This will create a simple window to test the AugmentationGroup parameter.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QPlainTextEdit
from pyqtgraph.parametertree import Parameter, ParameterTree

# Import our custom augmentation group class
from main import AugmentationGroup, get_parameter_tooltip
import pyqtgraph.parametertree.parameterTypes as pTypes

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Augmentation Group Test")
        self.resize(800, 600)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Register the custom parameter type
        pTypes.registerParameterType('augmentation_group', AugmentationGroup, override=True)
        
        # Create augmentation parameter
        self.aug_param = Parameter.create(
            name='Data Augmentation',
            type='augmentation_group',
            tip='Configure data augmentation methods'
        )
        
        # Create parameter tree
        self.tree = ParameterTree()
        self.tree.setParameters(self.aug_param, showTop=False)
        layout.addWidget(self.tree)
        
        # Add test button
        self.test_btn = QPushButton("Export Configuration")
        self.test_btn.clicked.connect(self.export_config)
        layout.addWidget(self.test_btn)
        
        # Add output text area
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        
        # Connect to parameter changes
        self.aug_param.sigTreeStateChanged.connect(self.on_param_changed)
    
    def on_param_changed(self, param, changes):
        """Handle parameter changes."""
        self.output.appendPlainText(f"Parameter changed: {changes}")
    
    def export_config(self):
        """Export the current configuration to see the structure."""
        config = {}
        
        for method_param in self.aug_param.children():
            method_name = method_param.name()
            method_config = {}
            
            # Extract configuration for each augmentation method
            for param in method_param.children():
                method_config[param.name()] = param.value()
            
            config[method_name] = method_config
        
        self.output.clear()
        self.output.appendPlainText("Current Augmentation Configuration:")
        self.output.appendPlainText("=" * 50)
        
        for method, params in config.items():
            self.output.appendPlainText(f"\n{method}:")
            for param_name, value in params.items():
                self.output.appendPlainText(f"  {param_name}: {value}")

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
