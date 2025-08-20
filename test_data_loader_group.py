#!/usr/bin/env python3
"""
Test script for Data Loader Group functionality in ModelGardener.

This script tests the data loader group integration and custom data loader loading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_group import DataLoaderGroup
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pyqtgraph.parametertree import ParameterTree, Parameter


def test_data_loader_group():
    """Test basic data loader group functionality."""
    
    # Create QApplication if not already exists
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    window.setCentralWidget(central_widget)
    
    # Create parameter tree
    tree = ParameterTree()
    layout.addWidget(tree)
    
    # Create data loader group parameter
    data_loader_param = DataLoaderGroup(name='Data Loader Configuration')
    
    # Create root parameter and add data loader group
    root_param = Parameter.create(name='Test Configuration', type='group')
    root_param.addChild(data_loader_param)
    
    # Set parameter tree
    tree.setParameters(root_param, showTop=False)
    
    # Show window
    window.setWindowTitle("Data Loader Group Test")
    window.resize(800, 600)
    window.show()
    
    print("Data Loader Group test window created successfully!")
    print("Available data loaders:", data_loader_param._get_data_loader_options())
    
    # Test getting configuration
    config = data_loader_param.get_data_loader_config()
    print("Initial configuration:", config)
    
    return app, window


if __name__ == "__main__":
    try:
        app, window = test_data_loader_group()
        
        print("\nTest successful! Data Loader Group is working correctly.")
        print("You can test loading custom data loaders by clicking the 'Load Custom Data Loader' button.")
        print("Try loading the example file: example_funcs/example_custom_data_loaders.py")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
