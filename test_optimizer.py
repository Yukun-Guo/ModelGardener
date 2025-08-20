#!/usr/bin/env python3
"""
Test script for optimizer configuration functionality.
This script tests the optimizer group parameter functionality independently.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from optimizer_group import OptimizerGroupWidget

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimizer Configuration Test")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create a dummy parameter for the optimizer widget
        from pyqtgraph.parametertree import Parameter
        from optimizer_group import OptimizerGroupParameter
        
        # Create the optimizer parameter
        param = OptimizerGroupParameter(name='optimizer', type='optimizer_group')
        
        # Create parameter tree
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setParameters(param, showTop=False)
        layout.addWidget(self.param_tree)
        
        # Test getting parameters
        print("Testing optimizer parameter...")
        print(f"Current value: {param.value()}")
        
        # Access the widget through the parameter
        widget = param.opts.get('widget')
        if widget and hasattr(widget, 'optimizer_combo'):
            print(f"Available optimizers: {[widget.optimizer_combo.itemText(i) for i in range(widget.optimizer_combo.count())]}")
            
            # Test setting different optimizer
            widget.optimizer_combo.setCurrentText("SGD")
            print(f"After setting SGD: {param.value()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = TestWindow()
    window.show()
    
    app.exec()
