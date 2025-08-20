#!/usr/bin/env python3
"""
Comprehensive test script for the optimizer functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import pyqtgraph as pg
from optimizer_group import OptimizerGroupParameter

class ComprehensiveOptimizerTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comprehensive Optimizer Test")
        self.setGeometry(100, 100, 900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create the optimizer parameter
        from pyqtgraph.parametertree import Parameter
        
        self.optimizer_param = OptimizerGroupParameter(name='optimizer', type='optimizer_group')
        
        # Create parameter tree
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setParameters(self.optimizer_param, showTop=False)
        layout.addWidget(self.param_tree)
        
        # Test buttons
        test_button = QPushButton("Run Tests")
        test_button.clicked.connect(self.run_tests)
        layout.addWidget(test_button)
        
        print("=== Optimizer Configuration Test ===")
        self.run_initial_tests()
        
    def run_initial_tests(self):
        """Run initial tests on the optimizer configuration."""
        print(f"1. Initial configuration: {self.optimizer_param.value()}")
        
        # Access the widget
        widget = None
        for item in self.param_tree.listAllItems():
            if hasattr(item, 'widget') and item.widget:
                widget = item.widget
                break
        
        if widget:
            print(f"2. Widget found: {type(widget)}")
            print(f"3. Available optimizers: {[widget.optimizer_combo.itemText(i) for i in range(widget.optimizer_combo.count())]}")
            
            # Test switching optimizers
            print("4. Testing optimizer switching:")
            for optimizer_name in ["SGD", "RMSprop", "Adagrad"]:
                if widget.optimizer_combo.findText(optimizer_name) >= 0:
                    widget.optimizer_combo.setCurrentText(optimizer_name)
                    widget.on_optimizer_changed(optimizer_name)
                    print(f"   {optimizer_name}: {self.optimizer_param.value()}")
        else:
            print("ERROR: Widget not found!")
            
    def run_tests(self):
        """Run comprehensive tests."""
        print("\n=== Running Comprehensive Tests ===")
        
        # Test config serialization
        config = self.optimizer_param.value()
        print(f"Current config: {config}")
        
        # Test custom optimizer file existence
        custom_file = os.path.join(os.path.dirname(__file__), "example_custom_optimizers.py")
        if os.path.exists(custom_file):
            print(f"Custom optimizer file exists: {custom_file}")
        else:
            print("Custom optimizer file not found")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = ComprehensiveOptimizerTest()
    window.show()
    
    app.exec()
