#!/usr/bin/env python3
"""
Test the new OptimizerGroup functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from optimizer_group import OptimizerGroup

class TestOptimizerGroupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptimizerGroup Test")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create the optimizer group parameter
        self.optimizer_group = OptimizerGroup(name='optimizer_test')
        
        # Create parameter tree
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setParameters(self.optimizer_group, showTop=False)
        layout.addWidget(self.param_tree)
        
        print("=== OptimizerGroup Test ===")
        self.run_initial_tests()
        
    def run_initial_tests(self):
        """Run initial tests on the optimizer group."""
        print("1. Optimizer group created successfully")
        
        # Check optimizer selection
        selection_group = self.optimizer_group.child('Optimizer Selection')
        if selection_group:
            optimizer_selector = selection_group.child('selected_optimizer')
            if optimizer_selector:
                print(f"2. Current optimizer: {optimizer_selector.value()}")
                print(f"3. Available optimizers: {optimizer_selector.opts['limits']}")
                
                # Test changing optimizer
                print("4. Testing optimizer changes:")
                for opt in ['SGD', 'RMSprop', 'AdamW']:
                    if opt in optimizer_selector.opts['limits']:
                        optimizer_selector.setValue(opt)
                        print(f"   Changed to {opt}")
                        # Print current parameters
                        params = []
                        for child in selection_group.children():
                            if child.name() != 'selected_optimizer':
                                params.append(f"{child.name()}={child.value()}")
                        print(f"     Parameters: {', '.join(params) if params else 'None'}")
            else:
                print("ERROR: Optimizer selector not found")
        else:
            print("ERROR: Optimizer selection group not found")
        
        # Check custom optimizer loading
        load_button = self.optimizer_group.child('Load Custom Optimizer')
        if load_button:
            print("5. Custom optimizer loading button found")
        else:
            print("ERROR: Custom optimizer loading button not found")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = TestOptimizerGroupWindow()
    window.show()
    
    app.exec()
