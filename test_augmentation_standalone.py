#!/usr/bin/env python3
"""
Standalone test for the augmentation configuration system.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QPlainTextEdit
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes

# Custom augmentation group that includes preset methods and allows adding custom methods
class AugmentationGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add Custom Augmentation"
        opts['addList'] = [
            'Custom Rotation',
            'Custom Noise',
            'Custom Blur', 
            'Custom Distortion',
            'Custom Filter'
        ]
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset augmentation methods
        self._add_preset_augmentations()
    
    def _add_preset_augmentations(self):
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
                'name': 'Color Jittering',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable color jittering'},
                    {'name': 'hue_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 50), 'tip': 'Maximum hue shift'},
                    {'name': 'sat_shift_limit', 'type': 'int', 'value': 30, 'limits': (0, 100), 'tip': 'Maximum saturation shift'},
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
                    {'name': 'probability', 'type': 'float', 'value': 1.0, 'limits': (0.0, 1.0), 'tip': 'Probability of random cropping'}
                ],
                'tip': 'Randomly crop parts of the image'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    def addNew(self, typ):
        """Add a new custom augmentation method."""
        custom_methods = {
            'Custom Rotation': {
                'name': f'Custom Rotation {len([c for c in self.children() if "Custom Rotation" in c.name()]) + 1}',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable this custom rotation'},
                    {'name': 'min_angle', 'type': 'float', 'value': -45.0, 'suffix': '°', 'tip': 'Minimum rotation angle'},
                    {'name': 'max_angle', 'type': 'float', 'value': 45.0, 'suffix': '°', 'tip': 'Maximum rotation angle'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of applying rotation'}
                ],
                'removable': True,
                'renamable': True,
                'tip': 'Custom rotation augmentation with specified angle range'
            },
            'Custom Noise': {
                'name': f'Custom Noise {len([c for c in self.children() if "Custom Noise" in c.name()]) + 1}',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable this custom noise'},
                    {'name': 'noise_type', 'type': 'list', 'values': ['gaussian', 'uniform', 'salt_pepper'], 'value': 'gaussian', 'tip': 'Type of noise to add'},
                    {'name': 'intensity', 'type': 'float', 'value': 0.05, 'limits': (0.0, 0.2), 'tip': 'Noise intensity'},
                    {'name': 'probability', 'type': 'float', 'value': 0.3, 'limits': (0.0, 1.0), 'tip': 'Probability of adding noise'}
                ],
                'removable': True,
                'renamable': True,
                'tip': 'Custom noise augmentation'
            }
        }
        
        if typ in custom_methods:
            self.addChild(custom_methods[typ])

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
        
        # Display initial state
        self.export_config()
    
    def on_param_changed(self, param, changes):
        """Handle parameter changes."""
        for param, change, data in changes:
            self.output.appendPlainText(f"Changed: {param.name()} -> {change}: {data}")
    
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
            enabled = params.get('enabled', False)
            status = "✓ ENABLED" if enabled else "✗ DISABLED"
            self.output.appendPlainText(f"\n{method} [{status}]:")
            for param_name, value in params.items():
                if param_name != 'enabled':
                    self.output.appendPlainText(f"  {param_name}: {value}")

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
