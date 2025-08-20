# gui_tf_trainer_tfmodels.py
# Full GUI that uses tf-models-official train_lib.run_experiment
# NOTE: adapt experiment_name to the task you want (default "image_classification_imagenet")

import os
import sys
import json
import yaml
import threading
import subprocess
import time
import copy
import importlib.util
import inspect
import ast
from typing import Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QPlainTextEdit, QLabel, QMessageBox, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, 
    QGridLayout, QProgressBar, QToolBar, QLineEdit,QSizePolicy,
    QTreeWidget
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtWebEngineWidgets import QWebEngineView
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
import numpy as np
import tensorflow_models as tfm

import cv2
import albumentations as A
ALBU_AVAILABLE = True


import tensorflow as tf
TF_AVAILABLE = True
from official.core import exp_factory, train_lib
from official.core import config_definitions as cfg_def  # for constructing base configs if needed

# ---------------------------
# Helper function to convert config dict to ParameterTree structure
# ---------------------------

def create_comprehensive_config():
    """Create a comprehensive configuration structure with Basic and Advanced sections."""
    
    # Basic Configuration - Most commonly used parameters
    basic_config = {
        'data': {
            'train_dir': '',
            'val_dir': '',
            'image_size': [224, 224],
            'batch_size': 32,
            'num_classes': 1000,
            'shuffle': True
        },
        'model': {
            'backbone_type': 'resnet',
            'model_id': 50,
            'dropout_rate': 0.0,
            'activation': 'relu'
        },
        'training': {
            'epochs': 100,
            'learning_rate_type': 'exponential',
            'initial_learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'label_smoothing': 0.0
        },
        'runtime': {
            'model_dir': './model_dir',
            'distribution_strategy': 'mirrored',
            'mixed_precision': None,
            'num_gpus': 0
        }
    }
    
    # Advanced Configuration - Expert-level parameters
    advanced_config = {
        'model_advanced': {
            'depth_multiplier': 1.0,
            'stem_type': 'v0',
            'se_ratio': 0.0,
            'stochastic_depth_drop_rate': 0.0,
            'scale_stem': True,
            'resnetd_shortcut': False,
            'replace_stem_max_pool': False,
            'bn_trainable': True,
            'use_sync_bn': False,
            'norm_momentum': 0.99,
            'norm_epsilon': 0.001,
            'add_head_batch_norm': False,
            'kernel_initializer': 'random_uniform',
            'output_softmax': False
        },
        'data_advanced': {
            'tfds_name': '',
            'tfds_split': '',
            'cache': False,
            'shuffle_buffer_size': 10000,
            'cycle_length': 10,
            'block_length': 1,
            'drop_remainder': True,
            'sharding': True,
            'prefetch_buffer_size': None,
            'dtype': 'float32',
            'file_type': 'tfrecord',
            'image_field_key': 'image/encoded',
            'label_field_key': 'image/class/label',
            'decode_jpeg_only': True
        },
        'augmentation': {
            'type': 'augmentation_group',
            'name': 'augmentation'
        },
        'preprocessing': {
            'type': 'preprocessing_group', 
            'name': 'preprocessing'
        },
        'callbacks': {
            'type': 'callbacks_group',
            'name': 'callbacks'
        },
        'loss_functions': {
            'type': 'loss_functions_group',
            'name': 'loss_functions'
        },
        'metrics': {
            'type': 'metrics_group',
            'name': 'metrics'
        },
        'training_advanced': {
            'train_tf_while_loop': True,
            'train_tf_function': True,
            'eval_tf_function': True,
            'steps_per_loop': 1000,
            'summary_interval': 1000,
            'checkpoint_interval': 1000,
            'max_to_keep': 5,
            'validation_interval': 1000,
            'validation_steps': -1,
            'loss_upper_bound': 1000000.0,
            'one_hot_labels': True,
            'use_binary_cross_entropy': False,
            'soft_labels': False
        },
        'evaluation': {
            'top_k': 5,
            'report_per_class_metrics': False,
            'best_checkpoint_metric': '',
            'best_checkpoint_export_subdir': '',
            'best_checkpoint_metric_comp': 'higher'
        },
        'runtime_advanced': {
            'enable_xla': False,
            'run_eagerly': False,
            'per_gpu_thread_count': 0,
            'num_packs': 1,
            'loss_scale': None,
            'batchnorm_spatial_persistent': False,
            'tpu_settings': None,
            'all_reduce_alg': None
        }
    }
    
    return {
        'basic': basic_config,
        'advanced': advanced_config
    }

# Custom widget for directory-only browsing (no file button)
class DirectoryOnlyBrowseWidget(QWidget):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sigChanged = None  # No change signal needed for this custom widget
        
        # Set minimum height to ensure widget stays visible
        self.setMinimumHeight(25)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Text field to show current path
        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(param.value()))
        self.lineEdit.textChanged.connect(self._on_text_changed)
        
        # Browse directory button only
        browse_dir_btn = QPushButton("Dir...")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setMinimumWidth(50)
        browse_dir_btn.clicked.connect(self._browse_directory)
        
        layout.addWidget(self.lineEdit)
        layout.addWidget(browse_dir_btn)
    
    def _on_text_changed(self, text):
        """Handle manual text changes in the line edit."""
        self.param.setValue(text)
    
    def _browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.param.name()}")
        if directory:
            self.lineEdit.setText(directory)
            self.param.setValue(directory)
    
    def value(self):
        """Return current value from the line edit."""
        return self.lineEdit.text()
    
    def setValue(self, value):
        """Set the value in the line edit."""
        self.lineEdit.setText(str(value))
    
    def focusInEvent(self, event):
        """Handle focus in event - ensure widget stays visible."""
        super().focusInEvent(event)
        self.lineEdit.setFocus()
    
    def focusOutEvent(self, event):
        """Handle focus out event - ensure widget stays visible."""
        super().focusOutEvent(event)
        # Don't hide the widget when losing focus
        self.show()

class DirectoryOnlyParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        widget = DirectoryOnlyBrowseWidget(self.param)
        # Ensure the widget is always visible
        widget.setVisible(True)
        return widget
    
    def valueChanged(self, param, data, info=None, force=False):
        """Handle external value changes."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.setValue(data)
            # Ensure widget stays visible after value change
            self.widget.show()
    
    def showEditor(self):
        """Override to ensure the widget is always shown."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
            return True
        return super().showEditor()
    
    def hideEditor(self):
        """Override to prevent hiding the widget."""
        # Don't actually hide the widget, just ensure it's visible
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
        return True

# Custom parameter type for directory-only browsing
class DirectoryOnlyParameter(pTypes.SimpleParameter):
    itemClass = DirectoryOnlyParameterItem

# Custom widget for directory/file browsing with buttons on same row
class DirectoryBrowseWidget(QWidget):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.sigChanged = None  # No change signal needed for this custom widget
        
        # Set minimum height to ensure widget stays visible
        self.setMinimumHeight(25)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Text field to show current path
        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(param.value()))
        self.lineEdit.textChanged.connect(self._on_text_changed)
        
        # Browse directory button
        browse_dir_btn = QPushButton("Dir...")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setMinimumWidth(50)
        browse_dir_btn.clicked.connect(self._browse_directory)
        
        # Browse file button
        browse_file_btn = QPushButton("File...")
        browse_file_btn.setMaximumWidth(50)
        browse_file_btn.setMinimumWidth(50)
        browse_file_btn.clicked.connect(self._browse_file)
        
        layout.addWidget(self.lineEdit)
        layout.addWidget(browse_dir_btn)
        layout.addWidget(browse_file_btn)
    
    def _on_text_changed(self, text):
        """Handle manual text changes in the line edit."""
        self.param.setValue(text)
    
    def _browse_directory(self):
        """Open directory browser dialog."""
        directory = QFileDialog.getExistingDirectory(None, f"Select directory for {self.param.name()}")
        if directory:
            self.lineEdit.setText(directory)
            self.param.setValue(directory)
    
    def _browse_file(self):
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(None, f"Select file for {self.param.name()}")
        if file_path:
            self.lineEdit.setText(file_path)
            self.param.setValue(file_path)
    
    def value(self):
        """Return current value from the line edit."""
        return self.lineEdit.text()
    
    def setValue(self, value):
        """Set the value in the line edit."""
        self.lineEdit.setText(str(value))
    
    def focusInEvent(self, event):
        """Handle focus in event - ensure widget stays visible."""
        super().focusInEvent(event)
        self.lineEdit.setFocus()
    
    def focusOutEvent(self, event):
        """Handle focus out event - ensure widget stays visible."""
        super().focusOutEvent(event)
        # Don't hide the widget when losing focus
        self.show()

class DirectoryParameterItem(pTypes.WidgetParameterItem):
    def makeWidget(self):
        widget = DirectoryBrowseWidget(self.param)
        # Ensure the widget is always visible
        widget.setVisible(True)
        return widget
    
    def valueChanged(self, param, data, info=None, force=False):
        """Handle external value changes."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.setValue(data)
            # Ensure widget stays visible after value change
            self.widget.show()
    
    def showEditor(self):
        """Override to ensure the widget is always shown."""
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
            return True
        return super().showEditor()
    
    def hideEditor(self):
        """Override to prevent hiding the widget."""
        # Don't actually hide the widget, just ensure it's visible
        if hasattr(self, 'widget') and self.widget is not None:
            self.widget.show()
        return True

# Custom parameter type for directory browsing
class DirectoryParameter(pTypes.SimpleParameter):
    itemClass = DirectoryParameterItem

# Custom augmentation group that includes preset methods and allows adding custom methods from files
class AugmentationGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset augmentation methods
        self._add_preset_augmentations()
        
        # Add custom augmentation button
        self._add_custom_button()
    
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
                'name': 'Gaussian Noise',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable Gaussian noise augmentation'},
                    {'name': 'variance_limit', 'type': 'float', 'value': 0.01, 'limits': (0.0, 0.1), 'tip': 'Maximum variance of Gaussian noise'},
                    {'name': 'probability', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Probability of adding noise'}
                ],
                'tip': 'Add random Gaussian noise to images'
            },
            {
                'name': 'Brightness Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable brightness adjustment'},
                    {'name': 'brightness_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum brightness change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of brightness adjustment'}
                ],
                'tip': 'Randomly adjust image brightness'
            },
            {
                'name': 'Contrast Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable contrast adjustment'},
                    {'name': 'contrast_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum contrast change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of contrast adjustment'}
                ],
                'tip': 'Randomly adjust image contrast'
            },
            {
                'name': 'Color Jittering',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable color jittering'},
                    {'name': 'hue_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 50), 'tip': 'Maximum hue shift'},
                    {'name': 'sat_shift_limit', 'type': 'int', 'value': 30, 'limits': (0, 100), 'tip': 'Maximum saturation shift'},
                    {'name': 'val_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 100), 'tip': 'Maximum value shift'},
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
                    {'name': 'aspect_ratio_min', 'type': 'float', 'value': 0.75, 'limits': (0.1, 2.0), 'tip': 'Minimum aspect ratio for cropping'},
                    {'name': 'aspect_ratio_max', 'type': 'float', 'value': 1.33, 'limits': (0.1, 2.0), 'tip': 'Maximum aspect ratio for cropping'},
                    {'name': 'probability', 'type': 'float', 'value': 1.0, 'limits': (0.0, 1.0), 'tip': 'Probability of random cropping'}
                ],
                'tip': 'Randomly crop parts of the image with specified area and aspect ratio constraints'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom augmentation functions from files."""
        self.addChild({
            'name': 'Load Custom Augmentations',
            'type': 'action',
            'tip': 'Click to load custom augmentation functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Augmentations')
        custom_button.sigActivated.connect(self._load_custom_augmentations)
    
    def _load_custom_augmentations(self):
        """Load custom augmentation functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom augmentation functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_augmentation_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid augmentation functions found in the selected file.\n\n"
                    "Functions should accept 'image' parameter and return modified image."
                )
                return
            
            # Add each found function as a custom augmentation
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom augmentation function(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom augmentations from file:\n{str(e)}"
            )
    
    def _extract_augmentation_functions(self, file_path):
        """Extract valid augmentation functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid augmentation function
                    if self._is_valid_augmentation_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom augmentation function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
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
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'image' parameter)."""
        params = []
        
        # Skip the first parameter (image) and extract others
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 0.5,   # Default value
                'limits': (0.0, 1.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'angle' in param_name.lower():
                param_info.update({'type': 'float', 'default': 15.0, 'limits': (0.0, 180.0), 'suffix': '°'})
            elif 'prob' in param_name.lower() or 'p' == param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'strength' in param_name.lower() or 'intensity' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 5.0)})
            elif 'size' in param_name.lower() or 'kernel' in param_name.lower():
                param_info.update({'type': 'int', 'default': 3, 'limits': (1, 15)})
            elif 'enable' in param_name.lower():
                param_info.update({'type': 'bool', 'default': True})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 1  # -1 to skip image param
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as an augmentation method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists (check both original and display names)
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} augmentation'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            children.append({
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'limits': param_info.get('limits'),
                'suffix': param_info.get('suffix', ''),
                'tip': param_info['tip']
            })
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
        ])
        
        # Create the augmentation method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,  # Keep original function name
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Augmentations" button
        # Find the button's index and insert before it
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Augmentations':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            # Fallback: add at the end if button not found
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        # This method is called by the parameter tree system but we use the button instead
        pass

# Custom preprocessing group that includes preset methods and allows adding custom methods from files
class PreprocessingGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset preprocessing methods
        self._add_preset_preprocessing()
        
        # Add custom preprocessing button
        self._add_custom_button()
    
    def _add_preset_preprocessing(self):
        """Add preset preprocessing methods with their parameters."""
        preset_methods = [
            {
                'name': 'Resizing',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable image resizing'},
                    {'name': 'target_size', 'type': 'group', 'children': [
                        {'name': 'width', 'type': 'int', 'value': 224, 'limits': (1, 2048), 'tip': 'Target width in pixels'},
                        {'name': 'height', 'type': 'int', 'value': 224, 'limits': (1, 2048), 'tip': 'Target height in pixels'},
                        {'name': 'depth', 'type': 'int', 'value': 1, 'limits': (1, 512), 'tip': 'Target depth for 3D data (1 for 2D)'}
                    ], 'tip': 'Target dimensions for resizing'},
                    {'name': 'interpolation', 'type': 'list', 'limits': ['bilinear', 'nearest', 'bicubic', 'area'], 'value': 'bilinear', 'tip': 'Interpolation method for resizing'},
                    {'name': 'preserve_aspect_ratio', 'type': 'bool', 'value': True, 'tip': 'Whether to preserve aspect ratio during resize'},
                    {'name': 'data_format', 'type': 'list', 'limits': ['2D', '3D'], 'value': '2D', 'tip': 'Data format (2D for images, 3D for volumes)'}
                ],
                'tip': 'Resize images to target dimensions with support for 2D and 3D data'
            },
            {
                'name': 'Normalization',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable data normalization'},
                    {'name': 'method', 'type': 'list', 'limits': ['min-max', 'zero-center', 'standardization', 'unit-norm', 'robust'], 'value': 'zero-center', 'tip': 'Normalization method'},
                    {'name': 'min_value', 'type': 'float', 'value': 0.0, 'limits': (-10.0, 10.0), 'tip': 'Minimum value for min-max normalization'},
                    {'name': 'max_value', 'type': 'float', 'value': 1.0, 'limits': (-10.0, 10.0), 'tip': 'Maximum value for min-max normalization'},
                    {'name': 'mean', 'type': 'group', 'children': [
                        {'name': 'r', 'type': 'float', 'value': 0.485, 'limits': (0.0, 1.0), 'tip': 'Mean value for red channel'},
                        {'name': 'g', 'type': 'float', 'value': 0.456, 'limits': (0.0, 1.0), 'tip': 'Mean value for green channel'},
                        {'name': 'b', 'type': 'float', 'value': 0.406, 'limits': (0.0, 1.0), 'tip': 'Mean value for blue channel'}
                    ], 'tip': 'Mean values for zero-center normalization (ImageNet defaults)'},
                    {'name': 'std', 'type': 'group', 'children': [
                        {'name': 'r', 'type': 'float', 'value': 0.229, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for red channel'},
                        {'name': 'g', 'type': 'float', 'value': 0.224, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for green channel'},
                        {'name': 'b', 'type': 'float', 'value': 0.225, 'limits': (0.001, 1.0), 'tip': 'Standard deviation for blue channel'}
                    ], 'tip': 'Standard deviation values for standardization (ImageNet defaults)'},
                    {'name': 'axis', 'type': 'int', 'value': -1, 'limits': (-3, 3), 'tip': 'Axis along which to normalize (-1 for all)'},
                    {'name': 'epsilon', 'type': 'float', 'value': 1e-7, 'limits': (1e-10, 1e-3), 'tip': 'Small constant to avoid division by zero'}
                ],
                'tip': 'Normalize data using various methods (min-max, zero-center, standardization, etc.)'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom preprocessing functions from files."""
        self.addChild({
            'name': 'Load Custom Preprocessing',
            'type': 'action',
            'tip': 'Click to load custom preprocessing functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Preprocessing')
        custom_button.sigActivated.connect(self._load_custom_preprocessing)
    
    def _load_custom_preprocessing(self):
        """Load custom preprocessing functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom preprocessing functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_preprocessing_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid preprocessing functions found in the selected file.\n\n"
                    "Functions should accept 'data' parameter and return processed data."
                )
                return
            
            # Add each found function as a custom preprocessing method
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom preprocessing function(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom preprocessing from file:\n{str(e)}"
            )
    
    def _extract_preprocessing_functions(self, file_path):
        """Extract valid preprocessing functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid preprocessing function
                    if self._is_valid_preprocessing_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom preprocessing function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_preprocessing_function(self, func_node):
        """Check if a function is a valid preprocessing function."""
        # Check if function has at least one parameter (should be 'data' or similar)
        if not func_node.args.args:
            return False
        
        # Check if first parameter is likely a data parameter
        first_param = func_node.args.args[0].arg
        if first_param not in ['data', 'x', 'input', 'array', 'tensor']:
            return False
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        if not has_return:
            return False
        
        return True
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'data' parameter)."""
        params = []
        
        # Skip the first parameter (data) and extract others
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 1.0,   # Default value
                'limits': (0.0, 10.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'size' in param_name.lower() or 'dim' in param_name.lower():
                param_info.update({'type': 'int', 'default': 224, 'limits': (1, 2048)})
            elif 'scale' in param_name.lower() or 'factor' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
            elif 'mean' in param_name.lower() or 'center' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'std' in param_name.lower() or 'deviation' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.25, 'limits': (0.001, 1.0)})
            elif 'enable' in param_name.lower():
                param_info.update({'type': 'bool', 'default': True})
            elif 'method' in param_name.lower() or 'mode' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'bilinear'})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 1  # -1 to skip data param
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
                        elif isinstance(default.value, str):
                            params[param_index]['type'] = 'str'
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as a preprocessing method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists (check both original and display names)
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} preprocessing'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            # Add limits for numeric types
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            children.append(param_config)
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
        ])
        
        # Create the preprocessing method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Preprocessing" button
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Preprocessing':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

# Custom callbacks group that includes preset callbacks and allows adding custom callbacks from files  
class CallbacksGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset callbacks
        self._add_preset_callbacks()
        
        # Add custom callbacks button
        self._add_custom_button()
    
    def _add_preset_callbacks(self):
        """Add preset callback methods with their parameters."""
        preset_callbacks = [
            {
                'name': 'Early Stopping',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable early stopping'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'patience', 'type': 'int', 'value': 10, 'limits': (1, 100), 'tip': 'Number of epochs with no improvement to wait'},
                    {'name': 'min_delta', 'type': 'float', 'value': 0.001, 'limits': (0.0, 1.0), 'tip': 'Minimum change to qualify as improvement'},
                    {'name': 'mode', 'type': 'list', 'limits': ['min', 'max', 'auto'], 'value': 'min', 'tip': 'Direction of improvement'},
                    {'name': 'restore_best_weights', 'type': 'bool', 'value': True, 'tip': 'Restore model weights from best epoch'}
                ],
                'tip': 'Stop training when monitored metric stops improving'
            },
            {
                'name': 'Learning Rate Scheduler',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable learning rate scheduling'},
                    {'name': 'scheduler_type', 'type': 'list', 'limits': ['ReduceLROnPlateau', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR'], 'value': 'ReduceLROnPlateau', 'tip': 'Type of learning rate scheduler'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'factor', 'type': 'float', 'value': 0.5, 'limits': (0.01, 1.0), 'tip': 'Factor by which learning rate is reduced'},
                    {'name': 'patience', 'type': 'int', 'value': 5, 'limits': (1, 50), 'tip': 'Number of epochs with no improvement to wait'},
                    {'name': 'min_lr', 'type': 'float', 'value': 1e-7, 'limits': (1e-10, 1e-2), 'tip': 'Minimum learning rate'},
                    {'name': 'step_size', 'type': 'int', 'value': 30, 'limits': (1, 1000), 'tip': 'Period of learning rate decay (for StepLR)'},
                    {'name': 'gamma', 'type': 'float', 'value': 0.1, 'limits': (0.01, 1.0), 'tip': 'Multiplicative factor of learning rate decay'}
                ],
                'tip': 'Adjust learning rate during training based on metrics or schedule'
            },
            {
                'name': 'Model Checkpoint',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable model checkpointing'},
                    {'name': 'filepath', 'type': 'str', 'value': './checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5', 'tip': 'Path template for checkpoint files'},
                    {'name': 'monitor', 'type': 'list', 'limits': ['val_loss', 'val_accuracy', 'loss', 'accuracy'], 'value': 'val_loss', 'tip': 'Metric to monitor'},
                    {'name': 'save_best_only', 'type': 'bool', 'value': True, 'tip': 'Save only the best model'},
                    {'name': 'save_weights_only', 'type': 'bool', 'value': False, 'tip': 'Save only model weights (not full model)'},
                    {'name': 'mode', 'type': 'list', 'limits': ['min', 'max', 'auto'], 'value': 'min', 'tip': 'Direction of improvement'},
                    {'name': 'period', 'type': 'int', 'value': 1, 'limits': (1, 100), 'tip': 'Interval between checkpoints'}
                ],
                'tip': 'Save model checkpoints during training'
            },
            {
                'name': 'CSV Logger',
                'type': 'group', 
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable CSV logging'},
                    {'name': 'filename', 'type': 'str', 'value': './logs/training_log.csv', 'tip': 'Path to CSV log file'},
                    {'name': 'separator', 'type': 'str', 'value': ',', 'tip': 'Delimiter for CSV file'},
                    {'name': 'append', 'type': 'bool', 'value': False, 'tip': 'Append to existing file or create new'}
                ],
                'tip': 'Log training metrics to CSV file'
            },
            {
                'name': 'TensorBoard',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable TensorBoard logging'},
                    {'name': 'log_dir', 'type': 'str', 'value': './logs/tensorboard', 'tip': 'Directory for TensorBoard logs'},
                    {'name': 'histogram_freq', 'type': 'int', 'value': 1, 'limits': (0, 100), 'tip': 'Frequency for histogram computation'},
                    {'name': 'write_graph', 'type': 'bool', 'value': True, 'tip': 'Write model graph to TensorBoard'},
                    {'name': 'write_images', 'type': 'bool', 'value': False, 'tip': 'Write model weights as images'},
                    {'name': 'update_freq', 'type': 'list', 'limits': ['epoch', 'batch'], 'value': 'epoch', 'tip': 'Update frequency for logging'}
                ],
                'tip': 'Log training metrics and model graph to TensorBoard'
            }
        ]
        
        # Add all preset callbacks
        for callback in preset_callbacks:
            self.addChild(callback)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom callback functions from files."""
        self.addChild({
            'name': 'Load Custom Callbacks',
            'type': 'action',
            'tip': 'Click to load custom callback functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Callbacks')
        custom_button.sigActivated.connect(self._load_custom_callbacks)
    
    def _load_custom_callbacks(self):
        """Load custom callback functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom callback functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_callback_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid callback functions found in the selected file.\n\n"
                    "Functions should inherit from tf.keras.callbacks.Callback or implement callback interface."
                )
                return
            
            # Add each found function as a custom callback
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom callback function(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom callbacks from file:\n{str(e)}"
            )
    
    def _extract_callback_functions(self, file_path):
        """Extract valid callback functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid callback function
                    if self._is_valid_callback_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom callback function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a callback class
                    if self._is_valid_callback_class(node):
                        # Extract class init parameters
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom callback class: {class_name}"
                        
                        custom_functions[class_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': class_name,
                            'type': 'class'
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_callback_function(self, func_node):
        """Check if a function is a valid callback function."""
        # Look for common callback method names or parameters
        func_name = func_node.name.lower()
        callback_indicators = ['callback', 'on_epoch', 'on_batch', 'on_train', 'monitor', 'log']
        
        return any(indicator in func_name for indicator in callback_indicators)
    
    def _is_valid_callback_class(self, class_node):
        """Check if a class is a valid callback class."""
        class_name = class_node.name.lower()
        
        # Check class name for callback indicators
        if 'callback' in class_name:
            return True
        
        # Check if class has callback-like methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_name = node.name.lower()
                if any(method in method_name for method in ['on_epoch', 'on_batch', 'on_train']):
                    return True
        
        return False
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
                    param_name = arg.arg
                    
                    param_info = {
                        'name': param_name,
                        'type': 'str',  # Default type
                        'default': '',
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference
                    if 'patience' in param_name.lower() or 'epoch' in param_name.lower():
                        param_info.update({'type': 'int', 'default': 10, 'limits': (1, 1000)})
                    elif 'rate' in param_name.lower() or 'factor' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 0.1, 'limits': (0.001, 1.0)})
                    elif 'enable' in param_name.lower():
                        param_info.update({'type': 'bool', 'default': True})
                    elif 'path' in param_name.lower() or 'dir' in param_name.lower():
                        param_info.update({'type': 'str', 'default': './logs'})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition."""
        params = []
        
        # Extract function arguments (skip common callback parameters like 'logs', 'epoch', etc.)
        skip_params = {'self', 'logs', 'epoch', 'batch', 'model'}
        
        for arg in func_node.args.args:
            param_name = arg.arg
            
            if param_name not in skip_params:
                param_info = {
                    'name': param_name,
                    'type': 'str',
                    'default': '',
                    'tip': f'Parameter for {param_name}'
                }
                
                # Basic type inference based on parameter name
                if 'patience' in param_name.lower() or 'step' in param_name.lower():
                    param_info.update({'type': 'int', 'default': 10, 'limits': (1, 1000)})
                elif 'rate' in param_name.lower() or 'threshold' in param_name.lower():
                    param_info.update({'type': 'float', 'default': 0.1, 'limits': (0.001, 1.0)})
                elif 'enable' in param_name.lower():
                    param_info.update({'type': 'bool', 'default': True})
                
                params.append(param_info)
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as a callback method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} callback'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            children.append(param_config)
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'callback_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of callback (function or class)'}
        ])
        
        # Create the callback method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Callbacks" button
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Callbacks':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

# Custom loss functions group that includes preset loss functions and allows adding custom loss functions from files
class LossFunctionsGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add model output configuration first
        self._add_output_configuration()
        
        # Add loss function selection
        self._add_loss_selection()
        
        # Add custom loss function button
        self._add_custom_button()
    
    def _add_output_configuration(self):
        """Add model output configuration."""
        self.addChild({
            'name': 'Model Output Configuration',
            'type': 'group',
            'children': [
                {'name': 'num_outputs', 'type': 'int', 'value': 1, 'limits': (1, 10), 'tip': 'Number of model outputs (1 for single output, >1 for multiple outputs)'},
                {'name': 'output_names', 'type': 'str', 'value': 'main_output', 'tip': 'Comma-separated names for multiple outputs (e.g., "main_output,aux_output")'},
                {'name': 'loss_strategy', 'type': 'list', 'limits': ['single_loss_all_outputs', 'different_loss_per_output'], 'value': 'single_loss_all_outputs', 'tip': 'Loss strategy: same loss for all outputs or different loss per output'}
            ],
            'tip': 'Configure model outputs and loss assignment strategy'
        })
        
        # Connect output configuration change to update loss selection
        output_config = self.child('Model Output Configuration')
        output_config.child('num_outputs').sigValueChanged.connect(self._update_loss_selection)
        output_config.child('loss_strategy').sigValueChanged.connect(self._update_loss_selection)
    
    def _add_loss_selection(self):
        """Add loss function selection based on output configuration."""
        # Initially add single loss selection
        self._update_loss_selection()
    
    def _update_loss_selection(self):
        """Update loss function selection based on output configuration."""
        # Remove existing loss selection if any
        existing_groups = []
        for child in self.children():
            if child.name().startswith('Loss Selection') or child.name().startswith('Output'):
                existing_groups.append(child)
        
        for group in existing_groups:
            self.removeChild(group)
        
        # Get current configuration
        output_config = self.child('Model Output Configuration')
        num_outputs = output_config.child('num_outputs').value()
        loss_strategy = output_config.child('loss_strategy').value()
        output_names = output_config.child('output_names').value().split(',')
        output_names = [name.strip() for name in output_names if name.strip()]
        
        if num_outputs == 1 or loss_strategy == 'single_loss_all_outputs':
            # Single loss function for all outputs
            self._add_single_loss_selection()
        else:
            # Different loss function per output
            self._add_multiple_loss_selection(num_outputs, output_names)
    
    def _add_single_loss_selection(self):
        """Add single loss function selection."""
        loss_options = self._get_loss_function_options()
        
        self.addChild({
            'name': 'Loss Selection',
            'type': 'group',
            'children': [
                {'name': 'selected_loss', 'type': 'list', 'limits': loss_options, 'value': 'Categorical Crossentropy', 'tip': 'Select the loss function to use'},
                {'name': 'loss_weight', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': 'Weight for this loss function'}
            ],
            'tip': 'Single loss function applied to all model outputs'
        })
        
        # Add loss function parameters
        self._add_selected_loss_parameters('Loss Selection')
    
    def _add_multiple_loss_selection(self, num_outputs, output_names):
        """Add multiple loss function selections for different outputs."""
        loss_options = self._get_loss_function_options()
        
        for i in range(num_outputs):
            output_name = output_names[i] if i < len(output_names) else f'output_{i+1}'
            
            self.addChild({
                'name': f'Output {i+1}: {output_name}',
                'type': 'group',
                'children': [
                    {'name': 'selected_loss', 'type': 'list', 'limits': loss_options, 'value': 'Categorical Crossentropy', 'tip': f'Select loss function for {output_name}'},
                    {'name': 'loss_weight', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': f'Weight for {output_name} loss function'}
                ],
                'tip': f'Loss function configuration for output: {output_name}'
            })
            
            # Add loss function parameters for this output
            self._add_selected_loss_parameters(f'Output {i+1}: {output_name}')
    
    def _get_loss_function_options(self):
        """Get list of available loss function names including custom ones."""
        base_options = [
            'Categorical Crossentropy',
            'Sparse Categorical Crossentropy', 
            'Binary Crossentropy',
            'Mean Squared Error',
            'Mean Absolute Error',
            'Focal Loss',
            'Huber Loss'
        ]
        
        # Add custom loss functions if any
        if hasattr(self, '_custom_loss_functions'):
            custom_options = list(self._custom_loss_functions.keys())
            return base_options + custom_options
        
        return base_options
    
    def _add_selected_loss_parameters(self, parent_name):
        """Add parameters for the selected loss function."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        # Connect selection change to parameter update
        if parent.child('selected_loss'):
            parent.child('selected_loss').sigValueChanged.connect(
                lambda: self._update_loss_parameters(parent_name)
            )
        
        # Add initial parameters
        self._update_loss_parameters(parent_name)
    
    def _update_loss_parameters(self, parent_name):
        """Update loss function parameters based on selection."""
        parent = self.child(parent_name)
        if not parent:
            return
            
        selected_loss = parent.child('selected_loss').value()
        
        # Remove existing parameters (except selected_loss and loss_weight)
        existing_params = []
        for child in parent.children():
            if child.name() not in ['selected_loss', 'loss_weight']:
                existing_params.append(child)
        
        for param in existing_params:
            parent.removeChild(param)
        
        # Add parameters based on selected loss function
        loss_params = self._get_loss_function_parameters(selected_loss)
        for param_config in loss_params:
            parent.addChild(param_config)
    
    def _get_loss_function_parameters(self, loss_name):
        """Get parameters for a specific loss function."""
        # Check if it's a custom loss function
        if hasattr(self, '_custom_loss_parameters') and loss_name in self._custom_loss_parameters:
            return self._custom_loss_parameters[loss_name]
        
        # Return built-in loss function parameters
        loss_parameters = {
            'Categorical Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'label_smoothing', 'type': 'float', 'value': 0.0, 'limits': (0.0, 0.5), 'tip': 'Label smoothing factor'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Sparse Categorical Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Binary Crossentropy': [
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'label_smoothing', 'type': 'float', 'value': 0.0, 'limits': (0.0, 0.5), 'tip': 'Label smoothing factor'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Mean Squared Error': [
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Mean Absolute Error': [
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Focal Loss': [
                {'name': 'alpha', 'type': 'float', 'value': 0.25, 'limits': (0.0, 1.0), 'tip': 'Weighting factor for rare class'},
                {'name': 'gamma', 'type': 'float', 'value': 2.0, 'limits': (0.0, 5.0), 'tip': 'Focusing parameter'},
                {'name': 'from_logits', 'type': 'bool', 'value': False, 'tip': 'Whether predictions are logits or probabilities'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ],
            'Huber Loss': [
                {'name': 'delta', 'type': 'float', 'value': 1.0, 'limits': (0.1, 10.0), 'tip': 'Threshold at which to change between MSE and MAE'},
                {'name': 'reduction', 'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'value': 'sum_over_batch_size', 'tip': 'Type of reduction to apply'}
            ]
        }
        
        return loss_parameters.get(loss_name, [])
    
    def _add_preset_loss_functions(self):
        """Add preset loss functions with their parameters - DEPRECATED."""
        # This method is now deprecated as we use selection-based approach
        pass
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom loss functions from files."""
        self.addChild({
            'name': 'Load Custom Loss Functions',
            'type': 'action',
            'tip': 'Click to load custom loss functions from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Loss Functions')
        custom_button.sigActivated.connect(self._load_custom_loss_functions)
    
    def _load_custom_loss_functions(self):
        """Load custom loss functions from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom loss functions",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_loss_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid loss functions found in the selected file.\n\n"
                    "Functions should accept 'y_true' and 'y_pred' parameters and return loss value."
                )
                return
            
            # Add custom functions to the available loss options
            for func_name, func_info in custom_functions.items():
                self._add_custom_loss_option(func_name, func_info)
            
            # Update all loss selection dropdowns
            self._update_all_loss_selections()
            
            QMessageBox.information(
                None,
                "Functions Loaded",
                f"Successfully loaded {len(custom_functions)} custom loss function(s):\n" +
                "\n".join(custom_functions.keys()) +
                "\n\nThese functions are now available in the loss selection dropdowns."
            )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom loss functions from file:\n{str(e)}"
            )
    
    def _add_custom_loss_option(self, func_name, func_info):
        """Add a custom loss function as an option in dropdowns."""
        # Store custom loss function info for later use
        if not hasattr(self, '_custom_loss_functions'):
            self._custom_loss_functions = {}
        
        display_name = f"{func_name} (custom)"
        self._custom_loss_functions[display_name] = func_info
        
        # Add parameters for this custom loss function
        params = []
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            # Add limits for numeric types
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            elif param_info['type'] == 'list' and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            params.append(param_config)
        
        # Add metadata parameters
        params.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'loss_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of loss (function or class)'}
        ])
        
        # Store parameters for this custom loss function
        if not hasattr(self, '_custom_loss_parameters'):
            self._custom_loss_parameters = {}
        self._custom_loss_parameters[display_name] = params
    
    def _update_all_loss_selections(self):
        """Update all loss selection dropdowns with custom functions."""
        # Get updated loss function options
        loss_options = self._get_loss_function_options()
        
        # Find all loss selection parameters and update their options
        for child in self.children():
            if child.name().startswith('Loss Selection') or child.name().startswith('Output'):
                selected_loss_param = child.child('selected_loss')
                if selected_loss_param:
                    # Update the limits (available options)
                    selected_loss_param.setLimits(loss_options)

    def _extract_loss_functions(self, file_path):
        """Extract valid loss functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid loss function
                    if self._is_valid_loss_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom loss function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a valid loss class
                    if self._is_valid_loss_class(node):
                        # Extract class parameters from __init__ method
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom loss class: {class_name}"
                        
                        custom_functions[class_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': class_name,
                            'type': 'class'
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_loss_function(self, func_node):
        """Check if a function is a valid loss function."""
        # Check if function has at least two parameters (should be 'y_true', 'y_pred')
        if len(func_node.args.args) < 2:
            return False
        
        # Check if parameters are likely loss function parameters
        param_names = [arg.arg for arg in func_node.args.args]
        
        # Common loss function parameter names
        valid_patterns = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits'],
            ['ground_truth', 'predictions']
        ]
        
        for pattern in valid_patterns:
            if all(any(p in param.lower() for p in pattern) for param in param_names[:2]):
                return True
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return
    
    def _is_valid_loss_class(self, class_node):
        """Check if a class is a valid loss class."""
        class_name = class_node.name.lower()
        
        # Check class name for loss indicators
        if 'loss' in class_name:
            return True
        
        # Check if class has call method (indicating it's callable)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__call__':
                return True
        
        return False
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
                    param_name = arg.arg
                    
                    # Try to infer parameter type and default values
                    param_info = {
                        'name': param_name,
                        'type': 'float',  # Default type
                        'default': 1.0,   # Default value
                        'limits': (0.0, 10.0),
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference based on parameter name
                    if 'alpha' in param_name.lower() or 'weight' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 10.0)})
                    elif 'gamma' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 2.0, 'limits': (0.0, 5.0)})
                    elif 'delta' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
                    elif 'reduction' in param_name.lower():
                        param_info.update({'type': 'list', 'limits': ['sum_over_batch_size', 'sum', 'none'], 'default': 'sum_over_batch_size'})
                    elif 'from_logits' in param_name.lower():
                        param_info.update({'type': 'bool', 'default': False})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'y_true', 'y_pred' parameters)."""
        params = []
        
        # Skip the first two parameters (y_true, y_pred) and extract others
        for arg in func_node.args.args[2:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 1.0,   # Default value
                'limits': (0.0, 10.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'alpha' in param_name.lower() or 'weight' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 10.0)})
            elif 'gamma' in param_name.lower():
                param_info.update({'type': 'float', 'default': 2.0, 'limits': (0.0, 5.0)})
            elif 'delta' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.1, 10.0)})
            elif 'reduction' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'sum_over_batch_size'})
            elif 'from_logits' in param_name.lower():
                param_info.update({'type': 'bool', 'default': False})
            elif 'smooth' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.0, 'limits': (0.0, 0.5)})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 2  # -2 to skip y_true, y_pred
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

# Custom metrics group that includes preset metrics and allows adding custom metrics from files  
class MetricsGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
        
        # Add preset metrics
        self._add_preset_metrics()
        
        # Add custom metrics button
        self._add_custom_button()
    
    def _add_preset_metrics(self):
        """Add preset metrics with their parameters."""
        preset_metrics = [
            {
                'name': 'Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Standard accuracy metric for classification tasks'
            },
            {
                'name': 'Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'categorical_accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Categorical accuracy metric for multi-class classification'
            },
            {
                'name': 'Sparse Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable sparse categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'sparse_categorical_accuracy', 'tip': 'Name for this metric'}
                ],
                'tip': 'Sparse categorical accuracy for integer label classification'
            },
            {
                'name': 'Top-K Categorical Accuracy',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable top-k categorical accuracy metric'},
                    {'name': 'name', 'type': 'str', 'value': 'top_5_accuracy', 'tip': 'Name for this metric'},
                    {'name': 'k', 'type': 'int', 'value': 5, 'limits': (1, 100), 'tip': 'Number of top predictions to consider'}
                ],
                'tip': 'Top-K accuracy metric (e.g., top-5 accuracy)'
            },
            {
                'name': 'Precision',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable precision metric'},
                    {'name': 'name', 'type': 'str', 'value': 'precision', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary precision (0 for first class, None for multiclass)'}
                ],
                'tip': 'Precision metric for classification tasks'
            },
            {
                'name': 'Recall',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable recall metric'},
                    {'name': 'name', 'type': 'str', 'value': 'recall', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary recall (0 for first class, None for multiclass)'}
                ],
                'tip': 'Recall metric for classification tasks'
            },
            {
                'name': 'F1 Score',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable F1 score metric'},
                    {'name': 'name', 'type': 'str', 'value': 'f1_score', 'tip': 'Name for this metric'},
                    {'name': 'average', 'type': 'list', 'limits': ['micro', 'macro', 'weighted', 'samples'], 'value': 'macro', 'tip': 'Averaging strategy'},
                    {'name': 'class_id', 'type': 'int', 'value': 0, 'limits': (0, 100), 'tip': 'Class ID for binary F1 (0 for first class, None for multiclass)'}
                ],
                'tip': 'F1 score metric (harmonic mean of precision and recall)'
            },
            {
                'name': 'AUC',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable AUC metric'},
                    {'name': 'name', 'type': 'str', 'value': 'auc', 'tip': 'Name for this metric'},
                    {'name': 'curve', 'type': 'list', 'limits': ['ROC', 'PR'], 'value': 'ROC', 'tip': 'Curve type (ROC or Precision-Recall)'},
                    {'name': 'multi_class', 'type': 'list', 'limits': ['ovr', 'ovo'], 'value': 'ovr', 'tip': 'Multiclass strategy (one-vs-rest or one-vs-one)'}
                ],
                'tip': 'Area Under the Curve (AUC) metric'
            },
            {
                'name': 'Mean Squared Error',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable mean squared error metric'},
                    {'name': 'name', 'type': 'str', 'value': 'mse', 'tip': 'Name for this metric'}
                ],
                'tip': 'Mean squared error metric for regression tasks'
            },
            {
                'name': 'Mean Absolute Error',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable mean absolute error metric'},
                    {'name': 'name', 'type': 'str', 'value': 'mae', 'tip': 'Name for this metric'}
                ],
                'tip': 'Mean absolute error metric for regression tasks'
            }
        ]
        
        # Add all preset metrics
        for metric in preset_metrics:
            self.addChild(metric)
    
    def _add_custom_button(self):
        """Add a button parameter for loading custom metrics from files."""
        self.addChild({
            'name': 'Load Custom Metrics',
            'type': 'action',
            'tip': 'Click to load custom metrics from a Python file'
        })
        
        # Connect the action to the file loading function
        custom_button = self.child('Load Custom Metrics')
        custom_button.sigActivated.connect(self._load_custom_metrics)
    
    def _load_custom_metrics(self):
        """Load custom metrics from a selected Python file."""
        
        # Open file dialog to select Python file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Python file with custom metrics",
            "",
            "Python Files (*.py)"
        )
        
        if not file_path:
            return
        
        try:
            # Load and parse the Python file
            custom_functions = self._extract_metric_functions(file_path)
            
            if not custom_functions:
                QMessageBox.warning(
                    None,
                    "No Functions Found",
                    "No valid metric functions found in the selected file.\n\n"
                    "Functions should accept 'y_true' and 'y_pred' parameters and return metric value."
                )
                return
            
            # Add each found function as a custom metric
            added_count = 0
            for func_name, func_info in custom_functions.items():
                if self._add_custom_function(func_name, func_info):
                    added_count += 1
            
            if added_count > 0:
                QMessageBox.information(
                    None,
                    "Functions Loaded",
                    f"Successfully loaded {added_count} custom metric(s):\n" +
                    "\n".join(custom_functions.keys())
                )
            else:
                QMessageBox.warning(
                    None,
                    "No New Functions",
                    "All functions from the file are already loaded or invalid."
                )
                
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load custom metrics from file:\n{str(e)}"
            )
    
    def _extract_metric_functions(self, file_path):
        """Extract valid metric functions from a Python file."""
        custom_functions = {}
        
        try:
            # Read and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find function definitions and class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    
                    # Check if it's a valid metric function
                    if self._is_valid_metric_function(node):
                        # Extract function parameters
                        params = self._extract_function_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom metric function: {func_name}"
                        
                        custom_functions[func_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': func_name,
                            'type': 'function'
                        }
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Check if it's a valid metric class
                    if self._is_valid_metric_class(node):
                        # Extract class parameters from __init__ method
                        params = self._extract_class_parameters(node)
                        
                        # Extract docstring if available
                        docstring = ast.get_docstring(node) or f"Custom metric class: {class_name}"
                        
                        custom_functions[class_name] = {
                            'parameters': params,
                            'docstring': docstring,
                            'file_path': file_path,
                            'function_name': class_name,
                            'type': 'class'
                        }
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        
        return custom_functions
    
    def _is_valid_metric_function(self, func_node):
        """Check if a function is a valid metric function."""
        # Check if function has at least two parameters (should be 'y_true', 'y_pred')
        if len(func_node.args.args) < 2:
            return False
        
        # Check if parameters are likely metric function parameters
        param_names = [arg.arg for arg in func_node.args.args]
        
        # Common metric function parameter names
        valid_patterns = [
            ['y_true', 'y_pred'],
            ['true', 'pred'],
            ['target', 'prediction'],
            ['labels', 'logits'],
            ['ground_truth', 'predictions']
        ]
        
        for pattern in valid_patterns:
            if all(any(p in param.lower() for p in pattern) for param in param_names[:2]):
                return True
        
        # Function should return something (basic check)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return
    
    def _is_valid_metric_class(self, class_node):
        """Check if a class is a valid metric class."""
        class_name = class_node.name.lower()
        
        # Check class name for metric indicators
        metric_indicators = ['metric', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'score']
        if any(indicator in class_name for indicator in metric_indicators):
            return True
        
        # Check if class has call method or update_state method (TensorFlow metric pattern)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name in ['__call__', 'update_state', 'result']:
                return True
        
        return False
    
    def _extract_class_parameters(self, class_node):
        """Extract parameters from class __init__ method."""
        params = []
        
        # Find __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                # Skip 'self' parameter and extract others
                for arg in node.args.args[1:]:
                    param_name = arg.arg
                    
                    # Try to infer parameter type and default values
                    param_info = {
                        'name': param_name,
                        'type': 'float',  # Default type
                        'default': 1.0,   # Default value
                        'limits': (0.0, 10.0),
                        'tip': f'Parameter for {param_name}'
                    }
                    
                    # Basic type inference based on parameter name
                    if 'name' in param_name.lower():
                        param_info.update({'type': 'str', 'default': 'custom_metric'})
                    elif 'k' in param_name.lower() and len(param_name) <= 2:
                        param_info.update({'type': 'int', 'default': 5, 'limits': (1, 100)})
                    elif 'threshold' in param_name.lower():
                        param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
                    elif 'average' in param_name.lower():
                        param_info.update({'type': 'list', 'limits': ['micro', 'macro', 'weighted'], 'default': 'macro'})
                    elif 'class_id' in param_name.lower():
                        param_info.update({'type': 'int', 'default': None})
                    
                    params.append(param_info)
                break
        
        return params
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'y_true', 'y_pred' parameters)."""
        params = []
        
        # Skip the first two parameters (y_true, y_pred) and extract others
        for arg in func_node.args.args[2:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 1.0,   # Default value
                'limits': (0.0, 10.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'name' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'custom_metric'})
            elif 'k' in param_name.lower() and len(param_name) <= 2:
                param_info.update({'type': 'int', 'default': 5, 'limits': (1, 100)})
            elif 'threshold' in param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'average' in param_name.lower():
                param_info.update({'type': 'str', 'default': 'macro'})
            elif 'class_id' in param_name.lower():
                param_info.update({'type': 'int', 'default': None})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 2  # -2 to skip y_true, y_pred
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    def _add_custom_function(self, func_name, func_info):
        """Add a custom function as a metric method."""
        # Add (custom) suffix to distinguish from presets
        display_name = f"{func_name} (custom)"
        
        # Check if function already exists
        existing_names = [child.name() for child in self.children()]
        if func_name in existing_names or display_name in existing_names:
            return False
        
        # Create parameters list
        children = [
            {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} metric'}
        ]
        
        # Add function-specific parameters
        for param_info in func_info['parameters']:
            param_config = {
                'name': param_info['name'],
                'type': param_info['type'],
                'value': param_info['default'],
                'tip': param_info['tip']
            }
            
            # Add limits for numeric types
            if param_info['type'] in ['int', 'float'] and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            elif param_info['type'] == 'list' and 'limits' in param_info:
                param_config['limits'] = param_info['limits']
            
            children.append(param_config)
        
        # Add metadata parameters
        children.extend([
            {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
            {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function/class name in source file'},
            {'name': 'metric_type', 'type': 'str', 'value': func_info['type'], 'readonly': True, 'tip': 'Type of metric (function or class)'}
        ])
        
        # Create the metric method
        method_config = {
            'name': display_name,
            'type': 'group',
            'children': children,
            'removable': True,
            'renamable': False,
            'tip': func_info['docstring']
        }
        
        # Insert before the "Load Custom Metrics" button
        button_index = None
        for i, child in enumerate(self.children()):
            if child.name() == 'Load Custom Metrics':
                button_index = i
                break
        
        if button_index is not None:
            self.insertChild(button_index, method_config)
        else:
            self.addChild(method_config)
        
        return True
    
    def addNew(self, typ=None):
        """Legacy method - no longer used since we load from files."""
        pass

# Register the custom parameter types
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
                'name': 'Gaussian Noise',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable Gaussian noise augmentation'},
                    {'name': 'variance_limit', 'type': 'float', 'value': 0.01, 'limits': (0.0, 0.1), 'tip': 'Maximum variance of Gaussian noise'},
                    {'name': 'probability', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Probability of adding noise'}
                ],
                'tip': 'Add random Gaussian noise to images'
            },
            {
                'name': 'Brightness Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable brightness adjustment'},
                    {'name': 'brightness_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum brightness change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of brightness adjustment'}
                ],
                'tip': 'Randomly adjust image brightness'
            },
            {
                'name': 'Contrast Adjustment',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': False, 'tip': 'Enable contrast adjustment'},
                    {'name': 'contrast_limit', 'type': 'float', 'value': 0.2, 'limits': (0.0, 1.0), 'tip': 'Maximum contrast change (±)'},
                    {'name': 'probability', 'type': 'float', 'value': 0.5, 'limits': (0.0, 1.0), 'tip': 'Probability of contrast adjustment'}
                ],
                'tip': 'Randomly adjust image contrast'
            },
            {
                'name': 'Color Jittering',
                'type': 'group',
                'children': [
                    {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': 'Enable color jittering'},
                    {'name': 'hue_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 50), 'tip': 'Maximum hue shift'},
                    {'name': 'sat_shift_limit', 'type': 'int', 'value': 30, 'limits': (0, 100), 'tip': 'Maximum saturation shift'},
                    {'name': 'val_shift_limit', 'type': 'int', 'value': 20, 'limits': (0, 100), 'tip': 'Maximum value shift'},
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
                    {'name': 'aspect_ratio_min', 'type': 'float', 'value': 0.75, 'limits': (0.1, 2.0), 'tip': 'Minimum aspect ratio for cropping'},
                    {'name': 'aspect_ratio_max', 'type': 'float', 'value': 1.33, 'limits': (0.1, 2.0), 'tip': 'Maximum aspect ratio for cropping'},
                    {'name': 'probability', 'type': 'float', 'value': 1.0, 'limits': (0.0, 1.0), 'tip': 'Probability of random cropping'}
                ],
                'tip': 'Randomly crop parts of the image with specified area and aspect ratio constraints'
            }
        ]
        
        # Add all preset methods
        for method in preset_methods:
            self.addChild(method)
    
    # def _add_custom_button(self):
    #     """Add a button parameter for loading custom augmentation functions from files."""
    #     self.addChild({
    #         'name': 'Load Custom Augmentations',
    #         'type': 'action',
    #         'tip': 'Click to load custom augmentation functions from a Python file'
    #     })
        
    #     # Connect the action to the file loading function
    #     custom_button = self.child('Load Custom Augmentations')
    #     custom_button.sigActivated.connect(self._load_custom_augmentations)
    
    # def _load_custom_augmentations(self):
    #     """Load custom augmentation functions from a selected Python file."""
    #     from PySide6.QtWidgets import QFileDialog, QMessageBox
        
    #     # Open file dialog to select Python file
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         None,
    #         "Select Python file with custom augmentation functions",
    #         "",
    #         "Python Files (*.py)"
    #     )
        
    #     if not file_path:
    #         return
        
    #     try:
    #         # Load and parse the Python file
    #         custom_functions = self._extract_augmentation_functions(file_path)
            
    #         if not custom_functions:
    #             QMessageBox.warning(
    #                 None,
    #                 "No Functions Found",
    #                 "No valid augmentation functions found in the selected file.\n\n"
    #                 "Functions should accept 'image' parameter and return modified image."
    #             )
    #             return
            
    #         # Add each found function as a custom augmentation
    #         added_count = 0
    #         for func_name, func_info in custom_functions.items():
    #             if self._add_custom_function(func_name, func_info):
    #                 added_count += 1
            
    #         if added_count > 0:
    #             QMessageBox.information(
    #                 None,
    #                 "Functions Loaded",
    #                 f"Successfully loaded {added_count} custom augmentation function(s):\n" +
    #                 "\n".join(custom_functions.keys())
    #             )
    #         else:
    #             QMessageBox.warning(
    #                 None,
    #                 "No New Functions",
    #                 "All functions from the file are already loaded or invalid."
    #             )
                
    #     except Exception as e:
    #         QMessageBox.critical(
    #             None,
    #             "Error Loading File",
    #             f"Failed to load custom augmentations from file:\n{str(e)}"
    #         )
    
    # def _extract_augmentation_functions(self, file_path):
    #     """Extract valid augmentation functions from a Python file."""
    #     custom_functions = {}
        
    #     try:
    #         # Read and parse the file
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             content = f.read()
            
    #         # Parse the AST
    #         tree = ast.parse(content)
            
    #         # Find function definitions
    #         for node in ast.walk(tree):
    #             if isinstance(node, ast.FunctionDef):
    #                 func_name = node.name
                    
    #                 # Check if it's a valid augmentation function
    #                 if self._is_valid_augmentation_function(node):
    #                     # Extract function parameters
    #                     params = self._extract_function_parameters(node)
                        
    #                     # Extract docstring if available
    #                     docstring = ast.get_docstring(node) or f"Custom augmentation function: {func_name}"
                        
    #                     custom_functions[func_name] = {
    #                         'parameters': params,
    #                         'docstring': docstring,
    #                         'file_path': file_path,
    #                         'function_name': func_name
    #                     }
            
    #     except Exception as e:
    #         print(f"Error parsing file {file_path}: {e}")
        
    #     return custom_functions
    
    # def _is_valid_augmentation_function(self, func_node):
    #     """Check if a function is a valid augmentation function."""
    #     # Check if function has at least one parameter (should be 'image')
    #     if not func_node.args.args:
    #         return False
        
    #     # Check if first parameter is likely an image parameter
    #     first_param = func_node.args.args[0].arg
    #     if first_param not in ['image', 'img', 'x', 'data']:
    #         return False
        
    #     # Function should return something (basic check)
    #     has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
    #     if not has_return:
    #         return False
        
    #     return True
    
    def _extract_function_parameters(self, func_node):
        """Extract parameters from function definition (excluding 'image' parameter)."""
        params = []
        
        # Skip the first parameter (image) and extract others
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            
            # Try to infer parameter type and default values
            param_info = {
                'name': param_name,
                'type': 'float',  # Default type
                'default': 0.5,   # Default value
                'limits': (0.0, 1.0),
                'tip': f'Parameter for {param_name}'
            }
            
            # Basic type inference based on parameter name
            if 'angle' in param_name.lower():
                param_info.update({'type': 'float', 'default': 15.0, 'limits': (0.0, 180.0), 'suffix': '°'})
            elif 'prob' in param_name.lower() or 'p' == param_name.lower():
                param_info.update({'type': 'float', 'default': 0.5, 'limits': (0.0, 1.0)})
            elif 'strength' in param_name.lower() or 'intensity' in param_name.lower():
                param_info.update({'type': 'float', 'default': 1.0, 'limits': (0.0, 5.0)})
            elif 'size' in param_name.lower() or 'kernel' in param_name.lower():
                param_info.update({'type': 'int', 'default': 3, 'limits': (1, 15)})
            elif 'enable' in param_name.lower():
                param_info.update({'type': 'bool', 'default': True})
            
            params.append(param_info)
        
        # Check for default values in function definition
        if func_node.args.defaults:
            num_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                param_index = len(func_node.args.args) - num_defaults + i - 1  # -1 to skip image param
                if param_index >= 0 and param_index < len(params):
                    if isinstance(default, ast.Constant):
                        params[param_index]['default'] = default.value
                        # Update type based on default value
                        if isinstance(default.value, bool):
                            params[param_index]['type'] = 'bool'
                        elif isinstance(default.value, int):
                            params[param_index]['type'] = 'int'
                        elif isinstance(default.value, float):
                            params[param_index]['type'] = 'float'
        
        return params
    
    # def _add_custom_function(self, func_name, func_info):
    #     """Add a custom function as an augmentation method."""
    #     # Add (custom) suffix to distinguish from presets
    #     display_name = f"{func_name} (custom)"
        
    #     # Check if function already exists (check both original and display names)
    #     existing_names = [child.name() for child in self.children()]
    #     if func_name in existing_names or display_name in existing_names:
    #         return False
        
    #     # Create parameters list
    #     children = [
    #         {'name': 'enabled', 'type': 'bool', 'value': True, 'tip': f'Enable {func_name} augmentation'}
    #     ]
        
    #     # Add function-specific parameters
    #     for param_info in func_info['parameters']:
    #         children.append({
    #             'name': param_info['name'],
    #             'type': param_info['type'],
    #             'value': param_info['default'],
    #             'limits': param_info.get('limits'),
    #             'suffix': param_info.get('suffix', ''),
    #             'tip': param_info['tip']
    #         })
        
    #     # Add metadata parameters
    #     children.extend([
    #         {'name': 'file_path', 'type': 'str', 'value': func_info['file_path'], 'readonly': True, 'tip': 'Source file path'},
    #         {'name': 'function_name', 'type': 'str', 'value': func_info['function_name'], 'readonly': True, 'tip': 'Function name in source file'}
    #     ])
        
    #     # Create the augmentation method
    #     method_config = {
    #         'name': display_name,
    #         'type': 'group',
    #         'children': children,
    #         'removable': True,
    #         'renamable': False,  # Keep original function name
    #         'tip': func_info['docstring']
    #     }
        
    #     # Insert before the "Load Custom Augmentations" button
    #     # Find the button's index and insert before it
    #     button_index = None
    #     for i, child in enumerate(self.children()):
    #         if child.name() == 'Load Custom Augmentations':
    #             button_index = i
    #             break
        
    #     if button_index is not None:
    #         self.insertChild(button_index, method_config)
    #     else:
    #         # Fallback: add at the end if button not found
    #         self.addChild(method_config)
        
    #     return True
    
    # def addNew(self, typ=None):
    #     """Legacy method - no longer used since we load from files."""
    #     # This method is called by the parameter tree system but we use the button instead
    #     pass

# Register the custom parameter types
pTypes.registerParameterType('directory', DirectoryParameter, override=True)
pTypes.registerParameterType('directory_only', DirectoryOnlyParameter, override=True)
pTypes.registerParameterType('augmentation_group', AugmentationGroup, override=True)
pTypes.registerParameterType('preprocessing_group', PreprocessingGroup, override=True)
pTypes.registerParameterType('callbacks_group', CallbacksGroup, override=True)
pTypes.registerParameterType('loss_functions_group', LossFunctionsGroup, override=True)
pTypes.registerParameterType('metrics_group', MetricsGroup, override=True)

def get_parameter_tooltip(param_name, section_name=None):
    """Get tooltip text for a parameter based on its name and section."""
    tooltips = {
        # Data section tooltips
        'train_dir': 'Path to the directory containing training data files (images, TFRecords, etc.)',
        'val_dir': 'Path to the directory containing validation/test data files',
        'image_size': 'Input image dimensions [width, height] - images will be resized to this size',
        'batch_size': 'Number of samples processed together in each training step. Larger values use more memory but may train faster',
        'num_classes': 'Number of different classes/categories in your dataset (e.g., 1000 for ImageNet)',
        'shuffle': 'Whether to randomly shuffle the training data order for each epoch',
        
        # Model section tooltips
        'backbone_type': 'The neural network architecture to use as the feature extractor (ResNet, EfficientNet, etc.)',
        'model_id': 'Specific variant of the backbone architecture (e.g., 50 for ResNet-50, 18 for ResNet-18)',
        'dropout_rate': 'Probability of randomly setting input units to 0 during training to prevent overfitting (0.0-1.0)',
        'activation': 'Activation function used in the neural network layers (ReLU, Swish, GELU, etc.)',
        
        # Training section tooltips
        'epochs': 'Number of complete passes through the entire training dataset',
        'learning_rate_type': 'Strategy for adjusting the learning rate during training (exponential decay, cosine annealing, etc.)',
        'initial_learning_rate': 'Starting learning rate value - how quickly the model learns from data',
        'momentum': 'SGD momentum factor - helps accelerate gradients in relevant directions (typically 0.9)',
        'weight_decay': 'L2 regularization strength to prevent overfitting by penalizing large weights',
        'label_smoothing': 'Technique to prevent overconfident predictions by softening target labels (0.0-0.3)',
        
        # Runtime section tooltips
        'model_dir': 'Directory where model checkpoints and training outputs will be saved',
        'distribution_strategy': 'Strategy for distributed training across multiple GPUs or machines',
        'mixed_precision': 'Use lower precision (float16/bfloat16) to speed up training and reduce memory usage',
        'num_gpus': 'Number of GPUs to use for training (0 for CPU-only)',
        
        # Model Advanced tooltips
        'depth_multiplier': 'Multiplier for the number of layers in the backbone network',
        'stem_type': 'Type of initial stem layers in the network architecture',
        'se_ratio': 'Squeeze-and-Excitation ratio for channel attention mechanisms',
        'stochastic_depth_drop_rate': 'Probability of dropping entire layers during training for regularization',
        'scale_stem': 'Whether to scale the stem layers in the network',
        'resnetd_shortcut': 'Use ResNet-D style shortcut connections for improved accuracy',
        'replace_stem_max_pool': 'Replace max pooling in stem with strided convolution',
        'bn_trainable': 'Whether batch normalization layers are trainable during fine-tuning',
        'use_sync_bn': 'Use synchronized batch normalization across multiple GPUs',
        'norm_momentum': 'Momentum for batch normalization moving average (typically 0.99)',
        'norm_epsilon': 'Small constant for numerical stability in batch normalization',
        'add_head_batch_norm': 'Add batch normalization before the final classification layer',
        'kernel_initializer': 'Method for initializing convolutional layer weights',
        'output_softmax': 'Apply softmax activation to final output (usually False for training)',
        
        # Data Advanced tooltips
        'tfds_name': 'TensorFlow Datasets name if using TFDS instead of custom data',
        'tfds_split': 'Which split of TFDS to use (train, validation, test)',
        'cache': 'Cache dataset in memory for faster access (requires sufficient RAM)',
        'shuffle_buffer_size': 'Size of buffer for shuffling data - larger values provide better randomness',
        'cycle_length': 'Number of input elements to process concurrently in parallel',
        'block_length': 'Number of consecutive elements from each input to read',
        'drop_remainder': 'Drop the last batch if it has fewer samples than batch_size',
        'sharding': 'Enable data sharding for distributed training',
        'prefetch_buffer_size': 'Number of batches to prefetch for pipeline optimization',
        'dtype': 'Data type for input tensors (float32 recommended for most cases)',
        'file_type': 'Format of input data files (TFRecord, SSTable, RecordIO)',
        'image_field_key': 'Key name for image data in TFRecord files',
        'label_field_key': 'Key name for label data in TFRecord files',
        'decode_jpeg_only': 'Only decode JPEG images, skip other formats for speed',
        
        # Augmentation tooltips
        'aug_rand_hflip': 'Randomly flip images horizontally during training',
        'aug_crop': 'Apply random cropping augmentation during training',
        'crop_area_range': 'Range of crop area as fraction of original image [min, max]',
        'center_crop_fraction': 'Fraction of image to keep when center cropping (for validation)',
        'color_jitter': 'Amount of random color variation (brightness, contrast, etc.)',
        'randaug_magnitude': 'Magnitude of RandAugment transformations (0-30)',
        'tf_resize_method': 'Method for resizing images (bilinear, nearest, bicubic, area)',
        'three_augment': 'Apply Three-Augment policy for advanced data augmentation',
        'is_multilabel': 'Whether this is a multi-label classification task',
        
        # Preprocessing tooltips
        'Resizing': 'Resize images to target dimensions with support for 2D and 3D data',
        'Normalization': 'Normalize data using various methods (min-max, zero-center, standardization, etc.)',
        'target_size': 'Target dimensions for resizing',
        'interpolation': 'Interpolation method for resizing',
        'preserve_aspect_ratio': 'Whether to preserve aspect ratio during resize',
        'data_format': 'Data format (2D for images, 3D for volumes)',
        'method': 'Normalization/processing method',
        'min_value': 'Minimum value for min-max normalization',
        'max_value': 'Maximum value for min-max normalization',
        'mean': 'Mean values for zero-center normalization',
        'std': 'Standard deviation values for standardization',
        'axis': 'Axis along which to normalize',
        'epsilon': 'Small constant to avoid division by zero',
        'r': 'Red channel value',
        'g': 'Green channel value', 
        'b': 'Blue channel value',
        
        # Callbacks tooltips
        'Early Stopping': 'Stop training when monitored metric stops improving',
        'Learning Rate Scheduler': 'Adjust learning rate during training based on metrics or schedule',
        'Model Checkpoint': 'Save model checkpoints during training',
        'CSV Logger': 'Log training metrics to CSV file',
        'TensorBoard': 'Log training metrics and model graph to TensorBoard',
        'monitor': 'Metric to monitor',
        'patience': 'Number of epochs with no improvement to wait',
        
        # Loss Functions tooltips  
        'loss_functions': 'Configure loss functions for training optimization',
        'Categorical Crossentropy': 'Standard categorical crossentropy loss for multi-class classification',
        'Sparse Categorical Crossentropy': 'Categorical crossentropy with integer labels (not one-hot)',
        'Binary Crossentropy': 'Binary crossentropy loss for binary classification',
        'Mean Squared Error': 'Mean squared error loss for regression tasks',
        'Mean Absolute Error': 'Mean absolute error loss for regression tasks',
        'Focal Loss': 'Focal loss for addressing class imbalance',
        'Huber Loss': 'Huber loss (smooth L1 loss) for robust regression',
        'from_logits': 'Whether predictions are logits or probabilities',
        'label_smoothing': 'Label smoothing factor to prevent overconfident predictions',
        'reduction': 'Type of reduction to apply to the loss',
        'alpha': 'Weighting factor for rare class (Focal Loss)',
        'gamma': 'Focusing parameter (Focal Loss)',
        'delta': 'Threshold at which to change between MSE and MAE (Huber Loss)',
        
        # Metrics tooltips
        'metrics': 'Configure metrics for training and evaluation monitoring',
        'Accuracy': 'Standard accuracy metric for classification tasks',
        'Categorical Accuracy': 'Categorical accuracy metric for multi-class classification',
        'Sparse Categorical Accuracy': 'Sparse categorical accuracy for integer label classification',
        'Top-K Categorical Accuracy': 'Top-K accuracy metric (e.g., top-5 accuracy)',
        'Precision': 'Precision metric for classification tasks',
        'Recall': 'Recall metric for classification tasks',
        'F1 Score': 'F1 score metric (harmonic mean of precision and recall)',
        'AUC': 'Area Under the Curve (AUC) metric',
        'name': 'Name for this metric in logs and outputs',
        'k': 'Number of top predictions to consider (Top-K accuracy)',
        'average': 'Averaging strategy for multi-class metrics',
        'class_id': 'Class ID for binary metrics (None for multiclass)',
        'curve': 'Curve type (ROC or Precision-Recall) for AUC',
        'multi_class': 'Multiclass strategy (one-vs-rest or one-vs-one) for AUC',
        'min_delta': 'Minimum change to qualify as improvement',
        'mode': 'Direction of improvement (min/max)',
        'restore_best_weights': 'Restore model weights from best epoch',
        'scheduler_type': 'Type of learning rate scheduler',
        'factor': 'Factor by which learning rate is reduced',
        'min_lr': 'Minimum learning rate',
        'step_size': 'Period of learning rate decay',
        'gamma': 'Multiplicative factor of learning rate decay',
        'filepath': 'Path template for checkpoint files',
        'save_best_only': 'Save only the best model',
        'save_weights_only': 'Save only model weights (not full model)',
        'period': 'Interval between checkpoints',
        'filename': 'Path to CSV log file',
        'separator': 'Delimiter for CSV file',
        'append': 'Append to existing file or create new',
        'log_dir': 'Directory for TensorBoard logs',
        'histogram_freq': 'Frequency for histogram computation',
        'write_graph': 'Write model graph to TensorBoard',
        'write_images': 'Write model weights as images',
        'update_freq': 'Update frequency for logging',
        
        # New augmentation method tooltips
        'Horizontal Flip': 'Randomly flip images horizontally for data augmentation',
        'Vertical Flip': 'Randomly flip images vertically for data augmentation',
        'Rotation': 'Randomly rotate images by specified angle range',
        'Gaussian Noise': 'Add random Gaussian noise to images for robustness',
        'Brightness Adjustment': 'Randomly adjust image brightness within specified limits',
        'Contrast Adjustment': 'Randomly adjust image contrast within specified limits',
        'Color Jittering': 'Randomly adjust hue, saturation, and value for color variations',
        'Random Cropping': 'Randomly crop portions of images with area and aspect ratio constraints',
        'enabled': 'Enable or disable this augmentation method',
        'probability': 'Probability of applying this augmentation (0.0 = never, 1.0 = always)',
        'angle_range': 'Maximum rotation angle in degrees (±)',
        'variance_limit': 'Maximum variance for Gaussian noise',
        'brightness_limit': 'Maximum brightness adjustment (±)',
        'contrast_limit': 'Maximum contrast adjustment (±)',
        'hue_shift_limit': 'Maximum hue shift in degrees',
        'sat_shift_limit': 'Maximum saturation shift percentage',
        'val_shift_limit': 'Maximum value/brightness shift percentage',
        'crop_area_min': 'Minimum crop area as fraction of original image',
        'crop_area_max': 'Maximum crop area as fraction of original image',
        'aspect_ratio_min': 'Minimum aspect ratio for cropped region',
        'aspect_ratio_max': 'Maximum aspect ratio for cropped region',
        
        # Custom augmentation tooltips
        'min_angle': 'Minimum rotation angle for custom rotation',
        'max_angle': 'Maximum rotation angle for custom rotation',
        'noise_type': 'Type of noise to add (gaussian, uniform, salt_pepper)',
        'intensity': 'Intensity of the noise effect',
        'blur_type': 'Type of blur to apply (gaussian, motion, median)',
        'blur_limit': 'Maximum blur kernel size',
        'distortion_type': 'Type of distortion to apply (elastic, perspective, barrel)',
        'distortion_strength': 'Strength of the distortion effect',
        'filter_type': 'Type of filter to apply (sharpen, emboss, edge_enhance)',
        'filter_strength': 'Strength of the filter effect',
        
        # Training Advanced tooltips
        'train_tf_while_loop': 'Use TensorFlow while loops for training (usually faster)',
        'train_tf_function': 'Use tf.function compilation for training loops',
        'eval_tf_function': 'Use tf.function compilation for evaluation loops',
        'steps_per_loop': 'Number of training steps per loop iteration',
        'summary_interval': 'How often to write training summaries (in steps)',
        'checkpoint_interval': 'How often to save model checkpoints (in steps)',
        'max_to_keep': 'Maximum number of recent checkpoints to keep',
        'validation_interval': 'How often to run validation evaluation (in steps)',
        'validation_steps': 'Number of validation steps per evaluation (-1 for full dataset)',
        'loss_upper_bound': 'Upper bound for loss values - training stops if exceeded',
        'one_hot_labels': 'Use one-hot encoded labels instead of sparse labels',
        'use_binary_cross_entropy': 'Use binary cross-entropy loss for binary classification',
        'soft_labels': 'Use soft labels instead of hard labels for training',
        
        # Evaluation tooltips
        'top_k': 'Compute top-K accuracy (e.g., top-5 accuracy for ImageNet)',
        'report_per_class_metrics': 'Report precision, recall, F1 for each class separately',
        'best_checkpoint_metric': 'Metric to use for selecting the best checkpoint',
        'best_checkpoint_export_subdir': 'Subdirectory to export the best checkpoint',
        'best_checkpoint_metric_comp': 'Whether higher or lower metric values are better',
        
        # Runtime Advanced tooltips
        'enable_xla': 'Enable XLA (Accelerated Linear Algebra) compilation for faster execution',
        'run_eagerly': 'Run in eager mode for debugging (disables optimizations)',
        'per_gpu_thread_count': 'Number of threads per GPU for data processing',
        'num_packs': 'Number of gradient packs for gradient compression',
        'loss_scale': 'Loss scaling factor for mixed precision training',
        'batchnorm_spatial_persistent': 'Use persistent batch normalization for spatial data',
        'tpu_settings': 'Special settings for TPU training',
        'all_reduce_alg': 'Algorithm for all-reduce operations in distributed training',
        
        # Special group parameter tooltips
        'width': 'Image width in pixels',
        'height': 'Image height in pixels',
        'min': 'Minimum value for the range',
        'max': 'Maximum value for the range',
    }
    
    # Section-specific tooltips for groups
    section_tooltips = {
        'basic': 'Essential parameters that most users need to configure',
        'advanced': 'Advanced parameters for expert users and fine-tuning',
        'data': 'Dataset and data loading configuration',
        'model': 'Neural network architecture settings',
        'training': 'Training process configuration',
        'runtime': 'Runtime and system configuration',
        'model_advanced': 'Advanced model architecture parameters',
        'data_advanced': 'Advanced data pipeline configuration',
        'augmentation': 'Data augmentation and preprocessing settings',
        'preprocessing': 'Data preprocessing methods including resizing and normalization',
        'callbacks': 'Training callbacks for monitoring, checkpointing, and scheduling',
        'training_advanced': 'Advanced training loop and optimization settings',
        'evaluation': 'Model evaluation and metrics configuration',
        'runtime_advanced': 'Advanced runtime and performance settings',
    }
    
    # Return tooltip based on parameter name or section
    if param_name in tooltips:
        return tooltips[param_name]
    elif param_name in section_tooltips:
        return section_tooltips[param_name]
    else:
        return f'Configuration parameter: {param_name}'

def dict_to_params(data, name="Config"):
    """Convert a nested dictionary to Parameter tree structure with enhanced parameter types and tooltips."""
    if isinstance(data, dict):
        # Check if this is a special augmentation group type
        if data.get('type') == 'augmentation_group':
            return {
                'name': data.get('name', name),
                'type': 'augmentation_group',
                'tip': get_parameter_tooltip('augmentation')
            }
        
        # Check if this is a special preprocessing group type
        if data.get('type') == 'preprocessing_group':
            return {
                'name': data.get('name', name),
                'type': 'preprocessing_group',
                'tip': get_parameter_tooltip('preprocessing')
            }
        
        # Check if this is a special callbacks group type
        if data.get('type') == 'callbacks_group':
            return {
                'name': data.get('name', name),
                'type': 'callbacks_group',
                'tip': get_parameter_tooltip('callbacks')
            }
        
        # Check if this is a special loss functions group type
        if data.get('type') == 'loss_functions_group':
            return {
                'name': data.get('name', name),
                'type': 'loss_functions_group',
                'tip': get_parameter_tooltip('loss_functions')
            }
        
        # Check if this is a special metrics group type
        if data.get('type') == 'metrics_group':
            return {
                'name': data.get('name', name),
                'type': 'metrics_group',
                'tip': get_parameter_tooltip('metrics')
            }
        
        children = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary - create a group
                group_param = dict_to_params(value, key)
                group_param['tip'] = get_parameter_tooltip(key)
                children.append(group_param)
            elif isinstance(value, list):
                # Handle list values
                if len(value) == 2 and all(isinstance(x, int) for x in value):
                    # Image size or similar pair - create group with two int parameters
                    if key == 'image_size':
                        children.append({
                            'name': key,
                            'type': 'group',
                            'tip': get_parameter_tooltip(key),
                            'children': [
                                {'name': 'width', 'type': 'int', 'value': value[0], 'limits': [1, 2048], 'tip': get_parameter_tooltip('width')},
                                {'name': 'height', 'type': 'int', 'value': value[1], 'limits': [1, 2048], 'tip': get_parameter_tooltip('height')}
                            ]
                        })
                    elif key == 'crop_area_range':
                        children.append({
                            'name': key,
                            'type': 'group',
                            'tip': get_parameter_tooltip(key),
                            'children': [
                                {'name': 'min', 'type': 'float', 'value': value[0], 'limits': [0.0, 1.0], 'step': 0.01, 'tip': get_parameter_tooltip('min')},
                                {'name': 'max', 'type': 'float', 'value': value[1], 'limits': [0.0, 1.0], 'step': 0.01, 'tip': get_parameter_tooltip('max')}
                            ]
                        })
                    else:
                        # Generic two-element list
                        children.append({
                            'name': key,
                            'type': 'str',
                            'value': str(value),
                            'tip': get_parameter_tooltip(key)
                        })
                else:
                    # Other lists - convert to string representation
                    children.append({
                        'name': key,
                        'type': 'str',
                        'value': str(value),
                        'tip': get_parameter_tooltip(key)
                    })
            else:
                # Handle special directory/file parameters
                if key in ['train_dir', 'val_dir'] and isinstance(value, str):
                    children.append({
                        'name': key,
                        'type': 'directory',
                        'value': value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key in ['model_dir'] and isinstance(value, str):
                    children.append({
                        'name': key,
                        'type': 'directory_only',
                        'value': value,
                        'tip': get_parameter_tooltip(key)
                    })
                # Handle choice parameters
                elif key == 'backbone_type':
                    values = ['resnet', 'efficientnet', 'mobilenet', 'vit', 'densenet']
                    # Ensure current value is valid, default to first item if not
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'activation':
                    values = ['relu', 'swish', 'gelu', 'leaky_relu', 'tanh']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'distribution_strategy':
                    values = ['mirrored', 'multi_worker_mirrored', 'tpu', 'parameter_server']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'learning_rate_type':
                    values = ['exponential', 'polynomial', 'cosine', 'constant', 'piecewise_constant']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'mixed_precision':
                    values = ['None', 'float16', 'bfloat16']  # Convert None to 'None' string for dropdown
                    # Handle None value properly
                    if value is None:
                        current_value = 'None'
                    else:
                        current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'optimizer_type':
                    values = ['sgd', 'adam', 'adamw', 'rmsprop', 'lars']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'file_type':
                    values = ['tfrecord', 'sstable', 'recordio']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'tf_resize_method':
                    values = ['bilinear', 'nearest', 'bicubic', 'area']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'kernel_initializer':
                    values = ['random_uniform', 'random_normal', 'glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'stem_type':
                    values = ['v0', 'v1', 'v2']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'dtype':
                    values = ['float32', 'float16', 'bfloat16']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key == 'best_checkpoint_metric_comp':
                    values = ['higher', 'lower']
                    current_value = value if value in values else values[0]
                    children.append({
                        'name': key,
                        'type': 'list',
                        'limits': values,
                        'value': current_value,
                        'tip': get_parameter_tooltip(key)
                    })
                # Handle numeric parameters with appropriate ranges
                elif key in ['batch_size', 'num_classes', 'epochs', 'model_id']:
                    limits = {
                        'batch_size': [1, 1024],
                        'num_classes': [1, 100000],
                        'epochs': [1, 1000],
                        'model_id': [18, 152]
                    }
                    children.append({
                        'name': key,
                        'type': 'int',
                        'value': int(value) if value else 0,
                        'limits': limits.get(key, [0, 1000000]),
                        'tip': get_parameter_tooltip(key)
                    })
                elif key in ['dropout_rate', 'learning_rate', 'initial_learning_rate', 'momentum', 'weight_decay', 'label_smoothing', 
                            'depth_multiplier', 'se_ratio', 'stochastic_depth_drop_rate', 'norm_momentum', 'norm_epsilon',
                            'color_jitter', 'center_crop_fraction']:
                    step = 0.01 if 'rate' in key or 'momentum' in key else 0.001
                    limits = [0.0, 1.0] if 'rate' in key or 'momentum' in key else [0.0, 10.0]
                    if key == 'initial_learning_rate':
                        limits = [1e-6, 1.0]
                        step = 0.001
                    elif key == 'norm_epsilon':
                        limits = [1e-8, 1e-3]
                        step = 1e-6
                    elif key == 'weight_decay':
                        limits = [0.0, 0.01]
                        step = 1e-5
                    
                    children.append({
                        'name': key,
                        'type': 'float',
                        'value': float(value) if value else 0.0,
                        'limits': limits,
                        'step': step,
                        'tip': get_parameter_tooltip(key)
                    })
                elif key in ['steps_per_loop', 'summary_interval', 'checkpoint_interval', 'validation_interval',
                            'shuffle_buffer_size', 'cycle_length', 'block_length', 'max_to_keep', 'validation_steps',
                            'top_k', 'randaug_magnitude', 'per_gpu_thread_count', 'num_packs', 'num_gpus']:
                    limits = {
                        'steps_per_loop': [1, 10000],
                        'summary_interval': [1, 10000],
                        'checkpoint_interval': [1, 10000],
                        'validation_interval': [1, 10000],
                        'shuffle_buffer_size': [1, 100000],
                        'cycle_length': [1, 100],
                        'block_length': [1, 100],
                        'max_to_keep': [1, 50],
                        'validation_steps': [-1, 10000],
                        'top_k': [1, 10],
                        'randaug_magnitude': [0, 30],
                        'per_gpu_thread_count': [0, 16],
                        'num_packs': [1, 8],
                        'num_gpus': [0, 8]
                    }
                    children.append({
                        'name': key,
                        'type': 'int',
                        'value': int(value) if value is not None else -1,
                        'limits': limits.get(key, [0, 100000]),
                        'tip': get_parameter_tooltip(key)
                    })
                # Handle boolean parameters
                elif isinstance(value, bool):
                    children.append({
                        'name': key,
                        'type': 'bool',
                        'value': value,
                        'tip': get_parameter_tooltip(key)
                    })
                else:
                    # Default string parameter
                    children.append({
                        'name': key,
                        'type': 'str',
                        'value': str(value) if value is not None else '',
                        'tip': get_parameter_tooltip(key)
                    })
        
        return {
            'name': name,
            'type': 'group',
            'children': children,
            'tip': get_parameter_tooltip(name)
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
            'value': data,
            'tip': get_parameter_tooltip(name)
        }

def params_to_dict(param):
    """Convert Parameter tree back to dictionary with proper handling of special cases."""
    if param.hasChildren():
        result = {}
        for child in param.children():
            child_name = child.name()
            
            # Special handling for augmentation group
            if isinstance(child, AugmentationGroup):
                result[child_name] = extract_augmentation_config(child)
            elif isinstance(child, PreprocessingGroup):
                result[child_name] = extract_preprocessing_config(child)
            elif isinstance(child, CallbacksGroup):
                result[child_name] = extract_callbacks_config(child)
            elif child.hasChildren():
                # Handle special group parameters like image_size or crop_area_range
                if child_name == 'image_size' and len(child.children()) == 2:
                    width_child = next((c for c in child.children() if c.name() == 'width'), None)
                    height_child = next((c for c in child.children() if c.name() == 'height'), None)
                    if width_child and height_child:
                        result[child_name] = [width_child.value(), height_child.value()]
                    else:
                        result[child_name] = params_to_dict(child)
                elif child_name == 'crop_area_range' and len(child.children()) == 2:
                    min_child = next((c for c in child.children() if c.name() == 'min'), None)
                    max_child = next((c for c in child.children() if c.name() == 'max'), None)
                    if min_child and max_child:
                        result[child_name] = [min_child.value(), max_child.value()]
                    else:
                        result[child_name] = params_to_dict(child)
                else:
                    # Regular nested group
                    result[child_name] = params_to_dict(child)
            else:
                # Leaf parameter
                try:
                    value = child.value()
                except Exception:
                    # Handle parameters without values set
                    value = None
                    
                # Handle None values properly
                if value == 'None' or value == '' or value is None:
                    # Check if this parameter should be None vs empty string
                    if child_name in ['mixed_precision', 'prefetch_buffer_size', 'tpu_settings', 'all_reduce_alg', 'loss_scale']:
                        result[child_name] = None
                    else:
                        result[child_name] = value
                else:
                    result[child_name] = value
        return result
    else:
        return param.value()

def extract_augmentation_config(aug_group):
    """Extract configuration from AugmentationGroup parameter."""
    config = {}
    
    for method_param in aug_group.children():
        method_name = method_param.name()
        method_config = {}
        
        # Extract configuration for each augmentation method
        for param in method_param.children():
            method_config[param.name()] = param.value()
        
        config[method_name] = method_config
    
    return config

def extract_preprocessing_config(prep_group):
    """Extract configuration from PreprocessingGroup parameter."""
    config = {}
    
    for method_param in prep_group.children():
        method_name = method_param.name()
        
        # Skip the button
        if method_name == 'Load Custom Preprocessing':
            continue
            
        method_config = {}
        
        # Extract configuration for each preprocessing method
        for param in method_param.children():
            if param.hasChildren():
                # Handle nested parameters (like target_size, mean, std)
                nested_config = {}
                for child_param in param.children():
                    nested_config[child_param.name()] = child_param.value()
                method_config[param.name()] = nested_config
            else:
                method_config[param.name()] = param.value()
        
        config[method_name] = method_config
    
    return config

def extract_callbacks_config(callbacks_group):
    """Extract configuration from CallbacksGroup parameter."""
    config = {}
    
    for callback_param in callbacks_group.children():
        callback_name = callback_param.name()
        
        # Skip the button
        if callback_name == 'Load Custom Callbacks':
            continue
            
        callback_config = {}
        
        # Extract configuration for each callback
        for param in callback_param.children():
            callback_config[param.name()] = param.value()
        
        config[callback_name] = callback_config
    
    return config

# ---------------------------
# Bridge: GUI <-> Callbacks
# ---------------------------
class Bridge(QObject):
    log = Signal(str)
    update_plots = Signal(int, list, list, list, list)  # epoch_count, t_loss, v_loss, t_acc, v_acc
    progress = Signal(int)
    finished = Signal()

BRIDGE = Bridge()

# ---------------------------
# Qt callback class for tf-models-official
# ---------------------------

class QtBridgeCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_train_steps: int = 1000, log_every_n: int = 1):
        super().__init__()
        self.total_train_steps = int(total_train_steps)
        self.log_every_n = int(log_every_n)
        self._epoch = 0
        # lists for plotting
        self._train_losses = []
        self._val_losses = []
        self._train_accs = []
        self._val_accs = []
        self._batch_count = 0
    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        if self.total_train_steps:
            pct = int(min(100, (self._batch_count / max(1, self.total_train_steps)) * 100))
            BRIDGE.progress.emit(pct)
        if self._batch_count % max(1, self.log_every_n) == 0:
            BRIDGE.log.emit(f"[batch {self._batch_count}] {logs or {}}")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._epoch += 1
        tl = logs.get("loss", float("nan"))
        ta = logs.get("accuracy", logs.get("acc", float("nan")))
        vl = logs.get("val_loss", float("nan"))
        va = logs.get("val_accuracy", logs.get("val_acc", float("nan")))
        self._train_losses.append(float(tl) if tl is not None else float("nan"))
        self._val_losses.append(float(vl) if vl is not None else float("nan"))
        self._train_accs.append(float(ta) if ta is not None else float("nan"))
        self._val_accs.append(float(va) if va is not None else float("nan"))
        BRIDGE.log.emit(f"[Epoch {self._epoch}] loss={tl:.4f} acc={ta:.4f} val_loss={vl} val_acc={va}")
        BRIDGE.update_plots.emit(self._epoch, self._train_losses, self._val_losses, self._train_accs, self._val_accs)
    def on_train_end(self, logs=None):
        BRIDGE.log.emit("Callback: training ended.")
        BRIDGE.finished.emit()

# ---------------------------
# Helper: map GUI config -> exp_config
# This is a defensive mapping because different experiments expect different fields.
# You should adapt this function to match the exact exp_name you will use.
# ---------------------------
def map_gui_to_expconfig(gui_cfg: Dict[str, Any], exp_name: str):
    """
    Returns a ConfigDict from exp_factory.get_exp_config(exp_name) with fields
    updated from comprehensive gui_cfg structure.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow / tf-models-official not available")

    exp_cfg = exp_factory.get_exp_config(exp_name)  # ConfigDict

    # Map basic configuration
    try:
        # Runtime settings
        if 'runtime' in gui_cfg:
            runtime = gui_cfg['runtime']
            exp_cfg.runtime.model_dir = runtime.get('model_dir', './model_dir')
            if runtime.get('distribution_strategy'):
                exp_cfg.runtime.distribution_strategy = runtime['distribution_strategy']
            if runtime.get('mixed_precision'):
                exp_cfg.runtime.mixed_precision_dtype = runtime['mixed_precision']
            if runtime.get('num_gpus'):
                exp_cfg.runtime.num_gpus = runtime['num_gpus']
    except Exception:
        pass

    # Map data configuration
    try:
        if 'data' in gui_cfg:
            data = gui_cfg['data']
            
            # Training data
            if data.get('train_dir'):
                exp_cfg.task.train_data.input_path = data['train_dir']
            if data.get('batch_size'):
                exp_cfg.task.train_data.global_batch_size = int(data['batch_size'])
            if data.get('image_size') and isinstance(data['image_size'], list) and len(data['image_size']) >= 2:
                exp_cfg.task.model.input_size = data['image_size'][:2]
            elif data.get('image_size'):
                size = int(data['image_size']) if isinstance(data['image_size'], (int, str)) else 224
                exp_cfg.task.model.input_size = [size, size]
                
            # Validation data
            if data.get('val_dir'):
                exp_cfg.task.validation_data.input_path = data['val_dir']
                exp_cfg.task.validation_data.global_batch_size = int(data.get('batch_size', 32))
            
            # Number of classes
            if data.get('num_classes'):
                exp_cfg.task.model.num_classes = int(data['num_classes'])
    except Exception as e:
        print(f"Error mapping data config: {e}")

    # Map model configuration
    try:
        if 'model' in gui_cfg:
            model = gui_cfg['model']
            
            if model.get('backbone_type'):
                exp_cfg.task.model.backbone.type = model['backbone_type']
            if model.get('model_id'):
                if model['backbone_type'] == 'resnet':
                    exp_cfg.task.model.backbone.resnet.model_id = int(model['model_id'])
            if model.get('dropout_rate') is not None:
                exp_cfg.task.model.dropout_rate = float(model['dropout_rate'])
            if model.get('activation'):
                exp_cfg.task.model.norm_activation.activation = model['activation']
    except Exception as e:
        print(f"Error mapping model config: {e}")

    # Map training configuration
    try:
        if 'training' in gui_cfg:
            training = gui_cfg['training']
            
            if training.get('epochs'):
                exp_cfg.trainer.train_steps = int(training['epochs']) * 1000  # Approximate
            
            # Learning rate setup
            if training.get('initial_learning_rate'):
                lr_type = training.get('learning_rate_type', 'exponential')
                if lr_type == 'exponential':
                    exp_cfg.trainer.optimizer_config.learning_rate = {
                        'type': 'exponential',
                        'exponential': {
                            'initial_learning_rate': float(training['initial_learning_rate']),
                            'decay_steps': 10000,
                            'decay_rate': 0.96
                        }
                    }
                elif lr_type == 'constant':
                    exp_cfg.trainer.optimizer_config.learning_rate = {
                        'type': 'constant',
                        'constant': {
                            'learning_rate': float(training['initial_learning_rate'])
                        }
                    }
            
            # Optimizer settings
            if training.get('momentum') and training.get('weight_decay'):
                exp_cfg.trainer.optimizer_config.optimizer = {
                    'type': 'sgd',
                    'sgd': {
                        'momentum': float(training['momentum']),
                        'weight_decay': float(training['weight_decay'])
                    }
                }
            
            # Loss configuration
            if training.get('label_smoothing') is not None:
                exp_cfg.task.losses.label_smoothing = float(training['label_smoothing'])
    except Exception as e:
        print(f"Error mapping training config: {e}")

    # Map augmentation configuration
    try:
        if 'augmentation' in gui_cfg:
            aug = gui_cfg['augmentation']
            
            # Handle new augmentation structure
            if isinstance(aug, dict):
                # Check for horizontal flip
                hflip = aug.get('Horizontal Flip', {})
                if hflip.get('enabled', False):
                    exp_cfg.task.train_data.aug_rand_hflip = True
                else:
                    exp_cfg.task.train_data.aug_rand_hflip = False
                
                # Check for random cropping
                crop = aug.get('Random Cropping', {})
                if crop.get('enabled', False):
                    exp_cfg.task.train_data.aug_crop = True
                    # Set crop area range if available
                    min_area = crop.get('crop_area_min', 0.08)
                    max_area = crop.get('crop_area_max', 1.0)
                    exp_cfg.task.train_data.crop_area_range = [min_area, max_area]
                else:
                    exp_cfg.task.train_data.aug_crop = False
                
                # Check for color jittering
                color_jitter = aug.get('Color Jittering', {})
                if color_jitter.get('enabled', False):
                    # Map to color jitter strength (simplified mapping)
                    hue_shift = color_jitter.get('hue_shift_limit', 20) / 50.0  # Normalize to 0-1
                    exp_cfg.task.train_data.color_jitter = hue_shift
                else:
                    exp_cfg.task.train_data.color_jitter = 0.0
            
            # Fallback to legacy structure for backward compatibility
            else:
                exp_cfg.task.train_data.aug_rand_hflip = bool(aug.get('aug_rand_hflip', True))
                exp_cfg.task.train_data.aug_crop = bool(aug.get('aug_crop', True))
                if aug.get('crop_area_range'):
                    exp_cfg.task.train_data.crop_area_range = aug['crop_area_range']
                if aug.get('color_jitter') is not None:
                    exp_cfg.task.train_data.color_jitter = float(aug['color_jitter'])
                if aug.get('randaug_magnitude') is not None:
                    exp_cfg.task.train_data.randaug_magnitude = int(aug['randaug_magnitude'])
    except Exception as e:
        print(f"Error mapping augmentation config: {e}")

    # Map training advanced settings
    try:
        if 'training_advanced' in gui_cfg:
            adv = gui_cfg['training_advanced']
            if adv.get('steps_per_loop'):
                exp_cfg.trainer.steps_per_loop = int(adv['steps_per_loop'])
            if adv.get('checkpoint_interval'):
                exp_cfg.trainer.checkpoint_interval = int(adv['checkpoint_interval'])
            if adv.get('validation_interval'):
                exp_cfg.trainer.validation_interval = int(adv['validation_interval'])
            if adv.get('max_to_keep'):
                exp_cfg.trainer.max_to_keep = int(adv['max_to_keep'])
    except Exception as e:
        print(f"Error mapping advanced training config: {e}")

    return exp_cfg

# ---------------------------
# Trainer thread that calls train_lib.run_experiment
# ---------------------------
class TFModelsTrainerThread(threading.Thread):
    def __init__(self, gui_cfg: Dict[str, Any], exp_name: str = "image_classification_imagenet", resume_ckpt: str = None):
        super().__init__()
        self.gui_cfg = copy.deepcopy(gui_cfg)
        self.exp_name = exp_name
        self.resume_ckpt = resume_ckpt
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):

        try:
            BRIDGE.log.emit(f"Building exp_config for '{self.exp_name}' ...")
            exp_cfg = map_gui_to_expconfig(self.gui_cfg, self.exp_name)

            # set init_checkpoint if resume path provided
            if self.resume_ckpt:
                try:
                    exp_cfg.task.init_checkpoint = self.resume_ckpt
                    BRIDGE.log.emit(f"Set init_checkpoint: {self.resume_ckpt}")
                except Exception:
                    pass

            # ensure model_dir using new config structure
            runtime_cfg = self.gui_cfg.get("runtime", {})
            model_dir = runtime_cfg.get("model_dir", "./model_dir")
            try:
                exp_cfg.runtime.model_dir = model_dir
            except Exception:
                try:
                    exp_cfg.runtime = exp_cfg.get("runtime", {})
                    exp_cfg.runtime.model_dir = model_dir
                except Exception:
                    pass
            os.makedirs(model_dir, exist_ok=True)

            # Add our QtBridgeCallback - get epochs from training config
            training_cfg = self.gui_cfg.get("training", {})
            total_steps = int(training_cfg.get("epochs", 1))
            cb = {"type": "QtBridgeCallback", "total_train_steps": total_steps, "log_every_n": 1}
            # ensure callbacks list exists
            try:
                if hasattr(exp_cfg, "callbacks") and exp_cfg.callbacks:
                    # remove previous QtBridgeCallback entries
                    exp_cfg.callbacks = [c for c in exp_cfg.callbacks if not (isinstance(c, dict) and c.get("type") == "QtBridgeCallback")]
                    exp_cfg.callbacks.append(cb)
                else:
                    exp_cfg.callbacks = [cb]
            except Exception:
                try:
                    exp_cfg.callbacks = [cb]
                except Exception:
                    pass

            BRIDGE.log.emit("Starting train_lib.run_experiment ...")
            # Note: some run_experiment wrappers accept distribution_strategy as arg; we pass runtime setting if present
            ds = None
            distribution = None
            try:
                distribution = getattr(exp_cfg.runtime, "distribution_strategy", None)
            except Exception:
                distribution = None

            # Run training. This is a blocking call.
            train_lib.run_experiment(
                distribution_strategy=distribution or "mirrored",
                mode="train",
                params=exp_cfg
            )

            BRIDGE.log.emit("train_lib.run_experiment returned (training finished).")
        except Exception as e:
            BRIDGE.log.emit(f"Training exception: {e}")
        finally:
            BRIDGE.finished.emit()

# ---------------------------
# MainWindow: similar UI as earlier, but start TFModelsTrainerThread
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self, experiment_name: str = "image_classification_imagenet"):
        super().__init__()
        self.setWindowTitle("TF-Models-Official GUI Trainer")
        self.resize(1600, 1000)

        # initialize GUI config (comprehensive TensorFlow Models config)
        comprehensive_config = create_comprehensive_config()
        self.gui_cfg = {
            **comprehensive_config['basic'],
            **comprehensive_config['advanced']
        }
        
        # Also maintain the comprehensive structure for the parameter tree
        self.comprehensive_cfg = comprehensive_config
        self.experiment_name = experiment_name
        self.trainer_thread: TFModelsTrainerThread = None
        self.resume_ckpt_path = None
        self.tb_proc = None

        # toolbar
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        act_save_json = QAction("Save JSON", self); act_save_json.triggered.connect(lambda: self.save_config("json"))
        toolbar.addAction(act_save_json)
        act_save_yaml = QAction("Save YAML", self); act_save_yaml.triggered.connect(lambda: self.save_config("yaml"))
        toolbar.addAction(act_save_yaml)
        act_load = QAction("Load Config", self); act_load.triggered.connect(self.load_config)
        toolbar.addAction(act_load)
        toolbar.addSeparator()
        act_ckpt = QAction("Choose checkpoint", self); act_ckpt.triggered.connect(self.choose_checkpoint)
        toolbar.addAction(act_ckpt)
        act_model_dir = QAction("Choose model_dir", self); act_model_dir.triggered.connect(self.choose_model_dir)
        toolbar.addAction(act_model_dir)

        # left layout: config tree + augment controls + controls
        left_layout = QVBoxLayout()
        
        # Create ParameterTree with comprehensive config data organized in Basic/Advanced sections
        self.params = Parameter.create(**dict_to_params(self.comprehensive_cfg, "Configuration"))
        self.tree = ParameterTree()
        self.tree.setParameters(self.params, showTop=False)
        
        # Set up directory parameter callbacks
        self._setup_directory_callbacks()
        
        # Connect to parameter change signals
        self.params.sigTreeStateChanged.connect(self._on_param_changed)
        
        # Add tooltip handling using a simple approach
        try:
            # Install event filter on the parameter tree to catch mouse events
            self.tree.installEventFilter(self)
            print("Installed event filter on ParameterTree")
            
        except Exception as e:
            print(f"Could not install event filter: {e}")
            
        # Add a test for tooltip display using button clicks
        self.last_clicked_param = None
        
        left_layout.addWidget(QLabel("Config"))
        left_layout.addWidget(self.tree, stretch=3)

        # Preview Data button
        btn_preview = QPushButton("Preview Data")
        btn_preview.clicked.connect(self.preview_augmentation)
        left_layout.addWidget(btn_preview)

        # training controls (directory selection is now integrated in parameter tree)
        self.btn_start = QPushButton("Start Training"); self.btn_start.clicked.connect(self.start_training); left_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop Training"); self.btn_stop.clicked.connect(self.stop_training); left_layout.addWidget(self.btn_stop)

        self.progress = QProgressBar(); left_layout.addWidget(self.progress)

        left_widget = QWidget(); left_widget.setLayout(left_layout)

        # right layout: TensorBoard + logs + plots
        right_layout = QVBoxLayout()
        self.tb_view = QWebEngineView()
        
        self.tb_view.setUrl(f"http://localhost:6006")  # Default URL for TensorBoard
        right_layout.addWidget(QLabel("TensorBoard"))
        right_layout.addWidget(self.tb_view, stretch=2)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(QLabel("Logs"))
        right_layout.addWidget(self.log_edit, stretch=1)
        self.plot = pg.PlotWidget(title="Metrics")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True)
        self.curves = {"train_loss": self.plot.plot(pen='r', name="train_loss"), "val_loss": self.plot.plot(pen='b', name="val_loss"),
                       "train_acc": self.plot.plot(pen='g', name="train_acc"), "val_acc": self.plot.plot(pen='y', name="val_acc")}
        # right_layout.addWidget(self.plot, stretch=2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(right_widget, stretch=2)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # signals
        BRIDGE.log.connect(self.append_log); BRIDGE.update_plots.connect(self.on_update_plots); BRIDGE.progress.connect(self.progress.setValue)
        BRIDGE.finished.connect(self.on_training_finished)

        # state
        self.resume_ckpt_path = None

    def closeEvent(self, event):
        """Handle window closing to clean up resources."""
        super().closeEvent(event)

    # tree helpers
    def _setup_directory_callbacks(self):
        """Set up callbacks for directory browser parameters."""
        # With the new DirectoryParameter implementation, changes are handled
        # automatically through the sigValueChanged signal, so we don't need
        # special callback setup here anymore
        pass
    
    def _on_param_changed(self, param, changes):
        """Handle parameter changes from the ParameterTree."""
        try:
            # Update gui_cfg when parameters change
            self.gui_cfg = params_to_dict(self.params)
            self.append_log("Config updated from parameter tree")
            # Update other UI elements if needed
            self.apply_cfg_to_widgets()
            
            # Also try to show tooltip for the changed parameter
            # for change in changes:
            #     param_obj, change_type, data = change
            #     if change_type in ['value', 'expanded']:
            #         param_name = param_obj.name()
            #         self.last_clicked_param = param_name
                    # self._show_param_tooltip_simple(param_name)
                    # print(f"Parameter changed: {param_name} ({change_type})")
                    
        except Exception as e:
            self.append_log(f"Error updating config from parameters: {e}")
            self.append_log(f"Error updating config from parameters: {e}")
    
    # def _show_param_tooltip_simple(self, param_name):
    #     """Simple tooltip display method."""
    #     try:
    #         tooltip_text = get_parameter_tooltip(param_name)
    #         # self.tooltip_label.setText(f"<b>{param_name}:</b> {tooltip_text}")
    #         print(f"Showing tooltip for: {param_name}")
    #     except Exception as e:
    #         print(f"Error showing tooltip: {e}")
    
    # def _on_tree_item_clicked(self, param, changes):
    #     """Handle tree item clicks to show tooltips in the label."""
    #     try:
    #         for change in changes:
    #             param_obj, change_type, data = change
    #             if change_type == 'selected':
    #                 # Get the parameter name and find its tooltip
    #                 param_name = param_obj.name()
    #                 # tooltip_text = get_parameter_tooltip(param_name)
                    
    #                 # Update the tooltip label
    #                 # self.tooltip_label.setText(f"<b>{param_name}:</b> {tooltip_text}")
    #                 break
    #     except Exception as e:
    #         print(f"Error showing tooltip: {e}")
    
    def _on_tree_item_clicked_direct(self, item, column):
        """Handle direct tree widget item clicks to show tooltips."""
        try:
            # Try multiple ways to get parameter name from tree item
            param_name = None
            
            # Method 1: Try to get from itemMap
            if hasattr(self.tree, 'itemMap') and item in self.tree.itemMap:
                param_item = self.tree.itemMap[item]
                if hasattr(param_item, 'param'):
                    param_name = param_item.param.name()
            
            # Method 2: Try to get from item text
            if not param_name and item:
                param_name = item.text(0)  # Get text from first column
                
            # Method 3: Try to get from item data
            if not param_name and item:
                param_data = item.data(0, Qt.UserRole)
                if param_data:
                    param_name = str(param_data)
            
            # if param_name:
            #     tooltip_text = get_parameter_tooltip(param_name)
            #     # self.tooltip_label.setText(f"<b>{param_name}:</b> {tooltip_text}")
            #     print(f"Showing tooltip for: {param_name}")  # Debug print
            # else:
            #     print(f"Could not determine parameter name for item: {item}")  # Debug print
                
        except Exception as e:
            print(f"Error showing tooltip: {e}")
    
    def _on_tree_selection_changed(self):
        """Handle tree selection changes to show tooltips."""
        try:
            # Find selected items in the parameter tree
            tree_widgets = self.tree.findChildren(QTreeWidget)
            if tree_widgets:
                tree_widget = tree_widgets[0]
                current_item = tree_widget.currentItem()
                if current_item:
                    # Use the same logic as click handler
                    self._on_tree_item_clicked_direct(current_item, 0)
        except Exception as e:
            print(f"Error in selection changed: {e}")  # Debug print
    
    # def _connect_parameter_signals(self, param):
    #     """Connect signals for parameter clicks recursively."""
    #     try:
    #         # Connect to parameter change signals that might indicate selection
    #         if hasattr(param, 'sigClicked'):
    #             param.sigClicked.connect(lambda p: self._show_param_tooltip(p.name()))
    #         elif hasattr(param, 'sigActivated'):
    #             param.sigActivated.connect(lambda p: self._show_param_tooltip(p.name()))
            
    #         # Recursively connect child parameters
    #         if hasattr(param, 'children'):
    #             for child in param.children():
    #                 self._connect_parameter_signals(child)
                    
    #     except Exception as e:
    #         print(f"Error connecting parameter signals: {e}")
    
    # def _show_param_tooltip(self, param_name):
    #     """Show tooltip for a parameter name."""
    #     try:
    #         tooltip_text = get_parameter_tooltip(param_name)
    #         self.tooltip_label.setText(f"<b>{param_name}:</b> {tooltip_text}")
    #         print(f"Showing tooltip for parameter: {param_name}")  # Debug print
    #     except Exception as e:
            # print(f"Error showing parameter tooltip: {e}")
    
    # def _on_param_selected(self, param):
    #     """Handle parameter selection from ParameterTree."""
    #     try:
    #         if param:
    #             param_name = param.name()
    #             self._show_param_tooltip(param_name)
    #     except Exception as e:
    #         print(f"Error in parameter selection: {e}")
    
    # def _on_current_item_changed(self, current, previous):
    #     """Handle current item changes in tree widget."""
    #     try:
    #         if current:
    #             param_name = current.text(0)
    #             if param_name:
    #                 self._show_param_tooltip(param_name)
    #     except Exception as e:
    #         print(f"Error in current item changed: {e}")
    
    # def _refresh_tooltip_from_tree(self):
    #     """Periodically check the tree for selected items and show tooltips."""
    #     try:
    #         # Try to get the currently selected parameter from the tree
    #         selected_items = self.tree.selectedItems()
    #         if selected_items and hasattr(selected_items[0], 'param'):
    #             param_name = selected_items[0].param.name()
    #             current_text = self.tooltip_label.text()
    #             if param_name not in current_text:  # Only update if different
    #                 self._show_param_tooltip(param_name)
    #     except Exception as e:
    #         # Fail silently for timer-based updates
    #         pass
    
    # def _test_tooltip(self):
    #     """Test method to verify tooltip functionality."""
    #     test_params = ['batch_size', 'learning_rate', 'model_dir', 'train_dir', 'epochs']
    #     import random
    #     param_name = random.choice(test_params)
    #     self._show_param_tooltip(param_name)
    #     print(f"Test: showing tooltip for {param_name}")
    
    def eventFilter(self, obj, event):
        """Event filter to catch mouse clicks on the parameter tree."""
        if obj == self.tree and event.type() == event.Type.MouseButtonPress:
            try:
                # Get the position and try to find what was clicked
                pos = event.position().toPoint()
                print(f"Mouse clicked on ParameterTree at position: {pos}")
                
                # Try to use a timer to check selection after click
                QTimer.singleShot(100, self._delayed_tooltip_update)
                
            except Exception as e:
                print(f"Error in event filter: {e}")
        
        return super().eventFilter(obj, event)
    
    def _delayed_tooltip_update(self):
        """Update tooltip after a brief delay to allow selection to update."""
        try:
            self._refresh_tooltip_from_tree()
        except Exception as e:
            print(f"Error in delayed tooltip update: {e}")
    
    def _on_tree_view_clicked(self, index):
        """Handle QTreeView clicks."""
        try:
            if index.isValid():
                # Try to get parameter name from the index
                param_name = index.data()
                if param_name:
                    self._show_param_tooltip(param_name)
                    print(f"TreeView clicked: {param_name}")
        except Exception as e:
            print(f"Error in tree view clicked: {e}")
    
    def refresh_tree(self):
        """Update the ParameterTree with current comprehensive config data."""
        try:
            # Update comprehensive config from current gui_cfg
            self.sync_gui_to_comprehensive()
            
            # Recreate the parameter structure
            new_params = Parameter.create(**dict_to_params(self.comprehensive_cfg, "Configuration"))
            
            # Disconnect old signals
            if hasattr(self, 'params'):
                self.params.sigTreeStateChanged.disconnect()
            
            # Set new parameters
            self.params = new_params
            self.tree.setParameters(self.params, showTop=False)
            
            # Set up directory callbacks
            self._setup_directory_callbacks()
            
            # Reconnect signals
            self.params.sigTreeStateChanged.connect(self._on_param_changed)
            
        except Exception as e:
            self.append_log(f"Error refreshing tree: {e}")
    
    def sync_gui_to_comprehensive(self):
        """Sync the flat gui_cfg back to the comprehensive config structure."""
        try:
            # Update basic configuration
            if 'data' in self.gui_cfg:
                self.comprehensive_cfg['basic']['data'].update(self.gui_cfg['data'])
            if 'model' in self.gui_cfg:
                self.comprehensive_cfg['basic']['model'].update(self.gui_cfg['model'])
            if 'training' in self.gui_cfg:
                self.comprehensive_cfg['basic']['training'].update(self.gui_cfg['training'])
            if 'runtime' in self.gui_cfg:
                self.comprehensive_cfg['basic']['runtime'].update(self.gui_cfg['runtime'])
                
            # Update advanced configuration sections as needed
            for section in ['model_advanced', 'data_advanced', 'augmentation', 'training_advanced', 'evaluation', 'runtime_advanced']:
                if section in self.gui_cfg:
                    self.comprehensive_cfg['advanced'][section].update(self.gui_cfg[section])
                    
        except Exception as e:
            self.append_log(f"Error syncing to comprehensive config: {e}")
    
    def write_back_tree(self):
        """Extract data from the parameter tree back to both comprehensive and flat gui_cfg."""
        try:
            # Extract data from parameter tree
            tree_data = params_to_dict(self.params)
            
            # Update comprehensive config
            if 'basic' in tree_data:
                self.comprehensive_cfg['basic'] = tree_data['basic']
            if 'advanced' in tree_data:
                self.comprehensive_cfg['advanced'] = tree_data['advanced']
            
            # Flatten to gui_cfg for backward compatibility
            self.gui_cfg = {}
            if 'basic' in tree_data:
                self.gui_cfg.update(tree_data['basic'])
            if 'advanced' in tree_data:
                self.gui_cfg.update(tree_data['advanced'])
                
            self.append_log("Config updated from parameter tree")
        except Exception as e:
            self.append_log(f"Could not write back tree data: {e}")

    # file ops
    def save_config(self, fmt="json"):
        self.sync_aug_to_cfg()
        path, _ = QFileDialog.getSaveFileName(self, "Save config", filter=f"*.{fmt}")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(self.gui_cfg, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(self.gui_cfg, f, allow_unicode=True)
        QMessageBox.information(self, "Saved", f"Saved config to {path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", filter="*.json *.yaml *.yml")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                self.gui_cfg = json.load(f)
            else:
                self.gui_cfg = yaml.safe_load(f)
        self.apply_cfg_to_widgets()
        self.refresh_tree()
        QMessageBox.information(self, "Loaded", f"Loaded: {path}")

    # data/model path
    def set_train_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select train_dir")
        if d:
            self.gui_cfg["data"]["train_dir"] = d
            self.refresh_tree()

    def set_val_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select val_dir")
        if d:
            self.gui_cfg["data"]["val_dir"] = d
            self.refresh_tree()

    def choose_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint (ckpt prefix or file)")
        if p:
            self.resume_ckpt_path = p
            self.append_log(f"Will resume from: {p}")

    def choose_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose model_dir")
        if d:
            if "runtime" not in self.gui_cfg:
                self.gui_cfg["runtime"] = {}
            self.gui_cfg["runtime"]["model_dir"] = d
            self.refresh_tree()

    # augment panel sync
    def apply_cfg_to_widgets(self):
        """Apply configuration to widgets (augmentation panel removed)."""
        # Note: Augmentation widgets have been removed from UI
        pass

    def sync_aug_to_cfg(self):
        """Sync augmentation configuration (widgets removed from UI)."""
        # Note: Augmentation widgets have been removed from UI
        # Configuration is now managed through the parameter tree
        pass
    
    def update_config_value(self, path_str, value):
        """Update a config value by dot-separated path string."""
        try:
            path = path_str.split('.')
            current = self.gui_cfg
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = value
            self.refresh_tree()
            self.append_log(f"Updated config: {path_str} = {value}")
        except Exception as e:
            self.append_log(f"Error updating config {path_str}: {e}")

    # preview augmentation (uses albumentations)
    def preview_augmentation(self):
        if not ALBU_AVAILABLE:
            QMessageBox.warning(self, "Missing libs", "Install albumentations/opencv-python for preview.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose image for preview")
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "Read failed", "Cannot read image.")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Get configuration from parameter tree instead of widgets
        self.write_back_tree()
        
        # Get image size from new config structure
        data_cfg = self.gui_cfg.get("data", {})
        image_size = data_cfg.get("image_size", [224, 224])
        if isinstance(image_size, list) and len(image_size) >= 2:
            size = int(image_size[0])
        else:
            size = 224
            
        # Use augmentation config from parameter tree
        aug_cfg = self.gui_cfg.get("augmentation", {})
        pipe = build_albu_pipeline(aug_cfg, size, size)
        samples = []
        for _ in range(4):
            try:
                aug = pipe(image=img_rgb)["image"]
                if aug.dtype != np.uint8 and np.max(aug) <= 1.1:
                    aug = np.clip(aug * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                aug = cv2.resize(img_rgb, (size, size))
            samples.append(aug)
        dlg = QDialog(self)
        dlg.setWindowTitle("Data Preview")
        layout = QGridLayout(dlg)
        pix = np_to_qpixmap(img_rgb)
        w0 = QLabel(); w0.setPixmap(pix.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel("Original"),0,0); layout.addWidget(w0,1,0)
        for i, s in enumerate(samples):
            p = np_to_qpixmap(s); l = QLabel(); l.setPixmap(p.scaled(300,300, Qt.KeepAspectRatio)); layout.addWidget(QLabel(f"Aug {i+1}"),0,i+1); layout.addWidget(l,1,i+1)
        dlg.exec()

    # logging / plotting handlers
    def append_log(self, text):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {text}")

    def on_update_plots(self, epoch, tl, vl, ta, va):
        x = list(range(1, epoch+1))
        try:
            self.curves["train_loss"].setData(x, tl)
            self.curves["val_loss"].setData(x, vl)
            self.curves["train_acc"].setData(x, ta)
            self.curves["val_acc"].setData(x, va)
        except Exception:
            pass

    # tensorboard
    def start_tensorboard(self, model_dir):
        try:
            port = 6006
            if self.tb_proc and getattr(self.tb_proc, "poll", None) is None and self.tb_proc.poll() is None:
                pass
            else:
                self.tb_proc = subprocess.Popen(["tensorboard", "--logdir", model_dir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.tb_view.setUrl(f"http://localhost:{port}")
            self.append_log(f"TensorBoard -> {model_dir}")
        except Exception as e:
            self.append_log(f"Failed to start TensorBoard: {e}")

    # training controls
    def start_training(self):
        if not TF_AVAILABLE:
            QMessageBox.critical(self, "TF missing", "TensorFlow or tf-models-official not available.")
            return
        # sync GUI config from parameter tree
        self.write_back_tree()
        
        # Get model_dir from new config structure
        runtime_cfg = self.gui_cfg.get("runtime", {})
        model_dir = runtime_cfg.get("model_dir", "./model_dir")
        
        os.makedirs(model_dir, exist_ok=True)
        self.start_tensorboard(model_dir)
        # start trainer thread that runs train_lib.run_experiment
        self.append_log("Launching tf-models-official trainer thread...")
        t = TFModelsTrainerThread(self.gui_cfg, exp_name=self.experiment_name, resume_ckpt=self.resume_ckpt_path)
        self.trainer_thread = t
        t.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

    def stop_training(self):
        if self.trainer_thread:
            self.append_log("Requested stop — trainer thread will attempt graceful stop.")
            self.trainer_thread.stop()
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def on_training_finished(self):
        self.append_log("Training finished (thread signalled).")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.progress.setValue(100)

# helpers used above (np_to_qpixmap, build_albu_pipeline) - include simple definitions here:

def create_custom_albumentations_transform(file_path, function_name, config):
    """Create a custom Albumentations transform from a Python function."""
    if not ALBU_AVAILABLE:
        return None
    
    try:
        # Load the module from file
        spec = importlib.util.spec_from_file_location("custom_augmentations", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        if not hasattr(module, function_name):
            print(f"Function {function_name} not found in {file_path}")
            return None
        
        custom_function = getattr(module, function_name)
        
        # Create a wrapper for Albumentations
        class CustomTransform(A.ImageOnlyTransform):
            def __init__(self, custom_func, func_config, always_apply=False, p=1.0):
                super().__init__(always_apply, p)
                self.custom_func = custom_func
                self.func_config = func_config
            
            def apply(self, image, **params):
                try:
                    # Extract parameters for the function (excluding enabled, file_path, function_name)
                    func_params = {}
                    for key, value in self.func_config.items():
                        if key not in ['enabled', 'file_path', 'function_name']:
                            func_params[key] = value
                    
                    # Call the custom function
                    result = self.custom_func(image, **func_params)
                    return result if result is not None else image
                except Exception as e:
                    print(f"Error in custom augmentation function: {e}")
                    return image
        
        # Get probability from config
        probability = config.get('probability', 0.5)
        if 'p' in config:
            probability = config['p']
        
        return CustomTransform(custom_function, config, p=probability)
        
    except Exception as e:
        print(f"Error creating custom transform from {file_path}: {e}")
        return None

def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h,w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h,w,c = img.shape
        if c == 3:
            qimg = QImage(img.data, w, h, w*3, QImage.Format.Format_RGB888)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if 'cv2' in globals() else img[:,:,0]
            h,w = gray.shape
            qimg = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())

def build_albu_pipeline(aug_cfg: Dict[str,Any], target_h:int, target_w:int):
    if not ALBU_AVAILABLE:
        raise RuntimeError("albumentations / cv2 missing")
    
    transforms = []
    
    # Handle new augmentation structure
    if isinstance(aug_cfg, dict):
        # Horizontal flip
        hflip = aug_cfg.get('Horizontal Flip', {})
        if hflip.get('enabled', False):
            prob = hflip.get('probability', 0.5)
            transforms.append(A.HorizontalFlip(p=prob))
        
        # Vertical flip
        vflip = aug_cfg.get('Vertical Flip', {})
        if vflip.get('enabled', False):
            prob = vflip.get('probability', 0.5)
            transforms.append(A.VerticalFlip(p=prob))
        
        # Rotation
        rotation = aug_cfg.get('Rotation', {})
        if rotation.get('enabled', False):
            angle = rotation.get('angle_range', 15.0)
            prob = rotation.get('probability', 0.5)
            transforms.append(A.Rotate(limit=angle, border_mode=cv2.BORDER_REFLECT_101, p=prob))
        
        # Gaussian noise
        noise = aug_cfg.get('Gaussian Noise', {})
        if noise.get('enabled', False):
            variance = noise.get('variance_limit', 0.01)
            prob = noise.get('probability', 0.2)
            transforms.append(A.GaussNoise(var_limit=(0, variance * 255**2), p=prob))
        
        # Brightness adjustment
        brightness = aug_cfg.get('Brightness Adjustment', {})
        if brightness.get('enabled', False):
            limit = brightness.get('brightness_limit', 0.2)
            prob = brightness.get('probability', 0.5)
            transforms.append(A.RandomBrightness(limit=limit, p=prob))
        
        # Contrast adjustment
        contrast = aug_cfg.get('Contrast Adjustment', {})
        if contrast.get('enabled', False):
            limit = contrast.get('contrast_limit', 0.2)
            prob = contrast.get('probability', 0.5)
            transforms.append(A.RandomContrast(limit=limit, p=prob))
        
        # Color jittering
        color_jitter = aug_cfg.get('Color Jittering', {})
        if color_jitter.get('enabled', False):
            hue = color_jitter.get('hue_shift_limit', 20)
            sat = color_jitter.get('sat_shift_limit', 30)
            val = color_jitter.get('val_shift_limit', 20)
            prob = color_jitter.get('probability', 0.5)
            transforms.append(A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=prob))
        
        # Random cropping
        crop = aug_cfg.get('Random Cropping', {})
        if crop.get('enabled', False):
            min_area = crop.get('crop_area_min', 0.08)
            max_area = crop.get('crop_area_max', 1.0)
            min_ratio = crop.get('aspect_ratio_min', 0.75)
            max_ratio = crop.get('aspect_ratio_max', 1.33)
            prob = crop.get('probability', 1.0)
            transforms.append(A.RandomResizedCrop(
                height=target_h, width=target_w, 
                scale=(min_area, max_area), 
                ratio=(min_ratio, max_ratio), 
                p=prob
            ))
        
        # Handle custom augmentations loaded from files
        for key, value in aug_cfg.items():
            if isinstance(value, dict) and value.get('enabled', False):
                # Check if this is a custom function (has file_path and function_name)
                if 'file_path' in value and 'function_name' in value:
                    try:
                        # Load and apply custom function
                        custom_transform = create_custom_albumentations_transform(
                            value['file_path'], 
                            value['function_name'], 
                            value
                        )
                        if custom_transform:
                            transforms.append(custom_transform)
                    except Exception as e:
                        print(f"Error loading custom augmentation {key}: {e}")
                
                # Handle legacy custom augmentations (without file_path)
                elif 'Custom' in key:
                    try:
                        if 'Rotation' in key:
                            min_angle = value.get('min_angle', -45.0)
                            max_angle = value.get('max_angle', 45.0)
                            prob = value.get('probability', 0.5)
                            transforms.append(A.Rotate(limit=(min_angle, max_angle), p=prob))
                        
                        elif 'Noise' in key:
                            noise_type = value.get('noise_type', 'gaussian')
                            intensity = value.get('intensity', 0.05)
                            prob = value.get('probability', 0.3)
                            if noise_type == 'gaussian':
                                transforms.append(A.GaussNoise(var_limit=(0, intensity * 255**2), p=prob))
                            elif noise_type == 'salt_pepper':
                                transforms.append(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=prob))
                        
                        elif 'Blur' in key:
                            blur_type = value.get('blur_type', 'gaussian')
                            blur_limit = value.get('blur_limit', 3)
                            prob = value.get('probability', 0.2)
                            if blur_type == 'gaussian':
                                transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=prob))
                            elif blur_type == 'motion':
                                transforms.append(A.MotionBlur(blur_limit=(3, blur_limit), p=prob))
                            elif blur_type == 'median':
                                transforms.append(A.MedianBlur(blur_limit=blur_limit, p=prob))
                        
                    except Exception as e:
                        print(f"Error adding legacy custom augmentation {key}: {e}")
    
    # Fallback to legacy structure
    else:
        if aug_cfg.get("aug_rand_hflip"):
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("three_augment"):
            transforms.append(A.VerticalFlip(p=0.5))
        
        # Use RandAugment magnitude instead of rotation limit
        randaug_mag = aug_cfg.get("randaug_magnitude", 0)
        if randaug_mag and randaug_mag > 0:
            transforms.append(A.Rotate(limit=min(randaug_mag, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.6))
        
        # Use crop area range
        crop_range = aug_cfg.get("crop_area_range", [0.08, 1.0])
        if isinstance(crop_range, list) and len(crop_range) >= 2:
            transforms.append(A.RandomResizedCrop(height=target_h, width=target_w, scale=(crop_range[0], crop_range[1]), ratio=(0.9,1.1), p=0.6))
        
        # Color jitter
        color_jitter = aug_cfg.get("color_jitter", 0.0)
        if color_jitter and color_jitter > 0:
            transforms.append(A.RandomBrightnessContrast(brightness_limit=color_jitter, contrast_limit=color_jitter, p=0.6))
    
    # Always add resize and normalize at the end
    if not any(isinstance(t, A.RandomResizedCrop) for t in transforms):
        transforms.append(A.Resize(target_h, target_w))
    transforms.append(A.Normalize())
    return A.Compose(transforms)

# run
def main():
    app = QApplication(sys.argv)
    win = MainWindow(experiment_name="image_classification_imagenet")
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
