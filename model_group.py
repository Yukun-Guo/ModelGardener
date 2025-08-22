"""
Model configuration group for dynamic model-specific parameters and custom model loading.
"""

import os
import sys
import importlib.util
import inspect
import tensorflow as tf

# Try to import PyQt5 (for GUI functionality) but make it optional
try:
    from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QLabel
    from PyQt5.QtCore import Qt
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    # Create dummy classes for testing without GUI
    class QFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return "", ""
    class QMessageBox:
        @staticmethod
        def information(*args, **kwargs):
            pass
        @staticmethod
        def warning(*args, **kwargs):
            pass
        @staticmethod
        def critical(*args, **kwargs):
            pass

try:
    from pyqtgraph.parametertree import Parameter, ParameterTree
    from pyqtgraph.parametertree.parameterTypes import GroupParameter
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    # Create dummy base class for testing
    class GroupParameter:
        def __init__(self, **opts):
            self.opts = opts
            self.name_value = opts.get('name', 'test')
            self._children = []
        
        def name(self):
            return self.name_value
            
        def children(self):
            return self._children
            
        def addChild(self, child):
            self._children.append(child)
            
        def removeChild(self, child):
            if child in self._children:
                self._children.remove(child)
        
        def child(self, name):
            for child in self._children:
                if child.name() == name:
                    return child
            return None


class ModelGroup(GroupParameter):
    """
    Custom parameter group for model-specific configuration with dynamic parameters
    based on the selected model and support for custom model loading.
    """
    
    def __init__(self, **opts):
        self.model_name = opts.get('value', {}).get('model_name', 'ResNet-50')
        self.task_type = opts.get('value', {}).get('task_type', 'image_classification')
        self.custom_model_path = None
        self.custom_model_function = None
        
        opts['type'] = 'group'
        opts['addText'] = "Add model parameter..."
        super().__init__(**opts)
        
        # Initialize with default model parameters
        self._update_model_parameters()
        
    def _update_model_parameters(self):
        """Update parameters based on the current model selection."""
        try:
            # Clear existing model-specific parameters but keep core ones
            current_children = list(self.children())
            for child in current_children:
                if child.name() not in ['model_family', 'model_name']:
                    self.removeChild(child)
            
            # Get model-specific parameters based on model_name
            model_params = self._get_model_parameters(self.model_name, self.task_type)
            
            # Add model-specific parameters
            for param_name, param_config in model_params.items():
                if PYQTGRAPH_AVAILABLE:
                    self.addChild(Parameter.create(name=param_name, **param_config))
            
            # Add custom model button
            if PYQTGRAPH_AVAILABLE:
                self.addChild(Parameter.create(
                    name='load_custom_model',
                    type='action',
                    title='Load Custom Model...',
                    tip='Load a custom model from a Python file'
                ))
                
                # Connect the custom model button
                custom_button_param = self.child('load_custom_model')
                if custom_button_param:
                    custom_button_param.sigActivated.connect(self._load_custom_model)
                
        except Exception as e:
            print(f"Error updating model parameters: {e}")
    
    def _get_model_parameters(self, model_name, task_type):
        """Get model-specific parameters based on model name and task type."""
        params = {}
        
        # Common parameters for all models
        params['input_shape'] = {
            'type': 'group',
            'children': [
                {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image height'},
                {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image width'},
                {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4], 'tip': 'Number of input channels'}
            ]
        }
        
        params['num_classes'] = {
            'type': 'int',
            'value': 1000,
            'limits': [1, 50000],
            'tip': 'Number of output classes for classification'
        }
        
        # Model family specific parameters
        if 'resnet' in model_name.lower():
            params.update(self._get_resnet_parameters(model_name))
        elif 'efficientnet' in model_name.lower():
            params.update(self._get_efficientnet_parameters(model_name))
        elif 'mobilenet' in model_name.lower():
            params.update(self._get_mobilenet_parameters(model_name))
        elif 'vit' in model_name.lower() or 'vision_transformer' in model_name.lower():
            params.update(self._get_vit_parameters(model_name))
        elif 'yolo' in model_name.lower():
            params.update(self._get_yolo_parameters(model_name))
        elif 'unet' in model_name.lower():
            params.update(self._get_unet_parameters(model_name))
        elif 'deeplabv3' in model_name.lower():
            params.update(self._get_deeplab_parameters(model_name))
        
        # Task-specific parameters
        if task_type == 'object_detection':
            params.update(self._get_detection_parameters())
        elif task_type == 'semantic_segmentation':
            params.update(self._get_segmentation_parameters())
            
        return params
    
    def _get_resnet_parameters(self, model_name):
        """ResNet-specific parameters."""
        return {
            'dropout_rate': {
                'type': 'float',
                'value': 0.0,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Dropout rate before final classification layer'
            },
            'activation': {
                'type': 'list',
                'value': 'relu',
                'limits': ['relu', 'swish', 'gelu', 'mish'],
                'tip': 'Activation function for the model'
            },
            'use_se': {
                'type': 'bool',
                'value': False,
                'tip': 'Use Squeeze-and-Excitation blocks'
            },
            'se_ratio': {
                'type': 'float',
                'value': 0.25,
                'limits': [0.0, 1.0],
                'step': 0.05,
                'tip': 'Squeeze-and-Excitation ratio (only if use_se is True)'
            }
        }
    
    def _get_efficientnet_parameters(self, model_name):
        """EfficientNet-specific parameters."""
        return {
            'dropout_rate': {
                'type': 'float',
                'value': 0.2,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Dropout rate in the final classification layer'
            },
            'drop_connect_rate': {
                'type': 'float',
                'value': 0.2,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Drop connect rate for stochastic depth'
            },
            'depth_divisor': {
                'type': 'int',
                'value': 8,
                'limits': [1, 16],
                'tip': 'Depth divisor for channel dimensions'
            },
            'width_coefficient': {
                'type': 'float',
                'value': 1.0,
                'limits': [0.5, 2.0],
                'step': 0.1,
                'tip': 'Width scaling coefficient'
            },
            'depth_coefficient': {
                'type': 'float',
                'value': 1.0,
                'limits': [0.5, 2.0],
                'step': 0.1,
                'tip': 'Depth scaling coefficient'
            }
        }
    
    def _get_mobilenet_parameters(self, model_name):
        """MobileNet-specific parameters."""
        return {
            'alpha': {
                'type': 'float',
                'value': 1.0,
                'limits': [0.35, 1.4],
                'step': 0.05,
                'tip': 'Width multiplier (alpha parameter)'
            },
            'dropout': {
                'type': 'float',
                'value': 0.001,
                'limits': [0.0, 0.9],
                'step': 0.001,
                'tip': 'Dropout rate'
            },
            'depth_multiplier': {
                'type': 'int',
                'value': 1,
                'limits': [1, 4],
                'tip': 'Depth multiplier for depthwise convolution'
            }
        }
    
    def _get_vit_parameters(self, model_name):
        """Vision Transformer-specific parameters."""
        return {
            'patch_size': {
                'type': 'int',
                'value': 16,
                'limits': [8, 32],
                'tip': 'Size of image patches'
            },
            'num_layers': {
                'type': 'int',
                'value': 12,
                'limits': [6, 24],
                'tip': 'Number of transformer layers'
            },
            'hidden_size': {
                'type': 'int',
                'value': 768,
                'limits': [256, 1536],
                'tip': 'Hidden size of transformer'
            },
            'num_heads': {
                'type': 'int',
                'value': 12,
                'limits': [4, 24],
                'tip': 'Number of attention heads'
            },
            'mlp_dim': {
                'type': 'int',
                'value': 3072,
                'limits': [1024, 6144],
                'tip': 'MLP hidden dimension'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.1,
                'limits': [0.0, 0.5],
                'step': 0.05,
                'tip': 'Dropout rate'
            }
        }
    
    def _get_yolo_parameters(self, model_name):
        """YOLO-specific parameters."""
        return {
            'anchors_per_scale': {
                'type': 'int',
                'value': 3,
                'limits': [1, 5],
                'tip': 'Number of anchors per scale'
            },
            'iou_threshold': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.1, 0.9],
                'step': 0.05,
                'tip': 'IoU threshold for NMS'
            },
            'confidence_threshold': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.1, 0.9],
                'step': 0.05,
                'tip': 'Confidence threshold for detection'
            },
            'max_detections': {
                'type': 'int',
                'value': 100,
                'limits': [10, 1000],
                'tip': 'Maximum number of detections per image'
            }
        }
    
    def _get_unet_parameters(self, model_name):
        """U-Net-specific parameters."""
        return {
            'filters': {
                'type': 'int',
                'value': 64,
                'limits': [16, 256],
                'tip': 'Number of filters in the first layer'
            },
            'num_layers': {
                'type': 'int',
                'value': 4,
                'limits': [3, 6],
                'tip': 'Number of encoder/decoder layers'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Dropout rate'
            },
            'batch_norm': {
                'type': 'bool',
                'value': True,
                'tip': 'Use batch normalization'
            },
            'activation': {
                'type': 'list',
                'value': 'relu',
                'limits': ['relu', 'leaky_relu', 'swish', 'gelu'],
                'tip': 'Activation function'
            }
        }
    
    def _get_deeplab_parameters(self, model_name):
        """DeepLab-specific parameters."""
        return {
            'output_stride': {
                'type': 'int',
                'value': 16,
                'limits': [8, 32],
                'tip': 'Output stride for the backbone'
            },
            'aspp_rates': {
                'type': 'str',
                'value': '6,12,18',
                'tip': 'ASPP dilation rates (comma-separated)'
            },
            'decoder_channels': {
                'type': 'int',
                'value': 256,
                'limits': [64, 512],
                'tip': 'Number of channels in decoder'
            }
        }
    
    def _get_detection_parameters(self):
        """Object detection-specific parameters."""
        return {
            'anchor_sizes': {
                'type': 'str',
                'value': '32,64,128,256,512',
                'tip': 'Anchor sizes (comma-separated)'
            },
            'aspect_ratios': {
                'type': 'str',
                'value': '0.5,1.0,2.0',
                'tip': 'Anchor aspect ratios (comma-separated)'
            }
        }
    
    def _get_segmentation_parameters(self):
        """Semantic segmentation-specific parameters."""
        return {
            'ignore_label': {
                'type': 'int',
                'value': 255,
                'limits': [0, 255],
                'tip': 'Label value to ignore in loss calculation'
            },
            'use_auxiliary_loss': {
                'type': 'bool',
                'value': False,
                'tip': 'Use auxiliary loss for training'
            }
        }
    
    def _load_custom_model(self):
        """Load a custom model from a Python file."""
        try:
            if not PYQT5_AVAILABLE:
                print("GUI not available - custom model loading requires PyQt5")
                return
                
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Custom Model File",
                "",
                "Python Files (*.py);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Validate and load the custom model
            success, model_info = self._validate_custom_model(file_path)
            
            if success:
                self.custom_model_path = file_path
                self.custom_model_function = model_info
                
                # Add custom model parameters
                self._add_custom_model_parameters(model_info)
                
                QMessageBox.information(
                    None,
                    "Custom Model Loaded",
                    f"Successfully loaded custom model: {model_info['name']}\n"
                    f"Type: {model_info['type']}\n"
                    f"File: {os.path.basename(file_path)}"
                )
            else:
                QMessageBox.warning(
                    None,
                    "Invalid Custom Model",
                    f"Could not load custom model from {os.path.basename(file_path)}.\n"
                    f"Error: {model_info}"
                )
                
        except Exception as e:
            if PYQT5_AVAILABLE:
                QMessageBox.critical(
                    None,
                    "Error Loading Custom Model",
                    f"An error occurred while loading the custom model:\n{str(e)}"
                )
            else:
                print(f"Error loading custom model: {e}")
    
    def _validate_custom_model(self, file_path):
        """Validate that the file contains a valid custom model definition."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_model", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for functions that return keras models
            model_functions = []
            model_classes = []
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    # Check if function signature suggests it returns a model
                    sig = inspect.signature(obj)
                    # Simple heuristic: function that can take common model parameters
                    if any(param in sig.parameters for param in ['input_shape', 'num_classes', 'inputs']):
                        model_functions.append((name, obj))
                
                elif inspect.isclass(obj):
                    # Check if class inherits from keras Model
                    try:
                        if (hasattr(tf.keras.models, 'Model') and 
                            issubclass(obj, tf.keras.models.Model)):
                            model_classes.append((name, obj))
                    except:
                        pass
            
            if model_functions:
                return True, {
                    'name': model_functions[0][0],
                    'type': 'function',
                    'object': model_functions[0][1],
                    'file_path': file_path
                }
            elif model_classes:
                return True, {
                    'name': model_classes[0][0],
                    'type': 'class',
                    'object': model_classes[0][1],
                    'file_path': file_path
                }
            else:
                return False, "No valid model function or class found. Expected function returning keras model or class inheriting from keras.Model"
                
        except Exception as e:
            return False, str(e)
    
    def _add_custom_model_parameters(self, model_info):
        """Add parameters specific to the custom model."""
        try:
            # Remove existing custom parameters
            existing_custom = self.child('custom_model_info')
            if existing_custom:
                self.removeChild(existing_custom)
            
            # Add custom model info
            custom_params = {
                'name': 'custom_model_info',
                'type': 'group',
                'title': f"Custom Model: {model_info['name']}",
                'children': [
                    {
                        'name': 'model_type',
                        'type': 'str',
                        'value': model_info['type'],
                        'readonly': True,
                        'tip': 'Type of custom model (function or class)'
                    },
                    {
                        'name': 'file_path',
                        'type': 'str', 
                        'value': os.path.basename(model_info['file_path']),
                        'readonly': True,
                        'tip': 'Source file for the custom model'
                    },
                    {
                        'name': 'use_custom',
                        'type': 'bool',
                        'value': True,
                        'tip': 'Use this custom model instead of built-in model'
                    }
                ]
            }
            
            # Add the custom model parameters
            if PYQTGRAPH_AVAILABLE:
                self.addChild(Parameter.create(**custom_params))
            
        except Exception as e:
            print(f"Error adding custom model parameters: {e}")
    
    def update_model_selection(self, model_name, task_type):
        """Update the model parameters when model selection changes."""
        self.model_name = model_name
        self.task_type = task_type
        self._update_model_parameters()
    
    def get_model_config(self):
        """Get the current model configuration."""
        config = {}
        
        for child in self.children():
            if child.name() not in ['load_custom_model']:
                if hasattr(child, 'value'):
                    if child.hasChildren():
                        # Handle group parameters
                        group_config = {}
                        for grandchild in child.children():
                            if hasattr(grandchild, 'value'):
                                group_config[grandchild.name()] = grandchild.value()
                        config[child.name()] = group_config
                    else:
                        config[child.name()] = child.value()
        
        return config
    
    def set_model_config(self, config):
        """Set the model configuration."""
        try:
            for key, value in config.items():
                param = self.child(key)
                if param:
                    if isinstance(value, dict) and param.hasChildren():
                        # Handle group parameters
                        for subkey, subvalue in value.items():
                            subparam = param.child(subkey)
                            if subparam:
                                subparam.setValue(subvalue)
                    else:
                        param.setValue(value)
        except Exception as e:
            print(f"Error setting model config: {e}")
