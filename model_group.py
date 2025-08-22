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
        self.model_name = opts.get('model_name', 'ResNet-50')
        self.task_type = opts.get('task_type', 'image_classification')
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
                    
                # Don't emit tree state change signal manually - let pyqtgraph handle it naturally
                # The parameter changes will automatically trigger the appropriate signals
                
        except Exception as e:
            print(f"Error updating model parameters: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_model_parameters(self, model_name, task_type):
        """Get model-specific parameters based on actual keras.applications and other model implementations."""
        params = {}
        
        # Get the model family from the model name
        model_family = self._get_model_family_from_name(model_name)
        
        if task_type == 'image_classification':
            params.update(self._get_classification_parameters(model_name, model_family))
        elif task_type == 'object_detection':
            params.update(self._get_detection_parameters(model_name, model_family))
        elif task_type == 'semantic_segmentation':
            params.update(self._get_segmentation_parameters(model_name, model_family))
        else:
            # Default parameters for other tasks
            params.update(self._get_default_parameters())
            
        return params
    
    def _get_model_family_from_name(self, model_name):
        """Extract model family from model name."""
        model_name_lower = model_name.lower()
        
        # Classification families
        if 'resnet' in model_name_lower:
            return 'resnet'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        elif 'mobilenet' in model_name_lower:
            return 'mobilenet'
        elif 'vit' in model_name_lower or 'vision' in model_name_lower:
            return 'vision_transformer'
        elif 'densenet' in model_name_lower:
            return 'densenet'
        elif 'vgg' in model_name_lower:
            return 'vgg'
        elif 'inception' in model_name_lower:
            return 'inception'
        elif 'xception' in model_name_lower:
            return 'xception'
        elif 'convnext' in model_name_lower:
            return 'convnext'
        elif 'regnet' in model_name_lower:
            return 'regnet'
        
        # Detection families
        elif 'yolo' in model_name_lower:
            return 'yolo'
        elif 'faster' in model_name_lower and 'rcnn' in model_name_lower:
            return 'faster_rcnn'
        elif 'ssd' in model_name_lower:
            return 'ssd'
        elif 'retinanet' in model_name_lower:
            return 'retinanet'
        elif 'efficientdet' in model_name_lower:
            return 'efficientdet'
            
        # Segmentation families
        elif 'unet' in model_name_lower or 'u-net' in model_name_lower:
            return 'unet'
        elif 'deeplab' in model_name_lower:
            return 'deeplabv3'
        elif 'pspnet' in model_name_lower:
            return 'pspnet'
        elif 'fcn' in model_name_lower:
            return 'fcn'
        elif 'segnet' in model_name_lower:
            return 'segnet'
            
        return 'unknown'
    
    def _get_classification_parameters(self, model_name, model_family):
        """Get parameters for image classification models based on keras.applications."""
        # Common parameters for all classification models (based on keras.applications inspection)
        params = {
            'input_shape': {
                'type': 'group',
                'children': [
                    {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image height'},
                    {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image width'},
                    {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4], 'tip': 'Number of input channels'}
                ]
            },
            'include_top': {
                'type': 'bool',
                'value': True,
                'tip': 'Whether to include the fully-connected layer at the top of the network'
            },
            'weights': {
                'type': 'list',
                'value': 'imagenet',
                'values': ['imagenet', 'None'],
                'tip': 'Pre-trained weights to load'
            },
            'pooling': {
                'type': 'list',
                'value': 'None',
                'values': ['None', 'avg', 'max'],
                'tip': 'Pooling mode for feature extraction when include_top is False'
            },
            'classes': {
                'type': 'int',
                'value': 1000,
                'limits': [1, 100000],
                'tip': 'Number of classes for classification'
            },
            'classifier_activation': {
                'type': 'list',
                'value': 'softmax',
                'values': ['softmax', 'sigmoid', 'linear', 'None'],
                'tip': 'Activation function for the classification layer'
            }
        }
        
        # Add model family specific parameters
        if model_family == 'mobilenet':
            params.update(self._get_mobilenet_specific_params(model_name))
        elif model_family == 'efficientnet':
            params.update(self._get_efficientnet_specific_params())
        elif model_family == 'vision_transformer':
            params.update(self._get_vit_specific_params())
        elif model_family == 'convnext':
            params.update(self._get_convnext_specific_params())
        elif model_family == 'regnet':
            params.update(self._get_regnet_specific_params())
            
        return params
    
    def _get_mobilenet_specific_params(self, model_name):
        """Get MobileNet-specific parameters."""
        params = {
            'alpha': {
                'type': 'list',
                'value': 1.0,
                'values': [0.25, 0.35, 0.5, 0.75, 1.0, 1.3, 1.4],
                'tip': 'Width multiplier for the model'
            }
        }
        
        if 'v1' in model_name.lower() or 'MobileNet' == model_name:
            params.update({
                'depth_multiplier': {
                    'type': 'list',
                    'value': 1,
                    'values': [1, 2, 3, 4],
                    'tip': 'Depth multiplier for depthwise convolution'
                },
                'dropout': {
                    'type': 'float',
                    'value': 0.001,
                    'limits': [0.0, 0.9],
                    'step': 0.001,
                    'tip': 'Dropout rate'
                }
            })
        elif 'v3' in model_name.lower():
            params.update({
                'minimalistic': {
                    'type': 'bool',
                    'value': False,
                    'tip': 'Use minimalistic version of the model'
                },
                'dropout_rate': {
                    'type': 'float',
                    'value': 0.2,
                    'limits': [0.0, 0.9],
                    'step': 0.05,
                    'tip': 'Dropout rate'
                },
                'include_preprocessing': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Whether to include preprocessing in the model'
                }
            })
        
        return params
    
    def _get_efficientnet_specific_params(self):
        """Get EfficientNet-specific parameters."""
        return {
            'drop_connect_rate': {
                'type': 'float',
                'value': 0.2,
                'limits': [0.0, 0.8],
                'step': 0.05,
                'tip': 'Drop connect rate for stochastic depth'
            }
        }
    
    def _get_vit_specific_params(self):
        """Get Vision Transformer specific parameters."""
        return {
            'patch_size': {
                'type': 'list',
                'value': 16,
                'values': [8, 16, 32],
                'tip': 'Size of image patches'
            },
            'num_layers': {
                'type': 'list',
                'value': 12,
                'values': [6, 12, 24],
                'tip': 'Number of transformer layers'
            },
            'num_heads': {
                'type': 'list',
                'value': 12,
                'values': [4, 8, 12, 16],
                'tip': 'Number of attention heads'
            },
            'hidden_size': {
                'type': 'list',
                'value': 768,
                'values': [256, 384, 768, 1024],
                'tip': 'Hidden size of transformer'
            },
            'mlp_dim': {
                'type': 'list',
                'value': 3072,
                'values': [1024, 3072, 4096],
                'tip': 'MLP hidden dimension'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.1,
                'limits': [0.0, 0.5],
                'step': 0.05,
                'tip': 'Dropout rate'
            },
            'attention_dropout_rate': {
                'type': 'float',
                'value': 0.0,
                'limits': [0.0, 0.3],
                'step': 0.05,
                'tip': 'Attention dropout rate'
            },
            'representation_size': {
                'type': 'list',
                'value': 0,
                'values': [0, 768, 1024],
                'tip': 'Size of representation layer (0 to disable)'
            }
        }
    
    def _get_convnext_specific_params(self):
        """Get ConvNeXt specific parameters."""
        return {
            'drop_path_rate': {
                'type': 'float',
                'value': 0.0,
                'limits': [0.0, 0.5],
                'step': 0.05,
                'tip': 'Drop path rate for stochastic depth'
            }
        }
    
    def _get_regnet_specific_params(self):
        """Get RegNet specific parameters."""
        return {
            'width_coefficient': {
                'type': 'list',
                'value': 1.0,
                'values': [0.5, 1.0, 1.5, 2.0],
                'tip': 'Width scaling coefficient'
            },
            'depth_coefficient': {
                'type': 'list',
                'value': 1.0,
                'values': [0.5, 1.0, 1.5, 2.0],
                'tip': 'Depth scaling coefficient'
            }
        }
    
    def _get_detection_parameters(self, model_name, model_family):
        """Get parameters for object detection models."""
        # Common detection parameters
        params = {
            'input_size': {
                'type': 'group',
                'children': [
                    {'name': 'height', 'type': 'int', 'value': 416, 'limits': [320, 1024], 'tip': 'Input image height'},
                    {'name': 'width', 'type': 'int', 'value': 416, 'limits': [320, 1024], 'tip': 'Input image width'},
                    {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4], 'tip': 'Number of input channels'}
                ]
            },
            'num_classes': {
                'type': 'int',
                'value': 80,
                'limits': [1, 10000],
                'tip': 'Number of object classes'
            },
            'iou_threshold': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.1, 0.9],
                'step': 0.05,
                'tip': 'IoU threshold for non-maximum suppression'
            },
            'confidence_threshold': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.1, 0.9],
                'step': 0.05,
                'tip': 'Confidence threshold for detections'
            }
        }
        
        # Add family-specific parameters
        if model_family == 'yolo':
            params.update(self._get_yolo_specific_params(model_name))
        elif model_family in ['faster_rcnn', 'ssd', 'retinanet']:
            params.update(self._get_rcnn_ssd_specific_params(model_family))
        elif model_family == 'efficientdet':
            params.update(self._get_efficientdet_specific_params())
            
        return params
    
    def _get_yolo_specific_params(self, model_name):
        """Get YOLO-specific parameters."""
        params = {
            'anchors': {
                'type': 'str',
                'value': '10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326',
                'tip': 'Anchor boxes (comma-separated)'
            },
            'anchor_masks': {
                'type': 'str', 
                'value': '0,1,2,3,4,5,6,7,8',
                'tip': 'Anchor masks for different scales'
            },
            'max_boxes_per_class': {
                'type': 'int',
                'value': 20,
                'limits': [1, 100],
                'tip': 'Maximum boxes per class'
            },
            'max_total_size': {
                'type': 'int',
                'value': 100,
                'limits': [10, 1000],
                'tip': 'Maximum total detections'
            }
        }
        
        # Version-specific parameters
        if 'v8' in model_name.lower():
            params.update({
                'use_ultralytics_format': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Use Ultralytics YOLOv8 format'
                }
            })
        elif 'v5' in model_name.lower():
            params.update({
                'focus_layer': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Use Focus layer in the model'
                }
            })
        elif 'v4' in model_name.lower():
            params.update({
                'use_csp': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Use CSP (Cross Stage Partial) connections'
                }
            })
        
        return params
    
    def _get_rcnn_ssd_specific_params(self, model_family):
        """Get Faster R-CNN, SSD, RetinaNet specific parameters."""
        params = {
            'backbone': {
                'type': 'list',
                'value': 'resnet50',
                'values': ['resnet50', 'resnet101', 'mobilenet_v2'],
                'tip': 'Backbone network architecture'
            },
            'aspect_ratios': {
                'type': 'str',
                'value': '0.5,1.0,2.0',
                'tip': 'Anchor aspect ratios (comma-separated)'
            }
        }
        
        if model_family == 'faster_rcnn':
            params.update({
                'include_mask': {
                    'type': 'bool',
                    'value': False,
                    'tip': 'Include mask prediction (Mask R-CNN)'
                },
                'anchor_scale': {
                    'type': 'float',
                    'value': 8.0,
                    'limits': [2.0, 16.0],
                    'step': 0.5,
                    'tip': 'Anchor scale factor'
                }
            })
        elif model_family == 'retinanet':
            params.update({
                'focal_loss_alpha': {
                    'type': 'float',
                    'value': 0.25,
                    'limits': [0.1, 0.5],
                    'step': 0.05,
                    'tip': 'Focal loss alpha parameter'
                },
                'focal_loss_gamma': {
                    'type': 'float',
                    'value': 2.0,
                    'limits': [0.5, 5.0],
                    'step': 0.1,
                    'tip': 'Focal loss gamma parameter'
                }
            })
        elif model_family == 'ssd':
            params.update({
                'scales': {
                    'type': 'str',
                    'value': '0.2,0.34,0.48,0.62,0.76,0.9',
                    'tip': 'Multi-scale anchor sizes'
                },
                'clip_boxes': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Clip bounding boxes to image boundaries'
                }
            })
        
        return params
    
    def _get_efficientdet_specific_params(self):
        """Get EfficientDet-specific parameters."""
        return {
            'model_name': {
                'type': 'list',
                'value': 'efficientdet-d0',
                'limits': [f'efficientdet-d{i}' for i in range(8)],
                'tip': 'EfficientDet variant'
            },
            'mixed_precision': {
                'type': 'bool',
                'value': False,
                'tip': 'Use mixed precision training'
            }
        }
    
    def _get_segmentation_parameters(self, model_name, model_family):
        """Get parameters for semantic segmentation models."""
        # Common segmentation parameters
        params = {
            'input_shape': {
                'type': 'group',
                'children': [
                    {'name': 'height', 'type': 'int', 'value': 256, 'limits': [128, 1024], 'tip': 'Input image height'},
                    {'name': 'width', 'type': 'int', 'value': 256, 'limits': [128, 1024], 'tip': 'Input image width'},
                    {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4], 'tip': 'Number of input channels'}
                ]
            },
            'num_classes': {
                'type': 'int',
                'value': 21,
                'limits': [1, 1000],
                'tip': 'Number of segmentation classes'
            },
            'activation': {
                'type': 'list',
                'value': 'softmax',
                'values': ['softmax', 'sigmoid', 'relu'],
                'tip': 'Output activation function'
            }
        }
        
        # Add family-specific parameters
        if model_family == 'unet':
            params.update(self._get_unet_specific_params())
        elif model_family == 'deeplabv3':
            params.update(self._get_deeplab_specific_params())
        elif model_family == 'pspnet':
            params.update(self._get_pspnet_specific_params())
        elif model_family in ['fcn', 'segnet']:
            params.update(self._get_fcn_segnet_specific_params(model_family))
            
        return params
    
    def _get_unet_specific_params(self):
        """Get U-Net specific parameters."""
        return {
            'filters': {
                'type': 'list',
                'value': 64,
                'values': [16, 32, 64, 128, 256],
                'tip': 'Base number of filters'
            },
            'num_layers': {
                'type': 'list',
                'value': 4,
                'values': [3, 4, 5, 6],
                'tip': 'Number of encoder/decoder layers'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Dropout rate'
            },
            'batch_normalization': {
                'type': 'bool',
                'value': True,
                'tip': 'Use batch normalization'
            },
            'use_attention': {
                'type': 'bool',
                'value': False,
                'tip': 'Use attention gates'
            },
            'deep_supervision': {
                'type': 'bool',
                'value': False,
                'tip': 'Use deep supervision'
            }
        }
    
    def _get_deeplab_specific_params(self):
        """Get DeepLabV3/V3+ specific parameters."""
        return {
            'backbone': {
                'type': 'list',
                'value': 'resnet50',
                'values': ['resnet50', 'resnet101', 'mobilenet_v2', 'xception'],
                'tip': 'Backbone network'
            },
            'output_stride': {
                'type': 'list',
                'value': 16,
                'values': [8, 16, 32],
                'tip': 'Output stride for the backbone'
            },
            'atrous_rates': {
                'type': 'str',
                'value': '6,12,18',
                'tip': 'ASPP atrous rates (comma-separated)'
            },
            'aspp_dropout': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'ASPP dropout rate'
            },
            'decoder_channels': {
                'type': 'list',
                'value': 256,
                'values': [64, 128, 256, 512],
                'tip': 'Number of decoder channels'
            }
        }
    
    def _get_pspnet_specific_params(self):
        """Get PSPNet specific parameters."""
        return {
            'backbone': {
                'type': 'list',
                'value': 'resnet50',
                'values': ['resnet50', 'resnet101', 'mobilenet_v2'],
                'tip': 'Backbone network'
            },
            'pyramid_bins': {
                'type': 'str',
                'value': '1,2,3,6',
                'tip': 'Pyramid pooling bins (comma-separated)'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.1,
                'limits': [0.0, 0.5],
                'step': 0.05,
                'tip': 'Dropout rate'
            },
            'aux_loss': {
                'type': 'bool',
                'value': True,
                'tip': 'Use auxiliary loss'
            }
        }
    
    def _get_fcn_segnet_specific_params(self, model_family):
        """Get FCN/SegNet specific parameters."""
        params = {
            'backbone': {
                'type': 'list',
                'value': 'vgg16' if model_family == 'fcn' else 'vgg16',
                'values': ['vgg16', 'resnet50', 'resnet101'] if model_family == 'fcn' else ['vgg16', 'resnet'],
                'tip': 'Backbone network'
            },
            'dropout_rate': {
                'type': 'float',
                'value': 0.5,
                'limits': [0.0, 0.9],
                'step': 0.05,
                'tip': 'Dropout rate'
            }
        }
        
        if model_family == 'fcn':
            params.update({
                'skip_connections': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Use skip connections'
                }
            })
        elif model_family == 'segnet':
            params.update({
                'decoder_use_batchnorm': {
                    'type': 'bool',
                    'value': True,
                    'tip': 'Use batch normalization in decoder'
                }
            })
        
        return params
    
    def _get_default_parameters(self):
        """Get default parameters for unknown models."""
        return {
            'input_shape': {
                'type': 'group',
                'children': [
                    {'name': 'height', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image height'},
                    {'name': 'width', 'type': 'int', 'value': 224, 'limits': [32, 1024], 'tip': 'Input image width'},
                    {'name': 'channels', 'type': 'int', 'value': 3, 'limits': [1, 4], 'tip': 'Number of input channels'}
                ]
            },
            'num_classes': {
                'type': 'int',
                'value': 1000,
                'limits': [1, 50000],
                'tip': 'Number of output classes'
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
        try:
            # Check if anything actually changed
            if self.model_name != model_name or self.task_type != task_type:
                old_model = self.model_name
                old_task = self.task_type
                
                self.model_name = model_name
                self.task_type = task_type
                
                print(f"ModelGroup: Updating from {old_model} ({old_task}) to {model_name} ({task_type})")
                self._update_model_parameters()
                print(f"ModelGroup: Updated successfully, now has {len(self.children())} parameters")
            else:
                print(f"ModelGroup: No change needed - already {model_name} ({task_type})")
        except Exception as e:
            print(f"Error in update_model_selection: {e}")
            import traceback
            traceback.print_exc()
    
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
