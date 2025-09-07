"""
Enhanced custom model architectures for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

This module contains custom model architectures that can be dynamically loaded
into the ModelGardener application.
"""

import keras
from keras import layers
from typing import Union, List, Dict, Tuple, Any
from .utils import (
    TaskType, DataDimension, detect_data_dimension, infer_task_type,
    get_conv_layer, get_pooling_layer, create_task_specific_output
)



def create_adaptive_cnn(input_shape=(32, 32, 3), 
                       num_classes=10, 
                       task_type='classification',
                       data_dimension='2d',
                       dropout_rate=0.5, 
                       multi_output=False,
                       **kwargs):
    """
    Create an adaptive CNN model that supports 2D/3D data and multiple task types.
    
    Args:
        input_shape: Input tensor shape - (H, W, C) for 2D or (H, W, D, C) for 3D
        num_classes: Number of output classes
        task_type: Type of task ('classification', 'segmentation', 'object_detection')
        data_dimension: Data dimension ('2d' or '3d')
        dropout_rate: Dropout rate for regularization
        multi_output: Whether to create multi-output model
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    # Determine data dimension from input shape if not specified
    if data_dimension == 'auto':
        data_dim = DataDimension.THREE_D if len(input_shape) == 4 else DataDimension.TWO_D
    else:
        data_dim = DataDimension.THREE_D if data_dimension == '3d' else DataDimension.TWO_D
    
    # Determine task type
    if isinstance(task_type, str):
        if task_type == 'classification':
            task = TaskType.CLASSIFICATION
        elif task_type == 'segmentation':
            task = TaskType.SEGMENTATION
        elif task_type == 'object_detection':
            task = TaskType.OBJECT_DETECTION
        else:
            task = TaskType.CLASSIFICATION
    else:
        task = task_type
    
    inputs = keras.Input(shape=input_shape, name="main_input")
    
    # Feature extraction backbone
    x = build_feature_backbone(inputs, data_dim, task)
    
    # Task-specific head
    if task == TaskType.CLASSIFICATION:
        # Global pooling for classification
        x = get_pooling_layer(data_dim, None, 'global_avg')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer(s)
        if multi_output:
            main_output = layers.Dense(num_classes, activation='softmax', name='main_output')(x)
            aux_output = layers.Dense(num_classes // 2, activation='softmax', name='aux_output')(x)
            outputs = [main_output, aux_output]
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)
            
    elif task == TaskType.SEGMENTATION:
        # For segmentation, we need to upsample to original spatial resolution
        outputs = build_segmentation_head(x, num_classes, data_dim, input_shape)
        
    elif task == TaskType.OBJECT_DETECTION:
        # Object detection head with multiple outputs
        outputs = build_detection_head(x, num_classes, data_dim)
    
    model_name = f"{data_dimension}_{task_type}_{'multi' if multi_output else 'single'}_cnn"
    model = keras.Model(inputs, outputs, name=model_name)
    return model

def build_feature_backbone(inputs, data_dim: DataDimension, task: TaskType):
    """Build feature extraction backbone."""
    x = inputs
    
    # First block
    x = get_conv_layer(data_dim, 32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = get_conv_layer(data_dim, 32, (3, 3), activation='relu', padding='same')(x)
    x = get_pooling_layer(data_dim, (2, 2), 'max')(x)
    x = layers.Dropout(0.25)(x)
    
    # Second block
    x = get_conv_layer(data_dim, 64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = get_conv_layer(data_dim, 64, (3, 3), activation='relu', padding='same')(x)
    x = get_pooling_layer(data_dim, (2, 2), 'max')(x)
    x = layers.Dropout(0.25)(x)
    
    # Third block
    x = get_conv_layer(data_dim, 128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = get_conv_layer(data_dim, 128, (3, 3), activation='relu', padding='same')(x)
    
    # For segmentation, we don't pool the final layer
    if task != TaskType.SEGMENTATION:
        x = get_pooling_layer(data_dim, (2, 2), 'max')(x)
    x = layers.Dropout(0.25)(x)
    
    return x

def build_segmentation_head(features, num_classes: int, data_dim: DataDimension, input_shape: Tuple):
    """Build segmentation head with upsampling."""
    x = features
    
    # Decoder path with upsampling
    x = get_conv_layer(data_dim, 256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Upsample to match input spatial dimensions
    if data_dim == DataDimension.TWO_D:
        # For 2D segmentation
        x = layers.UpSampling2D((2, 2))(x)
        x = get_conv_layer(data_dim, 128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = get_conv_layer(data_dim, 64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = get_conv_layer(data_dim, 32, (3, 3), activation='relu', padding='same')(x)
        
        # Final segmentation output
        outputs = get_conv_layer(data_dim, num_classes, (1, 1), activation='softmax', name='segmentation_output')(x)
    else:
        # For 3D segmentation
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = get_conv_layer(data_dim, 128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = get_conv_layer(data_dim, 64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = get_conv_layer(data_dim, 32, (3, 3, 3), activation='relu', padding='same')(x)
        
        # Final segmentation output
        outputs = get_conv_layer(data_dim, num_classes, (1, 1, 1), activation='softmax', name='segmentation_output')(x)
    
    return outputs

def build_detection_head(features, num_classes: int, data_dim: DataDimension):
    """Build object detection head."""
    # Global average pooling to get feature vector
    x = get_pooling_layer(data_dim, None, 'global_avg')(features)
    
    # Shared features
    shared = layers.Dense(512, activation='relu')(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.5)(shared)
    
    # Number of anchors per location
    num_anchors = 9
    
    # Bounding box regression head
    bbox_head = layers.Dense(256, activation='relu', name='bbox_head')(shared)
    bbox_output = layers.Dense(num_anchors * 4, name='bbox_output')(bbox_head)
    
    # Classification head  
    class_head = layers.Dense(256, activation='relu', name='class_head')(shared)
    class_output = layers.Dense(num_anchors * num_classes, activation='sigmoid', name='class_output')(class_head)
    
    # Objectness head
    obj_head = layers.Dense(128, activation='relu', name='obj_head')(shared)
    obj_output = layers.Dense(num_anchors, activation='sigmoid', name='objectness_output')(obj_head)
    
    return [bbox_output, class_output, obj_output]



def create_multi_input_model(input_shapes=[(32, 32, 3), (32, 32, 1)], 
                           num_classes=10, 
                           task_type='classification',
                           data_dimension='2d',
                           fusion_method='concatenate',
                           **kwargs):
    """
    Create a multi-input model that can handle different types of input data.
    
    Args:
        input_shapes: List of input shapes for different input branches
        num_classes: Number of output classes
        task_type: Type of task ('classification', 'segmentation', 'object_detection')
        data_dimension: Data dimension ('2d' or '3d')
        fusion_method: Method to fuse features ('concatenate', 'add', 'multiply')
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: Multi-input model
    """
    data_dim = DataDimension.THREE_D if data_dimension == '3d' else DataDimension.TWO_D
    
    # Create input branches
    inputs = []
    feature_branches = []
    
    for i, input_shape in enumerate(input_shapes):
        input_tensor = keras.Input(shape=input_shape, name=f'input_{i+1}')
        inputs.append(input_tensor)
        
        # Build feature extraction for each input
        x = build_feature_backbone(input_tensor, data_dim, TaskType.CLASSIFICATION)
        
        # Global pooling for fusion
        x = get_pooling_layer(data_dim, None, 'global_avg')(x)
        feature_branches.append(x)
    
    # Fuse features from different inputs
    if len(feature_branches) > 1:
        if fusion_method == 'concatenate':
            fused_features = layers.Concatenate()(feature_branches)
        elif fusion_method == 'add':
            fused_features = layers.Add()(feature_branches)
        elif fusion_method == 'multiply':
            fused_features = layers.Multiply()(feature_branches)
        else:
            fused_features = layers.Concatenate()(feature_branches)  # Default
    else:
        fused_features = feature_branches[0]
    
    # Task-specific output head
    x = layers.Dense(512, activation='relu')(fused_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    if task_type == 'classification':
        outputs = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)
    elif task_type == 'regression':
        outputs = layers.Dense(1, name='regression_output')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs, outputs, name='multi_input_model')
    return model

def create_3d_cnn(input_shape=(32, 32, 32, 1), 
                 num_classes=10, 
                 task_type='classification',
                 dropout_rate=0.5,
                 **kwargs):
    """
    Create a 3D CNN model for volumetric data.
    
    Args:
        input_shape: Input tensor shape (height, width, depth, channels)
        num_classes: Number of output classes
        task_type: Type of task ('classification', 'segmentation')
        dropout_rate: Dropout rate for regularization
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: 3D CNN model
    """
    inputs = keras.Input(shape=input_shape, name='volume_input')
    
    # 3D Convolutional blocks
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    
    if task_type == 'classification':
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)
    elif task_type == 'segmentation':
        # For 3D segmentation, build decoder
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling3D((2, 2, 2))(x)
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax', name='segmentation_output')(x)
    else:
        # Default to classification
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs, outputs, name=f'3d_{task_type}_cnn')
    return model

# Backward compatibility - keep original functions
def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, **kwargs):
    """Original simple CNN for backward compatibility."""
    return create_adaptive_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        task_type='classification',
        data_dimension='2d',
        dropout_rate=dropout_rate,
        multi_output=False,
        **kwargs
    )

def create_simple_cnn_two_outputs(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, **kwargs):
    """Original multi-output CNN for backward compatibility."""
    return create_adaptive_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        task_type='classification',
        data_dimension='2d',
        dropout_rate=dropout_rate,
        multi_output=True,
        **kwargs
    )
