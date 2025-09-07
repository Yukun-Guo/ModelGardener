"""
Enhanced multi-output model examples for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)
"""

import tensorflow as tf
import keras
from keras import layers
from typing import Union, List, Dict, Tuple, Any
from .utils import (
    TaskType, DataDimension, detect_data_dimension, 
    get_conv_layer, get_pooling_layer, create_task_specific_output
)

def create_enhanced_multi_output_model(input_shape=(224, 224, 3), 
                                     output_configs=None,
                                     data_dimension='2d',
                                     shared_layers_config=None):
    """
    Create an enhanced multi-output CNN model with flexible output configuration.
    
    Args:
        input_shape: Input shape for the model
        output_configs: List of dicts defining each output 
                       [{'name': 'main', 'type': 'classification', 'classes': 10}, ...]
        data_dimension: '2d' or '3d'
        shared_layers_config: Configuration for shared feature extraction layers
        
    Returns:
        Multi-output Keras model
    """
    if output_configs is None:
        output_configs = [
            {'name': 'main_output', 'type': 'classification', 'classes': 10},
            {'name': 'auxiliary_output', 'type': 'classification', 'classes': 5}
        ]
    
    data_dim = DataDimension.THREE_D if data_dimension == '3d' else DataDimension.TWO_D
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Shared feature extraction layers
    x = build_shared_features(inputs, data_dim, shared_layers_config)
    
    # Create multiple outputs
    outputs = []
    output_names = []
    
    for output_config in output_configs:
        output_name = output_config['name']
        output_type = output_config['type']
        num_classes = output_config.get('classes', 10)
        
        # Create task-specific branch
        if output_type == 'classification':
            # Classification branch
            branch = layers.Dense(256, activation='relu', name=f"{output_name}_branch")(x)
            branch = layers.Dropout(0.3, name=f"{output_name}_dropout")(branch)
            output = layers.Dense(num_classes, activation='softmax', name=output_name)(branch)
        
        elif output_type == 'regression':
            # Regression branch
            branch = layers.Dense(128, activation='relu', name=f"{output_name}_branch")(x)
            branch = layers.Dropout(0.2, name=f"{output_name}_dropout")(branch)
            output = layers.Dense(1, name=output_name)(branch)
        
        elif output_type == 'segmentation':
            # Segmentation branch (simplified - would need proper decoder)
            branch = layers.Dense(512, activation='relu', name=f"{output_name}_branch")(x)
            # This is simplified - real segmentation would need upsampling layers
            output = layers.Dense(num_classes, activation='softmax', name=output_name)(branch)
        
        else:
            # Default to classification
            branch = layers.Dense(256, activation='relu', name=f"{output_name}_branch")(x)
            output = layers.Dense(num_classes, activation='softmax', name=output_name)(branch)
        
        outputs.append(output)
        output_names.append(output_name)
    
    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="enhanced_multi_output_model"
    )
    
    return model

def build_shared_features(inputs, data_dim: DataDimension, config=None):
    """Build shared feature extraction layers."""
    if config is None:
        config = {
            'conv_blocks': [
                {'filters': 64, 'layers': 2},
                {'filters': 128, 'layers': 2},
                {'filters': 256, 'layers': 2}
            ],
            'pooling': True,
            'dropout': 0.25
        }
    
    x = inputs
    
    for i, block_config in enumerate(config['conv_blocks']):
        filters = block_config['filters']
        layers_count = block_config['layers']
        
        # Convolutional layers in this block
        for j in range(layers_count):
            x = get_conv_layer(data_dim, filters, (3, 3), activation='relu', padding='same')(x)
            if j == 0:  # Add batch norm after first conv in block
                x = layers.BatchNormalization()(x)
        
        # Pooling and dropout
        if config['pooling']:
            x = get_pooling_layer(data_dim, (2, 2), 'max')(x)
        
        if config['dropout'] > 0:
            x = layers.Dropout(config['dropout'])(x)
    
    # Global pooling for dense layers
    x = get_pooling_layer(data_dim, None, 'global_avg')(x)
    
    # Shared dense features
    x = layers.Dense(512, activation='relu', name="shared_features")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    return x

def create_multi_task_model(input_shape=(224, 224, 3),
                          classification_classes=10,
                          segmentation_classes=2,
                          enable_detection=False,
                          data_dimension='2d'):
    """
    Create a multi-task model that can handle classification, segmentation, and detection.
    
    Args:
        input_shape: Input shape for the model
        classification_classes: Number of classification classes
        segmentation_classes: Number of segmentation classes
        enable_detection: Whether to enable object detection output
        data_dimension: '2d' or '3d'
        
    Returns:
        Multi-task Keras model
    """
    data_dim = DataDimension.THREE_D if data_dimension == '3d' else DataDimension.TWO_D
    
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Shared backbone
    backbone = build_shared_features(inputs, data_dim)
    
    outputs = []
    
    # Classification head
    cls_branch = layers.Dense(256, activation='relu', name="classification_branch")(backbone)
    cls_branch = layers.Dropout(0.3)(cls_branch)
    classification_output = layers.Dense(classification_classes, activation='softmax', 
                                       name="classification_output")(cls_branch)
    outputs.append(classification_output)
    
    # Segmentation head (simplified)
    seg_branch = layers.Dense(512, activation='relu', name="segmentation_branch")(backbone)
    # In practice, this would be a proper decoder with upsampling
    segmentation_output = layers.Dense(segmentation_classes, activation='softmax',
                                     name="segmentation_output")(seg_branch)
    outputs.append(segmentation_output)
    
    # Object detection head (if enabled)
    if enable_detection:
        det_branch = layers.Dense(256, activation='relu', name="detection_branch")(backbone)
        
        # Bounding box regression
        bbox_output = layers.Dense(4, name="bbox_output")(det_branch)
        
        # Object classification
        obj_cls_output = layers.Dense(classification_classes, activation='softmax',
                                    name="object_classification_output")(det_branch)
        
        # Objectness score
        objectness_output = layers.Dense(1, activation='sigmoid',
                                       name="objectness_output")(det_branch)
        
        outputs.extend([bbox_output, obj_cls_output, objectness_output])
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="multi_task_model")
    return model

def create_multi_input_multi_output_model(input_shapes=[(224, 224, 3), (224, 224, 1)],
                                        output_configs=None,
                                        fusion_method='concatenate',
                                        data_dimension='2d'):
    """
    Create a model with multiple inputs and multiple outputs.
    
    Args:
        input_shapes: List of input shapes
        output_configs: Configuration for outputs
        fusion_method: How to fuse multiple inputs ('concatenate', 'add', 'attention')
        data_dimension: '2d' or '3d'
        
    Returns:
        Multi-input multi-output Keras model
    """
    if output_configs is None:
        output_configs = [
            {'name': 'main_output', 'type': 'classification', 'classes': 10},
            {'name': 'aux_output', 'type': 'regression', 'classes': 1}
        ]
    
    data_dim = DataDimension.THREE_D if data_dimension == '3d' else DataDimension.TWO_D
    
    # Create multiple inputs
    inputs = []
    feature_branches = []
    
    for i, input_shape in enumerate(input_shapes):
        input_tensor = keras.Input(shape=input_shape, name=f'input_{i+1}')
        inputs.append(input_tensor)
        
        # Extract features from each input
        features = build_input_branch(input_tensor, data_dim, f"branch_{i+1}")
        feature_branches.append(features)
    
    # Fuse features from multiple inputs
    if len(feature_branches) > 1:
        if fusion_method == 'concatenate':
            fused_features = layers.Concatenate(name="feature_fusion")(feature_branches)
        elif fusion_method == 'add':
            # Ensure same dimensions before adding
            fused_features = layers.Add(name="feature_fusion")(feature_branches)
        elif fusion_method == 'attention':
            # Simple attention mechanism
            attention_weights = []
            for i, features in enumerate(feature_branches):
                attention = layers.Dense(1, activation='sigmoid', name=f"attention_{i}")(features)
                attention_weights.append(attention)
            
            # Normalize attention weights
            attention_sum = layers.Add()(attention_weights)
            normalized_weights = [layers.Lambda(lambda x: x[0] / (x[1] + 1e-8))([w, attention_sum]) 
                                for w in attention_weights]
            
            # Apply attention weights
            weighted_features = [layers.Multiply()([features, weights]) 
                               for features, weights in zip(feature_branches, normalized_weights)]
            fused_features = layers.Add(name="feature_fusion")(weighted_features)
        else:
            fused_features = layers.Concatenate(name="feature_fusion")(feature_branches)
    else:
        fused_features = feature_branches[0]
    
    # Create multiple outputs from fused features
    outputs = []
    for output_config in output_configs:
        output_name = output_config['name']
        output_type = output_config['type']
        num_classes = output_config.get('classes', 10)
        
        # Create output-specific branch
        branch = layers.Dense(256, activation='relu', name=f"{output_name}_dense")(fused_features)
        branch = layers.Dropout(0.3)(branch)
        
        if output_type == 'classification':
            output = layers.Dense(num_classes, activation='softmax', name=output_name)(branch)
        elif output_type == 'regression':
            output = layers.Dense(1, name=output_name)(branch)
        else:
            output = layers.Dense(num_classes, activation='softmax', name=output_name)(branch)
        
        outputs.append(output)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="multi_input_multi_output_model")
    return model

def build_input_branch(input_tensor, data_dim: DataDimension, branch_name: str):
    """Build feature extraction branch for a single input."""
    x = input_tensor
    
    # Convolutional feature extraction
    x = get_conv_layer(data_dim, 64, (3, 3), activation='relu', padding='same', 
                      name=f"{branch_name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn1")(x)
    x = get_pooling_layer(data_dim, (2, 2), 'max', name=f"{branch_name}_pool1")(x)
    
    x = get_conv_layer(data_dim, 128, (3, 3), activation='relu', padding='same',
                      name=f"{branch_name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn2")(x)
    x = get_pooling_layer(data_dim, (2, 2), 'max', name=f"{branch_name}_pool2")(x)
    
    x = get_conv_layer(data_dim, 256, (3, 3), activation='relu', padding='same',
                      name=f"{branch_name}_conv3")(x)
    x = layers.BatchNormalization(name=f"{branch_name}_bn3")(x)
    
    # Global pooling
    x = get_pooling_layer(data_dim, None, 'global_avg', name=f"{branch_name}_global_pool")(x)
    
    # Dense features
    x = layers.Dense(512, activation='relu', name=f"{branch_name}_dense")(x)
    x = layers.Dropout(0.5, name=f"{branch_name}_dropout")(x)
    
    return x

# Backward compatibility functions
def create_multi_output_model(input_shape=(224, 224, 3), num_classes_main=10, num_classes_aux=5):
    """
    Original multi-output model for backward compatibility.
    
    Args:
        input_shape: Input shape for the model
        num_classes_main: Number of classes for main output
        num_classes_aux: Number of classes for auxiliary output
        
    Returns:
        Multi-output Keras model
    """
    output_configs = [
        {'name': 'main_output', 'type': 'classification', 'classes': num_classes_main},
        {'name': 'auxiliary_output', 'type': 'classification', 'classes': num_classes_aux}
    ]
    
    return create_enhanced_multi_output_model(
        input_shape=input_shape,
        output_configs=output_configs,
        data_dimension='2d'
    )

def create_simple_multi_output_classifier(input_shape=(224, 224, 3), classes=10):
    """
    Create a simple multi-output classifier for testing.
    
    Args:
        input_shape: Input image shape
        classes: Number of classes
        
    Returns:
        Multi-output model with main and auxiliary outputs
    """
    return create_multi_output_model(
        input_shape=input_shape, 
        num_classes_main=classes,
        num_classes_aux=classes // 2  # Auxiliary output has half the classes
    )

if __name__ == "__main__":
    # Test the model creation
    model = create_enhanced_multi_output_model()
    model.summary()
    
    print(f"\nModel has {len(model.outputs)} outputs:")
    for i, output in enumerate(model.outputs):
        print(f"  Output {i+1}: {output.name} - shape: {output.shape}")
