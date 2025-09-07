"""
Utility functions for enhanced ModelGardener example functions.

This module provides common utility functions used across all example functions
to support multi-inputs/multi-outputs, 3D data, and different task types.
"""

import tensorflow as tf
import numpy as np
from typing import Union, Tuple, List, Dict, Any
from enum import Enum

class TaskType(Enum):
    """Enumeration of supported task types."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    OBJECT_DETECTION = "object_detection"

class DataDimension(Enum):
    """Enumeration of data dimensions."""
    TWO_D = "2d"
    THREE_D = "3d"

def detect_data_dimension(data: tf.Tensor) -> DataDimension:
    """
    Detect if data is 2D or 3D based on tensor shape.
    
    Args:
        data: Input tensor
        
    Returns:
        DataDimension: 2D or 3D
    """
    shape = tf.shape(data)
    rank = len(data.shape)
    
    if rank == 3:  # (H, W, C)
        return DataDimension.TWO_D
    elif rank == 4:  # (H, W, D, C) or (B, H, W, C)
        # Check if it's batched 2D or unbatched 3D
        # This is a heuristic - in practice, context would be needed
        if data.shape[-1] <= 16:  # Likely channels
            return DataDimension.THREE_D
        else:
            return DataDimension.TWO_D
    elif rank == 5:  # (B, H, W, D, C)
        return DataDimension.THREE_D
    else:
        return DataDimension.TWO_D  # Default fallback

def infer_task_type(label_shape: Tuple, label_data: tf.Tensor = None) -> TaskType:
    """
    Infer task type based on label characteristics.
    
    Args:
        label_shape: Shape of the label tensor
        label_data: Optional label data for additional inference
        
    Returns:
        TaskType: Inferred task type
    """
    if len(label_shape) == 0 or (len(label_shape) == 1 and label_shape[0] == 1):
        # Scalar or single value - likely classification
        return TaskType.CLASSIFICATION
    elif len(label_shape) == 1:
        # Vector - could be one-hot classification or bounding box
        if label_shape[0] <= 1000:  # Arbitrary threshold for classes
            return TaskType.CLASSIFICATION
        else:
            return TaskType.OBJECT_DETECTION
    elif len(label_shape) >= 2:
        # Multi-dimensional - likely segmentation mask
        return TaskType.SEGMENTATION
    else:
        return TaskType.CLASSIFICATION  # Default fallback

def handle_multi_input(data: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], 
                      func: callable, 
                      *args, **kwargs) -> Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Apply a function to multi-input data structure.
    
    Args:
        data: Input data (tensor, list of tensors, or dict of tensors)
        func: Function to apply to each tensor
        *args, **kwargs: Additional arguments for the function
        
    Returns:
        Processed data in the same structure as input
    """
    if isinstance(data, tf.Tensor):
        return func(data, *args, **kwargs)
    elif isinstance(data, list):
        return [func(item, *args, **kwargs) for item in data]
    elif isinstance(data, dict):
        return {key: func(value, *args, **kwargs) for key, value in data.items()}
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def handle_multi_output(labels: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]], 
                       func: callable, 
                       *args, **kwargs) -> Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Apply a function to multi-output label structure.
    
    Args:
        labels: Label data (tensor, list of tensors, or dict of tensors)
        func: Function to apply to each tensor
        *args, **kwargs: Additional arguments for the function
        
    Returns:
        Processed labels in the same structure as input
    """
    return handle_multi_input(labels, func, *args, **kwargs)

def get_spatial_dimensions(data: tf.Tensor, data_dim: DataDimension) -> Tuple[int, ...]:
    """
    Get spatial dimensions of data tensor.
    
    Args:
        data: Input tensor
        data_dim: Data dimension (2D or 3D)
        
    Returns:
        Tuple of spatial dimensions
    """
    shape = tf.shape(data)
    
    if data_dim == DataDimension.TWO_D:
        if len(data.shape) == 3:  # (H, W, C)
            return (shape[0], shape[1])
        elif len(data.shape) == 4:  # (B, H, W, C)
            return (shape[1], shape[2])
    else:  # 3D
        if len(data.shape) == 4:  # (H, W, D, C)
            return (shape[0], shape[1], shape[2])
        elif len(data.shape) == 5:  # (B, H, W, D, C)
            return (shape[1], shape[2], shape[3])
    
    raise ValueError(f"Cannot determine spatial dimensions for shape {data.shape} and dimension {data_dim}")

def apply_2d_operation_to_3d(data: tf.Tensor, operation: callable, axis: int = -2) -> tf.Tensor:
    """
    Apply a 2D operation slice-by-slice to 3D data.
    
    Args:
        data: 3D input tensor
        operation: 2D operation to apply
        axis: Axis along which to apply the operation (default: depth axis)
        
    Returns:
        Processed 3D tensor
    """
    if axis == -2:  # Depth axis for (H, W, D, C)
        depth = tf.shape(data)[2]
        processed_slices = []
        
        for i in range(depth):
            slice_2d = data[:, :, i, :]  # Extract 2D slice
            processed_slice = operation(slice_2d)
            processed_slices.append(processed_slice)
        
        return tf.stack(processed_slices, axis=2)
    else:
        raise NotImplementedError(f"Axis {axis} not implemented for 3D operations")

def normalize_data_format(data: tf.Tensor, target_dim: DataDimension) -> tf.Tensor:
    """
    Normalize data to consistent format based on target dimension.
    
    Args:
        data: Input tensor
        target_dim: Target dimension format
        
    Returns:
        Normalized tensor
    """
    current_dim = detect_data_dimension(data)
    
    if current_dim == target_dim:
        return data
    
    # Add dimension conversion logic if needed
    # For now, return as-is
    return data

def get_conv_layer(data_dim: DataDimension, filters: int, kernel_size: Union[int, Tuple], **kwargs):
    """
    Get appropriate convolution layer based on data dimension.
    
    Args:
        data_dim: Data dimension (2D or 3D)
        filters: Number of filters
        kernel_size: Kernel size
        **kwargs: Additional layer arguments
        
    Returns:
        TensorFlow layer
    """
    if data_dim == DataDimension.TWO_D:
        return tf.keras.layers.Conv2D(filters, kernel_size, **kwargs)
    else:
        return tf.keras.layers.Conv3D(filters, kernel_size, **kwargs)

def get_pooling_layer(data_dim: DataDimension, pool_size: Union[int, Tuple], pool_type: str = "max", **kwargs):
    """
    Get appropriate pooling layer based on data dimension.
    
    Args:
        data_dim: Data dimension (2D or 3D)
        pool_size: Pooling size
        pool_type: Type of pooling ("max", "avg", "global_max", "global_avg")
        **kwargs: Additional layer arguments
        
    Returns:
        TensorFlow layer
    """
    if data_dim == DataDimension.TWO_D:
        if pool_type == "max":
            return tf.keras.layers.MaxPooling2D(pool_size, **kwargs)
        elif pool_type == "avg":
            return tf.keras.layers.AveragePooling2D(pool_size, **kwargs)
        elif pool_type == "global_max":
            return tf.keras.layers.GlobalMaxPooling2D(**kwargs)
        elif pool_type == "global_avg":
            return tf.keras.layers.GlobalAveragePooling2D(**kwargs)
    else:
        if pool_type == "max":
            return tf.keras.layers.MaxPooling3D(pool_size, **kwargs)
        elif pool_type == "avg":
            return tf.keras.layers.AveragePooling3D(pool_size, **kwargs)
        elif pool_type == "global_max":
            return tf.keras.layers.GlobalMaxPooling3D(**kwargs)
        elif pool_type == "global_avg":
            return tf.keras.layers.GlobalAveragePooling3D(**kwargs)
    
    raise ValueError(f"Unsupported pool_type: {pool_type} for dimension: {data_dim}")

def create_task_specific_output(features: tf.Tensor, 
                              task_type: TaskType, 
                              num_classes: int = None,
                              spatial_shape: Tuple = None,
                              name_prefix: str = "") -> tf.Tensor:
    """
    Create task-specific output layer.
    
    Args:
        features: Input features
        task_type: Type of task
        num_classes: Number of classes (for classification/segmentation)
        spatial_shape: Spatial shape for segmentation output
        name_prefix: Prefix for layer names
        
    Returns:
        Task-specific output tensor
    """
    if task_type == TaskType.CLASSIFICATION:
        if num_classes is None:
            raise ValueError("num_classes required for classification task")
        return tf.keras.layers.Dense(num_classes, activation='softmax', 
                                   name=f"{name_prefix}classification_output")(features)
    
    elif task_type == TaskType.SEGMENTATION:
        if num_classes is None or spatial_shape is None:
            raise ValueError("num_classes and spatial_shape required for segmentation task")
        
        # For segmentation, we need to upsample features to match spatial shape
        # This is a simplified approach - in practice, you'd use proper decoder architecture
        x = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax', 
                                 name=f"{name_prefix}segmentation_output")(features)
        return x
    
    elif task_type == TaskType.OBJECT_DETECTION:
        # Object detection typically has multiple outputs (boxes, classes, scores)
        # This is a simplified version
        num_anchors = 9  # Example number of anchors
        
        # Bounding box regression (4 coordinates per anchor)
        boxes = tf.keras.layers.Dense(num_anchors * 4, name=f"{name_prefix}bbox_output")(features)
        
        # Classification (num_classes per anchor)
        if num_classes is None:
            num_classes = 80  # COCO default
        classes = tf.keras.layers.Dense(num_anchors * num_classes, activation='sigmoid',
                                      name=f"{name_prefix}class_output")(features)
        
        # Objectness scores
        objectness = tf.keras.layers.Dense(num_anchors, activation='sigmoid',
                                         name=f"{name_prefix}objectness_output")(features)
        
        return [boxes, classes, objectness]
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
