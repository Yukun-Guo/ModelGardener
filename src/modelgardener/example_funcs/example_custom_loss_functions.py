"""
Enhanced custom loss functions for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

This file demonstrates how to create custom loss functions that can be loaded
into the ModelGardener parameter tree. All functions should follow the pattern:
- Accept 'y_true' and 'y_pred' as the first two parameters
- Additional parameters for customization
- Return a scalar loss value

Functions can be pure functions or classes with __call__ method.
"""

import tensorflow as tf
from typing import Union, List, Dict, Tuple, Any
from .utils import TaskType, DataDimension, detect_data_dimension, infer_task_type

def enhanced_dice_loss(y_true, y_pred, smooth=1.0, axis=None, apply_to_3d=True):
    """
    Enhanced Dice loss for segmentation tasks with 2D/3D support.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks  
        smooth: Smoothing factor to avoid division by zero
        axis: Axis along which to compute the loss (None for all axes)
        apply_to_3d: Whether to apply to 3D data
    
    Returns:
        Dice loss value (1 - Dice coefficient)
    """
    # Determine if we're dealing with 3D data
    is_3d = len(y_true.shape) > 4 or (len(y_true.shape) == 4 and y_true.shape[-1] > 3)
    
    if axis is None:
        if is_3d and apply_to_3d:
            # For 3D data, flatten spatial dimensions but keep batch and channel
            axis = [1, 2, 3] if len(y_true.shape) == 5 else [1, 2, 3]
        else:
            # For 2D data, flatten spatial dimensions
            axis = [1, 2] if len(y_true.shape) == 4 else [0, 1]
    
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
    
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_coef
    
    return tf.reduce_mean(dice_loss)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
    """
    Tversky loss for imbalanced segmentation tasks.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        alpha: Weight for false positives
        beta: Weight for false negatives (alpha + beta should = 1)
        smooth: Smoothing factor
    
    Returns:
        Tversky loss value
    """
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    true_pos = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=1)
    false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=1)
    
    tversky_coef = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    tversky_loss = 1.0 - tversky_coef
    
    return tf.reduce_mean(tversky_loss)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for dealing with class imbalance in classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    ce_loss = -y_true * tf.math.log(y_pred)
    
    # Calculate p_t
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    
    # Calculate alpha_t
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Calculate focal weight
    focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
    
    # Calculate focal loss
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

def combined_loss(y_true, y_pred, dice_weight=0.5, ce_weight=0.5, smooth=1.0):
    """
    Combined Dice and Cross-Entropy loss for segmentation.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for Cross-Entropy loss component
        smooth: Smoothing factor for Dice loss
    
    Returns:
        Combined loss value
    """
    # Dice loss component
    dice_loss_val = enhanced_dice_loss(y_true, y_pred, smooth=smooth)
    
    # Cross-entropy loss component
    ce_loss_val = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    ce_loss_val = tf.reduce_mean(ce_loss_val)
    
    # Combine losses
    combined = dice_weight * dice_loss_val + ce_weight * ce_loss_val
    
    return combined

def yolo_loss(y_true, y_pred, lambda_coord=5.0, lambda_noobj=0.5, num_classes=80):
    """
    YOLO-style loss for object detection (simplified version).
    
    Args:
        y_true: True bounding boxes and classes [batch, grid, grid, anchors, (x, y, w, h, conf, classes...)]
        y_pred: Predicted bounding boxes and classes
        lambda_coord: Weight for coordinate loss
        lambda_noobj: Weight for no-object loss
        num_classes: Number of classes
    
    Returns:
        YOLO loss value
    """
    # Extract components from predictions and ground truth
    # This is a simplified implementation - real YOLO loss is more complex
    
    # Coordinate loss (for boxes that contain objects)
    coord_mask = y_true[..., 4:5]  # Object mask
    coord_loss = tf.reduce_sum(coord_mask * tf.square(y_true[..., :4] - y_pred[..., :4]))
    
    # Confidence loss
    conf_loss_obj = tf.reduce_sum(coord_mask * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))
    conf_loss_noobj = tf.reduce_sum((1 - coord_mask) * tf.square(y_true[..., 4:5] - y_pred[..., 4:5]))
    
    # Classification loss (for boxes that contain objects)
    class_loss = tf.reduce_sum(coord_mask * tf.square(y_true[..., 5:] - y_pred[..., 5:]))
    
    # Total loss
    total_loss = (lambda_coord * coord_loss + 
                 conf_loss_obj + 
                 lambda_noobj * conf_loss_noobj + 
                 class_loss)
    
    return total_loss

def multi_output_loss(y_true, y_pred, loss_weights=[1.0, 0.5], loss_functions=None):
    """
    Multi-output loss function for models with multiple outputs.
    
    Args:
        y_true: List or dict of true labels for each output
        y_pred: List or dict of predictions for each output
        loss_weights: Weights for each output loss
        loss_functions: List of loss functions for each output
    
    Returns:
        Weighted sum of losses
    """
    if loss_functions is None:
        loss_functions = [tf.keras.losses.categorical_crossentropy] * len(y_true)
    
    total_loss = 0.0
    
    if isinstance(y_true, (list, tuple)):
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            weight = loss_weights[i] if i < len(loss_weights) else 1.0
            loss_fn = loss_functions[i] if i < len(loss_functions) else loss_functions[0]
            loss_val = tf.reduce_mean(loss_fn(true, pred))
            total_loss += weight * loss_val
    elif isinstance(y_true, dict):
        for i, (key, true) in enumerate(y_true.items()):
            pred = y_pred[key]
            weight = loss_weights[i] if i < len(loss_weights) else 1.0
            loss_fn = loss_functions[i] if i < len(loss_functions) else loss_functions[0]
            loss_val = tf.reduce_mean(loss_fn(true, pred))
            total_loss += weight * loss_val
    else:
        # Single output case
        loss_fn = loss_functions[0] if loss_functions else tf.keras.losses.categorical_crossentropy
        total_loss = tf.reduce_mean(loss_fn(y_true, y_pred))
    
    return total_loss

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss for robust regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold parameter
    
    Returns:
        Huber loss value
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
    
    huber_loss = tf.where(is_small_error, squared_loss, linear_loss)
    return tf.reduce_mean(huber_loss)

# Backward compatibility
dice_loss = enhanced_dice_loss
