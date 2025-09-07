"""
Enhanced custom metrics for ModelGardener.

Supports:
- Multi-input and multi-output models
- 2D and 3D data
- Multiple task types (classification, segmentation, object detection)

This file demonstrates how to create custom metrics that can be loaded
into the ModelGardener parameter tree. All functions/classes should follow the pattern:
- Accept 'y_true' and 'y_pred' as the first two parameters
- Additional parameters for customization
- Return a scalar metric value

Functions can be pure functions or classes with update_state/result methods (TensorFlow style).
"""

import tensorflow as tf
from typing import Union, List, Dict, Tuple, Any
from .utils import TaskType, DataDimension, detect_data_dimension, infer_task_type

def enhanced_balanced_accuracy(y_true, y_pred, threshold=0.5, apply_to_3d=True):
    """
    Enhanced balanced accuracy metric with 2D/3D support.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities or logits
        threshold: Decision threshold for binary classification
        apply_to_3d: Whether to apply to 3D data
    
    Returns:
        Balanced accuracy score
    """
    # Handle multi-class case
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        # Multi-class - convert to argmax
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        # Calculate per-class accuracy and average
        num_classes = y_true.shape[-1]
        class_accuracies = []
        
        for i in range(num_classes):
            class_mask = tf.equal(y_true_class, i)
            if tf.reduce_sum(tf.cast(class_mask, tf.float32)) > 0:
                class_pred_correct = tf.equal(y_pred_class, i)
                class_accuracy = tf.reduce_sum(tf.cast(tf.logical_and(class_mask, class_pred_correct), tf.float32)) / tf.reduce_sum(tf.cast(class_mask, tf.float32))
                class_accuracies.append(class_accuracy)
        
        if class_accuracies:
            return tf.reduce_mean(class_accuracies)
        else:
            return 0.0
    else:
        # Binary case
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)
        
        # Calculate confusion matrix elements
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # Balanced accuracy
        balanced_acc = (sensitivity + specificity) / 2.0
        return balanced_acc

def enhanced_dice_coefficient(y_true, y_pred, smooth=1e-6, axis=None, apply_to_3d=True):
    """
    Enhanced Dice coefficient for segmentation with 2D/3D support.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        smooth: Smoothing factor to avoid division by zero
        axis: Axis along which to compute the metric
        apply_to_3d: Whether to apply to 3D data

    Returns:
        Dice coefficient
    """
    # Determine if we're dealing with 3D data
    is_3d = len(y_true.shape) > 4 or (len(y_true.shape) == 4 and y_true.shape[-1] > 3)
    
    if axis is None:
        if is_3d and apply_to_3d:
            axis = [1, 2, 3] if len(y_true.shape) == 5 else [1, 2, 3]
        else:
            axis = [1, 2] if len(y_true.shape) == 4 else [0, 1]

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)

    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return tf.reduce_mean(dice)

def iou_score(y_true, y_pred, smooth=1e-6, threshold=0.5):
    """
    Intersection over Union (IoU) score for segmentation.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        smooth: Smoothing factor
        threshold: Threshold for binary segmentation
    
    Returns:
        IoU score
    """
    # Threshold predictions if needed
    if threshold is not None:
        y_pred = tf.cast(y_pred > threshold, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def precision_score(y_true, y_pred, threshold=0.5, average='macro'):
    """
    Precision score with support for multi-class and binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels/probabilities
        threshold: Threshold for binary classification
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        Precision score
    """
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        # Multi-class case
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        num_classes = y_true.shape[-1]
        precisions = []
        
        for i in range(num_classes):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_class, i), tf.equal(y_pred_class, i)), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true_class, i), tf.equal(y_pred_class, i)), tf.float32))
            precision = tp / (tp + fp + 1e-8)
            precisions.append(precision)
        
        if average == 'macro':
            return tf.reduce_mean(precisions)
        elif average == 'micro':
            # Micro-average
            total_tp = tf.reduce_sum([tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_class, i), tf.equal(y_pred_class, i)), tf.float32)) for i in range(num_classes)])
            total_fp = tf.reduce_sum([tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true_class, i), tf.equal(y_pred_class, i)), tf.float32)) for i in range(num_classes)])
            return total_tp / (total_tp + total_fp + 1e-8)
        else:
            return tf.reduce_mean(precisions)
    else:
        # Binary case
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        
        precision = tp / (tp + fp + 1e-8)
        return precision

def recall_score(y_true, y_pred, threshold=0.5, average='macro'):
    """
    Recall score with support for multi-class and binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels/probabilities
        threshold: Threshold for binary classification
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        Recall score
    """
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        # Multi-class case
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        num_classes = y_true.shape[-1]
        recalls = []
        
        for i in range(num_classes):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_class, i), tf.equal(y_pred_class, i)), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_class, i), tf.not_equal(y_pred_class, i)), tf.float32))
            recall = tp / (tp + fn + 1e-8)
            recalls.append(recall)
        
        if average == 'macro':
            return tf.reduce_mean(recalls)
        else:
            return tf.reduce_mean(recalls)
    else:
        # Binary case
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        recall = tp / (tp + fn + 1e-8)
        return recall

def f1_score(y_true, y_pred, threshold=0.5, average='macro'):
    """
    F1 score combining precision and recall.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels/probabilities
        threshold: Threshold for binary classification
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        F1 score
    """
    precision = precision_score(y_true, y_pred, threshold, average)
    recall = recall_score(y_true, y_pred, threshold, average)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

def mean_absolute_error_3d(y_true, y_pred):
    """
    Mean Absolute Error for 3D data (e.g., volumetric regression).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE for 3D data
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def peak_signal_noise_ratio(y_true, y_pred, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio for image quality assessment.
    
    Args:
        y_true: True images
        y_pred: Predicted/reconstructed images
        max_val: Maximum possible pixel value
    
    Returns:
        PSNR value
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    psnr = 20 * tf.math.log(max_val) / tf.math.log(10.0) - 10 * tf.math.log(mse) / tf.math.log(10.0)
    return psnr

def multi_output_accuracy(y_true, y_pred, weights=[1.0, 0.5]):
    """
    Multi-output accuracy for models with multiple outputs.
    
    Args:
        y_true: List or dict of true labels for each output
        y_pred: List or dict of predictions for each output
        weights: Weights for each output
    
    Returns:
        Weighted accuracy across outputs
    """
    total_accuracy = 0.0
    total_weight = 0.0
    
    if isinstance(y_true, (list, tuple)):
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            weight = weights[i] if i < len(weights) else 1.0
            accuracy = tf.keras.metrics.categorical_accuracy(true, pred)
            accuracy = tf.reduce_mean(accuracy)
            total_accuracy += weight * accuracy
            total_weight += weight
    elif isinstance(y_true, dict):
        for i, (key, true) in enumerate(y_true.items()):
            pred = y_pred[key]
            weight = weights[i] if i < len(weights) else 1.0
            accuracy = tf.keras.metrics.categorical_accuracy(true, pred)
            accuracy = tf.reduce_mean(accuracy)
            total_accuracy += weight * accuracy
            total_weight += weight
    else:
        # Single output case
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        return tf.reduce_mean(accuracy)
    
    return total_accuracy / (total_weight + 1e-8)

# Backward compatibility
balanced_accuracy = enhanced_balanced_accuracy
weighted_dice = enhanced_dice_coefficient