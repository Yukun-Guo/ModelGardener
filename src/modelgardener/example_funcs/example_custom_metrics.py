"""
Example custom metrics for the ModelGardener application.

This file demonstrates how to create custom metrics that can be loaded
into the ModelGardener parameter tree. All functions/classes should follow the pattern:
- Accept 'y_true' and 'y_pred' as the first two parameters
- Additional parameters for customization
- Return a scalar metric value

Functions can be pure functions or classes with update_state/result methods (TensorFlow style).
"""

import tensorflow as tf

def balanced_accuracy(y_true, y_pred, threshold=0.5):
    """
    Balanced accuracy metric that accounts for class imbalance.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities or logits
        threshold: Decision threshold for binary classification
    
    Returns:
        Balanced accuracy score
    """
    # Convert predictions to binary
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true, tf.float32)
    
    # Calculate true positives, true negatives, false positives, false negatives
    tp = tf.reduce_sum(y_true_binary * y_pred_binary)
    tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
    fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
    fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
    
    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Balanced accuracy is the average of sensitivity and specificity
    balanced_acc = (sensitivity + specificity) / 2.0
    
    return balanced_acc

def weighted_dice(y_true, y_pred, smooth=1e-6):
    """
    Weighted Dice coefficient for multi-class segmentation.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (one-hot encoded)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Weighted Dice coefficient
    """
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
    union = tf.reduce_sum(y_true + y_pred, axis=[0, 1])

    # Calculate Dice coefficient for each class
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Calculate class weights (e.g., based on frequency)
    class_weights = tf.reduce_sum(y_true, axis=[0, 1]) / tf.reduce_sum(y_true)

    # Calculate weighted Dice
    weighted_dice = tf.reduce_sum(class_weights * dice)

    return weighted_dice