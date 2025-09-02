"""
Example custom loss functions for the ModelGardener application.

This file demonstrates how to create custom loss functions that can be loaded
into the ModelGardener parameter tree. All functions should follow the pattern:
- Accept 'y_true' and 'y_pred' as the first two parameters
- Additional parameters for customization
- Return a scalar loss value

Functions can be pure functions or classes with __call__ method.
"""

import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for image segmentation tasks.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value (1 - Dice coefficient)
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_coef
    
    return dice_loss
