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
import numpy as np


def custom_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    """
    Custom implementation of Focal Loss for addressing class imbalance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        from_logits: Whether y_pred is logits or probabilities
    
    Returns:
        Focal loss value
    """
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)
    
    # Compute cross entropy
    ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    
    # Compute p_t
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    # Compute alpha_t
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # Compute focal weight
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    # Compute focal loss
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(focal_loss)


def weighted_categorical_crossentropy(y_true, y_pred, class_weights=None, from_logits=False):
    """
    Weighted categorical crossentropy loss for imbalanced datasets.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels
        class_weights: List of weights for each class
        from_logits: Whether y_pred is logits or probabilities
    
    Returns:
        Weighted categorical crossentropy loss
    """
    if class_weights is None:
        class_weights = [1.0] * tf.shape(y_pred)[-1]
    
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    
    # Compute cross entropy
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-8), axis=-1)
    
    # Apply class weights
    weights = tf.reduce_sum(y_true * class_weights, axis=-1)
    weighted_loss = ce_loss * weights
    
    return tf.reduce_mean(weighted_loss)


def smooth_l1_loss(y_true, y_pred, delta=1.0):
    """
    Smooth L1 loss (Huber loss) for robust regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        delta: Threshold for switching between L1 and L2 loss
    
    Returns:
        Smooth L1 loss value
    """
    error = y_true - y_pred
    abs_error = tf.abs(error)
    
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    loss = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(loss)


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss for siamese networks and similarity learning.
    
    Args:
        y_true: True similarity labels (0 for different, 1 for similar)
        y_pred: Predicted distances
        margin: Margin for dissimilar pairs
    
    Returns:
        Contrastive loss value
    """
    similar_loss = y_true * tf.square(y_pred)
    dissimilar_loss = (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
    
    loss = 0.5 * (similar_loss + dissimilar_loss)
    return tf.reduce_mean(loss)


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


class TverskyLoss:
    """
    Tversky loss class for imbalanced segmentation tasks.
    
    This is an example of a class-based loss function.
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Initialize Tversky loss.
        
        Args:
            alpha: Weight for false negatives
            beta: Weight for false positives  
            smooth: Smoothing factor
        """
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def __call__(self, y_true, y_pred):
        """
        Compute Tversky loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tversky loss value
        """
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        
        return 1.0 - tversky


class AdaptiveLoss:
    """
    Adaptive loss that changes behavior based on training progress.
    
    Example of a more complex custom loss function.
    """
    
    def __init__(self, initial_loss='mse', switch_loss='mae', switch_epoch=50):
        """
        Initialize adaptive loss.
        
        Args:
            initial_loss: Loss function to use initially
            switch_loss: Loss function to switch to
            switch_epoch: Epoch at which to switch losses
        """
        self.initial_loss = initial_loss
        self.switch_loss = switch_loss
        self.switch_epoch = switch_epoch
        self.current_epoch = tf.Variable(0, trainable=False)
    
    def __call__(self, y_true, y_pred):
        """
        Compute adaptive loss based on current epoch.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Adaptive loss value
        """
        if self.initial_loss == 'mse':
            initial_loss_fn = tf.keras.losses.mean_squared_error
        else:
            initial_loss_fn = tf.keras.losses.mean_absolute_error
            
        if self.switch_loss == 'mae':
            switch_loss_fn = tf.keras.losses.mean_absolute_error
        else:
            switch_loss_fn = tf.keras.losses.mean_squared_error
        
        # Use initial loss for early epochs, switch loss for later epochs
        loss = tf.cond(
            self.current_epoch < self.switch_epoch,
            lambda: initial_loss_fn(y_true, y_pred),
            lambda: switch_loss_fn(y_true, y_pred)
        )
        
        return tf.reduce_mean(loss)
    
    def update_epoch(self, epoch):
        """Update the current epoch counter."""
        self.current_epoch.assign(epoch)


def cosine_similarity_loss(y_true, y_pred):
    """
    Cosine similarity loss for embedding learning.
    
    Args:
        y_true: True embeddings
        y_pred: Predicted embeddings
        
    Returns:
        Negative cosine similarity (to minimize)
    """
    # Normalize vectors
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Compute cosine similarity
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Return negative similarity (to minimize)
    return -tf.reduce_mean(cosine_sim)
