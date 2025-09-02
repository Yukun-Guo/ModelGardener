"""
Custom Loss Functions Template for ModelGardener

This file provides templates for creating custom loss functions.
All loss functions should accept y_true and y_pred as the first two parameters.
"""

import tensorflow as tf
import numpy as np


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    """
    Focal loss for addressing class imbalance.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        alpha: Weighting factor for rare class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        from_logits: Whether predictions are logits or probabilities
        
    Returns:
        Focal loss value
    """
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    # Calculate focal weight
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = alpha * tf.pow((1 - p_t), gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(focal_loss)


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for segmentation tasks.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return dice loss
    return 1.0 - dice


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss for siamese networks.
    
    Args:
        y_true: Binary labels (1 for similar, 0 for dissimilar)
        y_pred: Distance between embeddings
        margin: Margin for dissimilar pairs
        
    Returns:
        Contrastive loss value
    """
    # Square the distance
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    
    # Calculate loss
    loss = y_true * square_pred + (1 - y_true) * margin_square
    
    return tf.reduce_mean(loss) / 2.0


def weighted_categorical_crossentropy(y_true, y_pred, class_weights=None):
    """
    Weighted categorical crossentropy for imbalanced datasets.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_weights: Weight for each class
        
    Returns:
        Weighted crossentropy loss
    """
    if class_weights is None:
        class_weights = tf.ones(tf.shape(y_pred)[-1])
    
    # Calculate crossentropy
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    # Apply class weights
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_loss = weights * ce_loss
    
    return tf.reduce_mean(weighted_loss)


def smooth_l1_loss(y_true, y_pred, delta=1.0):
    """
    Smooth L1 loss (Huber loss).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        delta: Threshold for switching between L1 and L2 loss
        
    Returns:
        Smooth L1 loss value
    """
    diff = tf.abs(y_true - y_pred)
    
    # Use L2 loss for small errors, L1 for large errors
    loss = tf.where(
        diff < delta,
        0.5 * tf.square(diff),
        delta * diff - 0.5 * tf.square(delta)
    )
    
    return tf.reduce_mean(loss)


def cosine_similarity_loss(y_true, y_pred):
    """
    Cosine similarity loss for embedding learning.
    
    Args:
        y_true: Ground truth embeddings
        y_pred: Predicted embeddings
        
    Returns:
        Cosine similarity loss
    """
    # Normalize vectors
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Calculate cosine similarity
    cosine_similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Return loss (1 - similarity)
    return tf.reduce_mean(1.0 - cosine_similarity)


class CustomLossClass:
    """
    Example of a class-based custom loss function.
    """
    
    def __init__(self, weight=1.0, reduction='mean'):
        self.weight = weight
        self.reduction = reduction
        
    def __call__(self, y_true, y_pred):
        """Calculate the loss."""
        # Implement your custom loss logic here
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = loss * self.weight
        
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        else:
            return loss


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom loss functions...")
    
    # Create dummy data for testing
    y_true = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], dtype=tf.float32)
    
    # Test focal loss
    fl = focal_loss(y_true, y_pred)
    print(f"Focal loss: {fl:.4f}")
    
    # Test weighted crossentropy
    weights = tf.constant([1.0, 2.0, 1.5])  # Higher weight for class 1
    wce = weighted_categorical_crossentropy(y_true, y_pred, weights)
    print(f"Weighted CE loss: {wce:.4f}")
    
    print("✅ Custom loss functions template ready!")
