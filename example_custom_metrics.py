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
import numpy as np


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


def matthews_correlation_coefficient(y_true, y_pred, threshold=0.5):
    """
    Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        Matthews correlation coefficient (-1 to 1)
    """
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true, tf.float32)
    
    tp = tf.reduce_sum(y_true_binary * y_pred_binary)
    tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
    fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
    fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
    
    # MCC formula
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    mcc = numerator / (denominator + 1e-8)
    return mcc


def weighted_f1_score(y_true, y_pred, class_weights=None, threshold=0.5):
    """
    Weighted F1 score for imbalanced datasets.
    
    Args:
        y_true: True labels (one-hot or sparse)
        y_pred: Predicted probabilities
        class_weights: Weights for each class
        threshold: Decision threshold (for binary classification)
    
    Returns:
        Weighted F1 score
    """
    if len(tf.shape(y_pred)) == 2 and tf.shape(y_pred)[-1] > 1:
        # Multi-class case
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true_classes = tf.argmax(y_true, axis=-1) if len(tf.shape(y_true)) == 2 else y_true
        
        num_classes = tf.shape(y_pred)[-1]
        f1_scores = []
        weights = class_weights if class_weights is not None else [1.0] * num_classes
        
        for i in range(num_classes):
            # Binary F1 for each class
            y_true_binary = tf.cast(tf.equal(y_true_classes, i), tf.float32)
            y_pred_binary = tf.cast(tf.equal(y_pred_classes, i), tf.float32)
            
            tp = tf.reduce_sum(y_true_binary * y_pred_binary)
            fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
            fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            f1_scores.append(f1 * weights[i])
        
        return tf.reduce_mean(f1_scores)
    else:
        # Binary case
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1


def mean_intersection_over_union(y_true, y_pred, num_classes=None):
    """
    Mean Intersection over Union (mIoU) for segmentation tasks.
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        num_classes: Number of classes (inferred if None)
    
    Returns:
        Mean IoU score
    """
    if num_classes is None:
        num_classes = tf.reduce_max(tf.cast(y_true, tf.int32)) + 1
    
    # Flatten predictions and true labels
    y_true_flat = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    y_pred_flat = tf.reshape(tf.cast(tf.argmax(y_pred, axis=-1), tf.int32), [-1])
    
    ious = []
    for i in range(num_classes):
        true_class = tf.equal(y_true_flat, i)
        pred_class = tf.equal(y_pred_flat, i)
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_class, pred_class), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(true_class, pred_class), tf.float32))
        
        iou = intersection / (union + 1e-8)
        ious.append(iou)
    
    return tf.reduce_mean(ious)


def top_k_accuracy_custom(y_true, y_pred, k=3):
    """
    Custom implementation of top-k accuracy.
    
    Args:
        y_true: True labels (sparse or one-hot)
        y_pred: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy
    """
    # Convert one-hot to sparse if necessary
    if len(tf.shape(y_true)) == 2:
        y_true = tf.argmax(y_true, axis=-1)
    
    # Get top-k predictions
    _, top_k_pred = tf.nn.top_k(y_pred, k=k)
    
    # Check if true label is in top-k predictions
    y_true_expanded = tf.expand_dims(y_true, axis=-1)
    matches = tf.reduce_any(tf.equal(top_k_pred, y_true_expanded), axis=-1)
    
    return tf.reduce_mean(tf.cast(matches, tf.float32))


class PerClassAccuracy:
    """
    Per-class accuracy metric that tracks accuracy for each class separately.
    
    This is an example of a class-based metric following TensorFlow's metric pattern.
    """
    
    def __init__(self, num_classes, name='per_class_accuracy'):
        """
        Initialize per-class accuracy metric.
        
        Args:
            num_classes: Number of classes
            name: Name of the metric
        """
        self.num_classes = num_classes
        self.name = name
        self.total_per_class = tf.Variable(tf.zeros(num_classes), trainable=False)
        self.correct_per_class = tf.Variable(tf.zeros(num_classes), trainable=False)
    
    def update_state(self, y_true, y_pred):
        """
        Update the metric state with new predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        """
        # Convert to class indices
        if len(tf.shape(y_true)) == 2:
            y_true = tf.argmax(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # Update counts for each class
        for i in range(self.num_classes):
            class_mask = tf.equal(y_true, i)
            class_correct = tf.logical_and(class_mask, tf.equal(y_true, y_pred_classes))
            
            self.total_per_class[i].assign_add(tf.reduce_sum(tf.cast(class_mask, tf.float32)))
            self.correct_per_class[i].assign_add(tf.reduce_sum(tf.cast(class_correct, tf.float32)))
    
    def result(self):
        """
        Compute the final metric result.
        
        Returns:
            Mean accuracy across all classes
        """
        per_class_acc = self.correct_per_class / (self.total_per_class + 1e-8)
        return tf.reduce_mean(per_class_acc)
    
    def reset_states(self):
        """Reset the metric state."""
        self.total_per_class.assign(tf.zeros(self.num_classes))
        self.correct_per_class.assign(tf.zeros(self.num_classes))


class CosineSimilarityMetric:
    """
    Cosine similarity metric for embedding evaluation.
    """
    
    def __init__(self, name='cosine_similarity'):
        """
        Initialize cosine similarity metric.
        
        Args:
            name: Name of the metric
        """
        self.name = name
        self.total_similarity = tf.Variable(0.0, trainable=False)
        self.count = tf.Variable(0.0, trainable=False)
    
    def update_state(self, y_true, y_pred):
        """
        Update metric state.
        
        Args:
            y_true: True embeddings
            y_pred: Predicted embeddings
        """
        # Normalize embeddings
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        
        # Compute cosine similarity
        similarities = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        
        self.total_similarity.assign_add(tf.reduce_sum(similarities))
        self.count.assign_add(tf.cast(tf.shape(similarities)[0], tf.float32))
    
    def result(self):
        """Compute mean cosine similarity."""
        return self.total_similarity / (self.count + 1e-8)
    
    def reset_states(self):
        """Reset metric state."""
        self.total_similarity.assign(0.0)
        self.count.assign(0.0)


def earth_movers_distance(y_true, y_pred):
    """
    Earth Mover's Distance (Wasserstein distance) for probability distributions.
    
    Args:
        y_true: True probability distribution
        y_pred: Predicted probability distribution
    
    Returns:
        Earth Mover's Distance
    """
    # Ensure distributions sum to 1
    y_true_norm = y_true / (tf.reduce_sum(y_true, axis=-1, keepdims=True) + 1e-8)
    y_pred_norm = y_pred / (tf.reduce_sum(y_pred, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cumulative distributions
    cum_true = tf.cumsum(y_true_norm, axis=-1)
    cum_pred = tf.cumsum(y_pred_norm, axis=-1)
    
    # Earth Mover's Distance is the sum of absolute differences between CDFs
    emd = tf.reduce_sum(tf.abs(cum_true - cum_pred), axis=-1)
    
    return tf.reduce_mean(emd)


class AdaptiveThresholdAccuracy:
    """
    Accuracy metric with adaptive threshold selection.
    
    Finds the best threshold to maximize accuracy during evaluation.
    """
    
    def __init__(self, num_thresholds=100, name='adaptive_accuracy'):
        """
        Initialize adaptive threshold accuracy.
        
        Args:
            num_thresholds: Number of thresholds to evaluate
            name: Name of the metric
        """
        self.num_thresholds = num_thresholds
        self.name = name
        self.thresholds = tf.linspace(0.0, 1.0, num_thresholds)
        self.predictions = []
        self.true_labels = []
    
    def update_state(self, y_true, y_pred):
        """Update state with new predictions."""
        self.predictions.append(y_pred)
        self.true_labels.append(y_true)
    
    def result(self):
        """
        Compute accuracy with best threshold.
        
        Returns:
            Maximum accuracy across all thresholds
        """
        if not self.predictions:
            return 0.0
        
        # Concatenate all predictions and labels
        all_pred = tf.concat(self.predictions, axis=0)
        all_true = tf.concat(self.true_labels, axis=0)
        
        accuracies = []
        for threshold in self.thresholds:
            pred_binary = tf.cast(all_pred > threshold, tf.float32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(all_true, pred_binary), tf.float32))
            accuracies.append(accuracy)
        
        return tf.reduce_max(accuracies)
    
    def reset_states(self):
        """Reset metric state."""
        self.predictions = []
        self.true_labels = []


def pearson_correlation(y_true, y_pred):
    """
    Pearson correlation coefficient for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Pearson correlation coefficient
    """
    # Flatten if needed
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Calculate means
    mean_true = tf.reduce_mean(y_true_flat)
    mean_pred = tf.reduce_mean(y_pred_flat)
    
    # Calculate correlation
    numerator = tf.reduce_sum((y_true_flat - mean_true) * (y_pred_flat - mean_pred))
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(y_true_flat - mean_true)) *
        tf.reduce_sum(tf.square(y_pred_flat - mean_pred))
    )
    
    correlation = numerator / (denominator + 1e-8)
    return correlation
