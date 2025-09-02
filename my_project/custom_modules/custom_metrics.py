"""
Custom Metrics Template for ModelGardener

This file provides templates for creating custom metrics.
Metrics should inherit from tf.keras.metrics.Metric or be functions.
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric.
    """
    
    def __init__(self, name='f1_score', average='macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.average = average
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class IoU(tf.keras.metrics.Metric):
    """
    Intersection over Union metric for segmentation.
    """
    
    def __init__(self, num_classes, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name='total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class predictions
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Calculate confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        self.total_cm.assign_add(tf.cast(cm, tf.float32))
    
    def result(self):
        # Calculate IoU from confusion matrix
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        true_positives = tf.linalg.diag_part(self.total_cm)
        
        denominator = sum_over_row + sum_over_col - true_positives
        iou = tf.math.divide_no_nan(true_positives, denominator)
        
        return tf.reduce_mean(iou)
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


class TopKAccuracy(tf.keras.metrics.Metric):
    """
    Custom Top-K accuracy metric.
    """
    
    def __init__(self, k=5, name='top_k_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.correct_predictions = self.add_weight(name='correct', initializer='zeros')
        self.total_predictions = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get true class indices
        y_true_indices = tf.argmax(y_true, axis=-1)
        
        # Get top-k predictions
        top_k_pred = tf.nn.top_k(y_pred, k=self.k).indices
        
        # Check if true class is in top-k
        correct = tf.reduce_any(tf.equal(tf.expand_dims(y_true_indices, -1), top_k_pred), axis=-1)
        correct = tf.cast(correct, tf.float32)
        
        self.correct_predictions.assign_add(tf.reduce_sum(correct))
        self.total_predictions.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.correct_predictions / self.total_predictions
    
    def reset_state(self):
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)


def balanced_accuracy(y_true, y_pred):
    """
    Balanced accuracy function-based metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Balanced accuracy score
    """
    # Convert to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Get unique classes
    classes = tf.unique(y_true)[0]
    
    # Calculate recall for each class
    recalls = []
    for cls in classes:
        mask = tf.equal(y_true, cls)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(mask, tf.equal(y_pred, cls)), tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(mask, tf.float32))
        recall = true_positives / (actual_positives + 1e-8)
        recalls.append(recall)
    
    # Return mean recall (balanced accuracy)
    return tf.reduce_mean(tf.stack(recalls))


def matthews_correlation_coefficient(y_true, y_pred):
    """
    Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        
    Returns:
        Matthews correlation coefficient
    """
    # Convert to binary predictions
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate confusion matrix elements
    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Calculate MCC
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + 1e-8)


class AverageMetric(tf.keras.metrics.Metric):
    """
    Metric that computes the average of multiple metrics.
    """
    
    def __init__(self, metrics, name='average_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.metrics = metrics
        self.total_sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update all metrics and compute average
        metric_values = []
        for metric in self.metrics:
            if hasattr(metric, 'update_state'):
                metric.update_state(y_true, y_pred, sample_weight)
                metric_values.append(metric.result())
            else:
                metric_values.append(metric(y_true, y_pred))
        
        avg_value = tf.reduce_mean(tf.stack(metric_values))
        self.total_sum.assign_add(avg_value)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.total_sum / self.count
    
    def reset_state(self):
        self.total_sum.assign(0)
        self.count.assign(0)
        for metric in self.metrics:
            if hasattr(metric, 'reset_state'):
                metric.reset_state()


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom metrics...")
    
    # Create dummy data
    y_true = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]], dtype=tf.float32)
    
    # Test F1 Score
    f1_metric = F1Score()
    f1_metric.update_state(y_true, y_pred)
    print(f"F1 Score: {f1_metric.result():.4f}")
    
    # Test Top-K Accuracy
    topk_metric = TopKAccuracy(k=2)
    topk_metric.update_state(y_true, y_pred)
    print(f"Top-2 Accuracy: {topk_metric.result():.4f}")
    
    # Test balanced accuracy
    ba = balanced_accuracy(y_true, y_pred)
    print(f"Balanced Accuracy: {ba:.4f}")
    
    print("✅ Custom metrics template ready!")
