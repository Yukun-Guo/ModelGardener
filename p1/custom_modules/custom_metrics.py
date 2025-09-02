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
