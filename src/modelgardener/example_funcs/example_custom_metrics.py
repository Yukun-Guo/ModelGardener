"""
Enhanced custom metrics for ModelGardener.

This file demonstrates how to create custom metric functions that can be loaded
into the ModelGardener parameter tree. All functions should follow the pattern:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (y_true, y_pred) and returns a scalar metric value or tensor
- Configuration parameters are set at the outer function level

Example usage pattern:
def metric_name(param1=default1, param2=default2):
    def wrapper(y_true, y_pred):
        # Apply metric calculation logic here
        metric_value = compute_metric(y_true, y_pred, param1, param2)
        return metric_value
    return wrapper
"""

import tensorflow as tf

def example_metric_1(param1=1.0, param2=0.5):
    def wrapper(y_true, y_pred):
        # Example metric calculation logic
        metric_value = tf.reduce_mean(tf.abs(y_true - y_pred)) * param1 + param2
        return metric_value
    return wrapper