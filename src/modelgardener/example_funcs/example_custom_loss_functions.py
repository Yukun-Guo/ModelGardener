"""
Enhanced custom loss functions for ModelGardener.

This file demonstrates how to create custom loss functions that can be loaded
into the ModelGardener parameter tree. All functions should follow the pattern:
- Outer function: Accepts configuration parameters (will be set in config.yaml)
- Inner wrapper function: Accepts (y_true, y_pred) and returns a scalar loss value or tensor
- Configuration parameters are set at the outer function level

Example usage pattern:
def loss_name(param1=default1, param2=default2):
    def wrapper(y_true, y_pred):
        # Apply loss calculation logic here
        loss_value = compute_loss(y_true, y_pred, param1, param2)
        return loss_value
    return wrapper
"""

import tensorflow as tf

def example_loss_1(param1=1.0, param2=0.5):
    def wrapper(y_true, y_pred):
        # Example loss calculation logic
        loss_value = tf.reduce_mean(tf.square(y_true - y_pred)) * param1 + param2
        return loss_value
    return wrapper

def example_loss_2(param1=1.0, param2=0.5):
    def wrapper(y_true, y_pred):
        # Example loss calculation logic
        loss_value = tf.reduce_mean(tf.abs(y_true - y_pred)) * param1 + param2
        return loss_value
    return wrapper