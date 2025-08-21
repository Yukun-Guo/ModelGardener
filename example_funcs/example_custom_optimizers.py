"""
Example custom optimizer for ModelGardener

This file demonstrates how to create custom optimizers that can be loaded
into the ModelGardener application.
"""

import tensorflow as tf


def custom_sgd_with_warmup(learning_rate=0.01, warmup_steps=1000, momentum=0.9):
    """
    Custom SGD optimizer with learning rate warmup.
    
    Args:
        learning_rate: Base learning rate
        warmup_steps: Number of steps for warmup period
        momentum: Momentum factor
    
    Returns:
        TensorFlow optimizer instance
    """
    
    # Create a learning rate schedule with warmup
    def warmup_schedule(step):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        else:
            return learning_rate
    
    lr_schedule = tf.keras.optimizers.schedules.LambdaCallback(warmup_schedule)
    
    return tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum,
        name="CustomSGDWithWarmup"
    )


def adaptive_adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                 epsilon=1e-7, decay_factor=0.99):
    """
    Custom Adam optimizer with adaptive learning rate decay.
    
    Args:
        learning_rate: Initial learning rate
        beta_1: Exponential decay rate for first moment estimates
        beta_2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
        decay_factor: Factor for exponential learning rate decay
    
    Returns:
        TensorFlow optimizer instance
    """
    
    # Create exponential decay schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_factor,
        staircase=True
    )
    
    return tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        name="AdaptiveAdam"
    )


def cyclical_sgd(base_lr=0.001, max_lr=0.01, step_size=2000, momentum=0.9):
    """
    SGD optimizer with cyclical learning rate.
    
    Args:
        base_lr: Minimum learning rate
        max_lr: Maximum learning rate
        step_size: Half period of the cycle
        momentum: Momentum factor
    
    Returns:
        TensorFlow optimizer instance
    """
    
    def cyclical_schedule(step):
        cycle = tf.floor(1 + step / (2 * step_size))
        x = tf.abs(step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, 1 - x)
        return lr
    
    lr_schedule = tf.keras.optimizers.schedules.LambdaCallback(cyclical_schedule)
    
    return tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum,
        name="CyclicalSGD"
    )
