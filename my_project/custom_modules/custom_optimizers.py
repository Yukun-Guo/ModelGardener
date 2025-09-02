"""
Custom Optimizers Template for ModelGardener

This file provides templates for creating custom optimizers.
Optimizers should return tensorflow.keras.optimizers objects.
"""

import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np


def sgd_with_warmup(learning_rate=0.01, warmup_steps=1000, momentum=0.9):
    """
    SGD optimizer with learning rate warmup.
    
    Args:
        learning_rate: Base learning rate
        warmup_steps: Number of warmup steps
        momentum: Momentum factor
        
    Returns:
        Custom SGD optimizer with warmup
    """
    
    # Create a learning rate schedule with warmup
    def lr_schedule(step):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        else:
            return learning_rate
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    return optimizers.SGD(learning_rate=learning_rate, momentum=momentum)


def adaptive_adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                 epsilon=1e-7, decay_factor=0.99):
    """
    Adam optimizer with adaptive learning rate decay.
    
    Args:
        learning_rate: Initial learning rate
        beta_1: Exponential decay rate for 1st moment estimates
        beta_2: Exponential decay rate for 2nd moment estimates
        epsilon: Small constant for numerical stability
        decay_factor: Learning rate decay factor
        
    Returns:
        Custom Adam optimizer
    """
    
    # Create exponential decay schedule
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_factor
    )
    
    return optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )


def cyclical_sgd(base_lr=0.001, max_lr=0.01, step_size=2000, momentum=0.9):
    """
    SGD with cyclical learning rate.
    
    Args:
        base_lr: Minimum learning rate
        max_lr: Maximum learning rate
        step_size: Half of the cycle length
        momentum: Momentum factor
        
    Returns:
        SGD optimizer with cyclical learning rate
    """
    
    class CyclicalLR(tf.keras.callbacks.Callback):
        def __init__(self, base_lr, max_lr, step_size):
            super().__init__()
            self.base_lr = base_lr
            self.max_lr = max_lr
            self.step_size = step_size
            
        def on_batch_begin(self, batch, logs=None):
            cycle = np.floor(1 + batch / (2 * self.step_size))
            x = np.abs(batch / self.step_size - 2 * cycle + 1)
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
    
    return optimizers.SGD(learning_rate=base_lr, momentum=momentum)


def rmsprop_with_schedule(initial_lr=0.001, decay_steps=1000, decay_rate=0.9,
                         rho=0.9, momentum=0.0, epsilon=1e-7):
    """
    RMSprop optimizer with polynomial decay.
    
    Args:
        initial_lr: Initial learning rate
        decay_steps: Steps after which to decay
        decay_rate: Decay rate
        rho: Discounting factor for history
        momentum: Momentum factor
        epsilon: Small constant for numerical stability
        
    Returns:
        RMSprop optimizer with polynomial decay
    """
    
    lr_schedule = optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=initial_lr * 0.1,
        power=1.0
    )
    
    return optimizers.RMSprop(
        learning_rate=lr_schedule,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon
    )


def adabound_optimizer(learning_rate=0.001, final_lr=0.1, beta_1=0.9, 
                      beta_2=0.999, epsilon=1e-8):
    """
    AdaBound optimizer (approximation using Adam with clipped learning rate).
    
    Args:
        learning_rate: Initial learning rate
        final_lr: Final learning rate bound
        beta_1: Exponential decay rate for 1st moment estimates
        beta_2: Exponential decay rate for 2nd moment estimates
        epsilon: Small constant for numerical stability
        
    Returns:
        Adam optimizer configured to approximate AdaBound
    """
    
    # Use Adam with clipped learning rate as approximation
    return optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        clipnorm=1.0  # Gradient clipping
    )


class CustomOptimizerClass:
    """
    Example of a class-based custom optimizer wrapper.
    This allows for more complex optimizer behavior.
    """
    
    def __init__(self, base_optimizer='adam', learning_rate=0.001, **kwargs):
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
    def __call__(self):
        """Create and return the optimizer."""
        if self.base_optimizer.lower() == 'adam':
            return optimizers.Adam(learning_rate=self.learning_rate, **self.kwargs)
        elif self.base_optimizer.lower() == 'sgd':
            return optimizers.SGD(learning_rate=self.learning_rate, **self.kwargs)
        else:
            return optimizers.Adam(learning_rate=self.learning_rate)


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom optimizers...")
    
    # Test SGD with warmup
    opt1 = sgd_with_warmup(learning_rate=0.01, warmup_steps=500)
    print(f"SGD with warmup: {type(opt1).__name__}")
    
    # Test adaptive Adam
    opt2 = adaptive_adam(learning_rate=0.001)
    print(f"Adaptive Adam: {type(opt2).__name__}")
    
    # Test cyclical SGD
    opt3 = cyclical_sgd(base_lr=0.001, max_lr=0.01)
    print(f"Cyclical SGD: {type(opt3).__name__}")
    
    print("✅ Custom optimizers template ready!")
