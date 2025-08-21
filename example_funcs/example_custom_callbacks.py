"""
Example custom callback functions and classes for the Model Gardener application.

These callbacks demonstrate how to create custom training callbacks that can be 
dynamically loaded into the callbacks parameter tree. Callbacks can be either:

1. Functions that return a callback instance
2. Classes that inherit from tf.keras.callbacks.Callback

For functions:
- Should return a tf.keras.callbacks.Callback instance
- Can accept configuration parameters

For classes:
- Should inherit from tf.keras.callbacks.Callback
- Should implement relevant callback methods (on_epoch_end, on_batch_end, etc.)
- __init__ method parameters become configuration options
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import tensorflow as tf


class LossThresholdStopping(tf.keras.callbacks.Callback):
    """
    Stop training when loss goes below a specified threshold.
    
    Useful when you want to stop training as soon as the model reaches
    a satisfactory loss level, rather than waiting for validation metrics.
    """
    
    def __init__(self, 
                 loss_threshold: float = 0.1,
                 monitor: str = 'loss',
                 patience: int = 0,
                 restore_best_weights: bool = False):
        """
        Initialize the threshold stopping callback.
        
        Args:
            loss_threshold: Loss value below which to stop training
            monitor: Metric to monitor ('loss', 'val_loss', etc.)
            patience: Number of epochs to wait after threshold is reached
            restore_best_weights: Whether to restore weights from best epoch
        """
        super().__init__()
        self.loss_threshold = loss_threshold
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_weights = None
        self.best_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        
        if current_loss is None:
            return
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        
        if current_loss <= self.loss_threshold:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEpoch {epoch + 1}: {self.monitor} reached threshold {self.loss_threshold}. Stopping training.")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True


class GradientLoggingCallback(tf.keras.callbacks.Callback):
    """
    Log gradient statistics during training for debugging.
    
    Useful for diagnosing gradient-related issues like vanishing or
    exploding gradients.
    """
    
    def __init__(self, 
                 log_frequency: int = 10,
                 log_dir: str = './gradient_logs',
                 layer_names: str = 'all'):
        """
        Initialize gradient logging callback.
        
        Args:
            log_frequency: How often to log gradients (every N batches)
            log_dir: Directory to save gradient logs
            layer_names: Which layers to monitor ('all' or comma-separated names)
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        self.layer_names = layer_names
        self.batch_count = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        
        if self.batch_count % self.log_frequency == 0:
            # Get gradients (this is a simplified example)
            # In practice, you'd need to set up gradient tape tracking
            gradient_stats = {}
            
            for layer in self.model.layers:
                if layer.trainable_weights:
                    layer_name = layer.name
                    if self.layer_names == 'all' or layer_name in self.layer_names.split(','):
                        # This is a placeholder - actual gradient computation would require
                        # more complex setup with tf.GradientTape
                        weights = layer.get_weights()
                        if weights:
                            gradient_stats[layer_name] = {
                                'weight_mean': float(np.mean([np.mean(w) for w in weights])),
                                'weight_std': float(np.mean([np.std(w) for w in weights]))
                            }
            
            # Log gradient statistics
            log_file = os.path.join(self.log_dir, f'gradients_batch_{self.batch_count}.txt')
            with open(log_file, 'w') as f:
                f.write(f"Batch {self.batch_count} Gradient Statistics:\n")
                for layer_name, stats in gradient_stats.items():
                    f.write(f"{layer_name}: mean={stats['weight_mean']:.6f}, std={stats['weight_std']:.6f}\n")


class LearningRateWarmup(tf.keras.callbacks.Callback):
    """
    Implement learning rate warmup for better training stability.
    
    Gradually increases learning rate from a small value to the target
    learning rate over the first few epochs.
    """
    
    def __init__(self,
                 warmup_epochs: int = 5,
                 target_lr: float = 0.001,
                 warmup_start_lr: float = 1e-6):
        """
        Initialize learning rate warmup callback.
        
        Args:
            warmup_epochs: Number of epochs for warmup period
            target_lr: Target learning rate after warmup
            warmup_start_lr: Starting learning rate for warmup
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.warmup_start_lr = warmup_start_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            print(f"Epoch {epoch + 1}: Warmup LR = {lr:.2e}")
        elif epoch == self.warmup_epochs:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.target_lr)
            print(f"Epoch {epoch + 1}: Warmup complete. LR = {self.target_lr:.2e}")


class MemoryUsageMonitor(tf.keras.callbacks.Callback):
    """
    Monitor GPU/CPU memory usage during training.
    
    Useful for optimizing batch sizes and detecting memory leaks.
    """
    
    def __init__(self,
                 log_frequency: int = 1,
                 monitor_gpu: bool = True,
                 alert_threshold: float = 0.9):
        """
        Initialize memory monitoring callback.
        
        Args:
            log_frequency: How often to log memory usage (every N epochs)
            monitor_gpu: Whether to monitor GPU memory
            alert_threshold: Memory usage threshold for alerts (0.0-1.0)
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.monitor_gpu = monitor_gpu
        self.alert_threshold = alert_threshold
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            try:
                if self.monitor_gpu and tf.config.list_physical_devices('GPU'):
                    # Get GPU memory info
                    gpus = tf.config.experimental.get_memory_info('GPU:0')
                    current_mb = gpus['current'] / (1024**2)
                    peak_mb = gpus['peak'] / (1024**2)
                    
                    print(f"Epoch {epoch + 1} - GPU Memory: Current={current_mb:.1f}MB, Peak={peak_mb:.1f}MB")
                    
                    # Alert if memory usage is high
                    if gpus['current'] / gpus['peak'] > self.alert_threshold:
                        print(f"WARNING: High GPU memory usage detected!")
                
                # Can also monitor CPU memory here using psutil if available
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")


class AdaptiveBatchSize(tf.keras.callbacks.Callback):
    """
    Dynamically adjust batch size during training based on loss progress.
    
    Note: This is a conceptual example - actual batch size changes during
    training require more complex dataset pipeline modifications.
    """
    
    def __init__(self,
                 initial_batch_size: int = 32,
                 max_batch_size: int = 128,
                 loss_patience: int = 3,
                 scale_factor: float = 1.5):
        """
        Initialize adaptive batch size callback.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            loss_patience: Epochs to wait before increasing batch size
            scale_factor: Factor by which to increase batch size
        """
        super().__init__()
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.loss_patience = loss_patience
        self.scale_factor = scale_factor
        self.current_batch_size = initial_batch_size
        self.wait = 0
        self.best_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('loss', np.inf)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        # If loss hasn't improved for patience epochs and we can increase batch size
        if (self.wait >= self.loss_patience and 
            self.current_batch_size < self.max_batch_size):
            
            new_batch_size = min(
                int(self.current_batch_size * self.scale_factor),
                self.max_batch_size
            )
            
            if new_batch_size != self.current_batch_size:
                print(f"\nEpoch {epoch + 1}: Increasing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
                self.wait = 0
                
                # Note: Actual implementation would require rebuilding the dataset
                # with the new batch size, which is complex and not shown here
