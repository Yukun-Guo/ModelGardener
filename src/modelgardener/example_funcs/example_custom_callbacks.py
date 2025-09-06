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

import tensorflow as tf
import keras

class MemoryUsageMonitor(keras.callbacks.Callback):
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


class CustomLearningRateScheduler(keras.callbacks.Callback):
    
    """
    custom learning rate scheduler that adjusts the learning rate based on epoch number.
    """
    def __init__(self):
        super().__init__()
        
    def on_batch_begin(self, batch, logs=None):
        return super().on_batch_begin(batch, logs)