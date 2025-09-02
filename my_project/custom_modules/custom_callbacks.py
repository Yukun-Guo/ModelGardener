"""
Custom Callbacks Template for ModelGardener

This file provides templates for creating custom training callbacks.
All callbacks should inherit from tf.keras.callbacks.Callback.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class EarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom early stopping callback with additional features.
    """
    
    def __init__(self, monitor='val_loss', patience=10, min_delta=0, 
                 restore_best_weights=True, baseline=None, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.verbose = verbose
        
        self.best = None
        self.wait = 0
        self.best_weights = None
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if 'loss' in self.monitor else -np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Check if improvement
        is_improvement = False
        if 'loss' in self.monitor:
            is_improvement = current < self.best - self.min_delta
        else:
            is_improvement = current > self.best + self.min_delta
        
        if is_improvement:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Custom learning rate scheduler with multiple scheduling strategies.
    """
    
    def __init__(self, schedule_type='exponential', initial_lr=0.001, 
                 decay_rate=0.95, decay_steps=1000, **kwargs):
        super().__init__()
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.kwargs = kwargs
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.schedule_type == 'exponential':
            lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_steps))
        elif self.schedule_type == 'cosine':
            lr = self.initial_lr * (1 + np.cos(np.pi * epoch / self.kwargs.get('max_epochs', 100))) / 2
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 30)
            lr = self.initial_lr * (self.decay_rate ** (epoch // step_size))
        else:
            lr = self.initial_lr
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"Learning rate for epoch {epoch + 1}: {lr:.6f}")


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Enhanced model checkpoint callback.
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True,
                 save_weights_only=False, mode='auto', save_freq='epoch', 
                 verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        if mode == 'auto':
            if 'loss' in monitor or monitor.startswith('val_'):
                self.mode = 'min'
            else:
                self.mode = 'max'
        
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Check if should save
        should_save = not self.save_best_only
        if self.save_best_only:
            if self.mode == 'min' and current < self.best:
                should_save = True
                self.best = current
            elif self.mode == 'max' and current > self.best:
                should_save = True
                self.best = current
        
        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
            
            if self.verbose > 0:
                print(f"\nSaved model to {filepath}")


class GradientClipping(tf.keras.callbacks.Callback):
    """
    Gradient clipping callback.
    """
    
    def __init__(self, clip_norm=1.0, clip_value=None):
        super().__init__()
        self.clip_norm = clip_norm
        self.clip_value = clip_value
    
    def on_train_begin(self, logs=None):
        # Add gradient clipping to optimizer
        if hasattr(self.model.optimizer, 'clipnorm'):
            if self.clip_norm is not None:
                self.model.optimizer.clipnorm = self.clip_norm
            if self.clip_value is not None:
                self.model.optimizer.clipvalue = self.clip_value


class MetricsLogger(tf.keras.callbacks.Callback):
    """
    Enhanced metrics logging callback.
    """
    
    def __init__(self, log_dir='./logs', log_freq=10):
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.history = {'loss': [], 'val_loss': []}
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        # Log metrics
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Save plot every log_freq epochs
        if (epoch + 1) % self.log_freq == 0:
            self._plot_metrics(epoch + 1)
    
    def _plot_metrics(self, epoch):
        """Plot and save metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        if 'loss' in self.history and 'val_loss' in self.history:
            axes[0, 0].plot(self.history['loss'], label='Training Loss')
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
        
        # Accuracy plot
        if 'accuracy' in self.history and 'val_accuracy' in self.history:
            axes[0, 1].plot(self.history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(self.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'metrics_epoch_{epoch}.png'))
        plt.close()


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Template for creating custom callbacks.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your custom parameters here
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        pass


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom callbacks...")
    
    # Test custom early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    print(f"Early Stopping: {early_stopping.__class__.__name__}")
    
    # Test custom learning rate scheduler
    lr_scheduler = LearningRateScheduler(schedule_type='exponential')
    print(f"LR Scheduler: {lr_scheduler.__class__.__name__}")
    
    # Test custom checkpoint
    checkpoint = ModelCheckpoint('./model_{epoch:02d}.h5')
    print(f"Model Checkpoint: {checkpoint.__class__.__name__}")
    
    print("✅ Custom callbacks template ready!")
