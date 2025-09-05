"""
Training Components Builder for ModelGardener

This module handles the creation and configuration of training components
including callbacks, cross-validation, and training execution strategies.
"""

import os
import tensorflow as tf
import keras
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from bridge_callback import BRIDGE, CLIBridgeCallback


class TrainingComponentsBuilder:
    """Builds and configures training components like callbacks and validation strategies."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        self.config = config
        self.custom_functions = custom_functions or {}
        self.callbacks_config = config.get('callbacks', {})
        self.training_config = config.get('training', {})
        self.runtime_config = config.get('runtime', {})
    
    def setup_training_callbacks(self, total_steps: Optional[int] = None) -> List[keras.callbacks.Callback]:
        """
        Setup all training callbacks.
        
        Args:
            total_steps: Total number of training steps for progress tracking
            
        Returns:
            List[keras.callbacks.Callback]: List of configured callbacks
        """
        try:
            BRIDGE.log("=== Setting up Training Callbacks ===")
            
            callbacks = []
            
            # Add CLI bridge callback for progress tracking
            cli_callback = self._setup_cli_callback(total_steps)
            callbacks.append(cli_callback)
            
            # Setup standard callbacks
            callbacks.extend(self._setup_standard_callbacks())
            
            # Setup custom callbacks
            callbacks.extend(self._setup_custom_callbacks())
            
            BRIDGE.log(f"Setup {len(callbacks)} callbacks for training")
            return callbacks
            
        except Exception as e:
            BRIDGE.log(f"Error setting up callbacks: {str(e)}")
            raise
    
    def _setup_cli_callback(self, total_steps: Optional[int] = None) -> CLIBridgeCallback:
        """Setup CLI bridge callback for progress tracking."""
        
        epochs = self.training_config.get('epochs', 100)
        
        if total_steps:
            cli_callback = CLIBridgeCallback(total_train_steps=total_steps, log_every_n=5)
        else:
            # Estimate total steps
            estimated_steps = epochs * 100  # Rough estimate
            cli_callback = CLIBridgeCallback(total_train_steps=estimated_steps, log_every_n=5)
        
        return cli_callback
    
    def _setup_standard_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup standard Keras callbacks."""
        
        callbacks = []
        model_dir = self.runtime_config.get('model_dir', './logs')
        os.makedirs(model_dir, exist_ok=True)
        
        # Early Stopping
        early_stopping_config = self.callbacks_config.get('Early Stopping', {})
        if early_stopping_config.get('enabled', False):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True),
                verbose=1
            )
            callbacks.append(early_stopping)
            BRIDGE.log(f"Added Early Stopping callback (monitor: {early_stopping_config.get('monitor', 'val_loss')})")
        
        # Model Checkpoint
        checkpoint_config = self.callbacks_config.get('Model Checkpoint', {})
        if checkpoint_config.get('enabled', True):
            filepath = checkpoint_config.get('filepath', './logs/checkpoints/model-{epoch:02d}-{val_loss:.2f}.keras')
            
            # Ensure the filepath is relative to model_dir if it's not an absolute path
            if not os.path.isabs(filepath):
                filepath = os.path.join(model_dir, os.path.basename(filepath))
            
            # Ensure checkpoint directory exists
            checkpoint_dir = os.path.dirname(filepath)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_weights_only=checkpoint_config.get('save_weights_only', False),
                mode=checkpoint_config.get('mode', 'min'),
                save_freq='epoch',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            BRIDGE.log(f"Added Model Checkpoint callback (filepath: {filepath})")
        
        # TensorBoard
        tensorboard_config = self.callbacks_config.get('TensorBoard', {})
        if tensorboard_config.get('enabled', True):
            log_dir = tensorboard_config.get('log_dir', './logs/tensorboard')
            
            # Make path relative to model_dir if not absolute
            if not os.path.isabs(log_dir) and not log_dir.startswith('./'):
                log_dir = os.path.join(model_dir, log_dir)
            
            os.makedirs(log_dir, exist_ok=True)
            
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=tensorboard_config.get('histogram_freq', 1),
                write_graph=tensorboard_config.get('write_graph', True),
                write_images=tensorboard_config.get('write_images', False),
                update_freq=tensorboard_config.get('update_freq', 'epoch')
            )
            callbacks.append(tensorboard_callback)
            BRIDGE.log(f"Added TensorBoard callback (log_dir: {log_dir})")
        
        # CSV Logger
        csv_config = self.callbacks_config.get('CSV Logger', {})
        if csv_config.get('enabled', True):
            filename = csv_config.get('filename', './logs/training_log.csv')
            
            # Ensure the filename is relative to model_dir if it's not an absolute path
            if not os.path.isabs(filename):
                filename = os.path.join(model_dir, os.path.basename(filename))
            
            # Ensure log directory exists
            log_dir = os.path.dirname(filename)
            os.makedirs(log_dir, exist_ok=True)
            
            csv_callback = keras.callbacks.CSVLogger(
                filename=filename,
                separator=csv_config.get('separator', ','),
                append=csv_config.get('append', False)
            )
            callbacks.append(csv_callback)
            BRIDGE.log(f"Added CSV Logger callback (filename: {filename})")
        
        # Learning Rate Scheduler
        lr_config = self.callbacks_config.get('Learning Rate Scheduler', {})
        if lr_config.get('enabled', False):
            scheduler_type = lr_config.get('scheduler_type', 'ReduceLROnPlateau')
            
            if scheduler_type == 'ReduceLROnPlateau':
                lr_callback = keras.callbacks.ReduceLROnPlateau(
                    monitor=lr_config.get('monitor', 'val_loss'),
                    factor=lr_config.get('factor', 0.5),
                    patience=lr_config.get('patience', 5),
                    min_lr=lr_config.get('min_lr', 1e-7),
                    mode='min',
                    verbose=1
                )
                callbacks.append(lr_callback)
                BRIDGE.log("Added ReduceLROnPlateau callback")
            elif scheduler_type == 'ExponentialDecay':
                initial_lr = self.training_config.get('initial_learning_rate', 0.001)
                decay_rate = lr_config.get('decay_rate', 0.9)
                decay_steps = lr_config.get('decay_steps', 1000)
                
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate
                )
                
                # Note: This would need to be applied to the optimizer, not as a callback
                BRIDGE.log("Exponential decay schedule configured")
        
        return callbacks
    
    def _setup_custom_callbacks(self) -> List[keras.callbacks.Callback]:
        """Setup custom callbacks from user-defined functions."""
        
        callbacks = []
        custom_callbacks = self.custom_functions.get('callbacks', {})
        
        for callback_name, callback_info in custom_callbacks.items():
            if callback_name.startswith('Custom_'):
                try:
                    callback_func = callback_info['function']
                    callback_type = callback_info['type']
                    
                    if callback_type == 'function':
                        custom_callback = callback_func()
                    elif callback_type == 'class':
                        # Instantiate the class
                        custom_callback = callback_func()
                    else:
                        BRIDGE.log(f"Unknown callback type for {callback_name}: {callback_type}")
                        continue
                    
                    if isinstance(custom_callback, keras.callbacks.Callback):
                        callbacks.append(custom_callback)
                        BRIDGE.log(f"Added custom callback: {callback_name}")
                    else:
                        BRIDGE.log(f"Custom callback {callback_name} is not a valid Keras callback")
                        
                except Exception as e:
                    BRIDGE.log(f"Error adding custom callback {callback_name}: {str(e)}")
        
        return callbacks
    
    def create_cv_folds(self, dataset: tf.data.Dataset, k_folds: int = 5, 
                       stratified: bool = True) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            dataset: Full dataset to split
            k_folds: Number of folds
            stratified: Whether to use stratified splitting
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        try:
            BRIDGE.log(f"Creating {k_folds}-fold cross-validation splits (stratified: {stratified})")
            
            # Convert dataset to numpy arrays for splitting
            images, labels = self._dataset_to_arrays(dataset)
            
            # Setup cross-validation splitter
            cv_config = self.training_config.get('cross_validation', {})
            random_seed = cv_config.get('random_seed', 42)
            shuffle = cv_config.get('shuffle', True)
            
            if stratified and len(labels.shape) > 1:
                # Convert one-hot labels to sparse for stratification
                sparse_labels = np.argmax(labels, axis=1)
                splitter = StratifiedKFold(n_splits=k_folds, shuffle=shuffle, random_state=random_seed)
                splits = list(splitter.split(images, sparse_labels))
            else:
                splitter = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_seed)
                splits = list(splitter.split(images))
            
            # Create dataset folds
            folds = []
            for fold_idx, (train_indices, val_indices) in enumerate(splits):
                # Create train fold
                train_images = images[train_indices]
                train_labels = labels[train_indices]
                train_fold = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                
                # Create validation fold
                val_images = images[val_indices]
                val_labels = labels[val_indices]
                val_fold = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
                
                # Apply basic preprocessing (batching, prefetching)
                batch_size = self.config.get('data', {}).get('batch_size', 32)
                train_fold = train_fold.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                val_fold = val_fold.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                folds.append((train_fold, val_fold))
                BRIDGE.log(f"Fold {fold_idx + 1}: {len(train_indices)} train, {len(val_indices)} val samples")
            
            return folds
            
        except Exception as e:
            BRIDGE.log(f"Error creating CV folds: {str(e)}")
            raise
    
    def _dataset_to_arrays(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tf.data.Dataset to numpy arrays."""
        
        images_list = []
        labels_list = []
        
        # Iterate through dataset to collect all data
        for batch in dataset:
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_images, batch_labels = batch
                images_list.append(batch_images.numpy())
                labels_list.append(batch_labels.numpy())
            else:
                raise ValueError("Dataset must return (images, labels) tuples")
        
        # Concatenate all batches
        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return images, labels
    
    def log_cv_results(self, fold_results: List[List[float]]):
        """Log cross-validation results."""
        
        if not fold_results:
            return
        
        BRIDGE.log("=== Cross-Validation Results ===")
        
        # Assuming fold_results contains [loss, accuracy, ...] for each fold
        num_metrics = len(fold_results[0])
        metric_names = ['loss', 'accuracy'] + [f'metric_{i}' for i in range(2, num_metrics)]
        
        # Calculate statistics for each metric
        fold_results_array = np.array(fold_results)
        
        for i, metric_name in enumerate(metric_names):
            values = fold_results_array[:, i]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            BRIDGE.log(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Log individual fold results
            for fold_idx, value in enumerate(values):
                BRIDGE.log(f"  Fold {fold_idx + 1}: {value:.4f}")
    
    def should_use_cross_validation(self) -> bool:
        """Check if cross-validation should be used."""
        
        cv_config = self.training_config.get('cross_validation', {})
        return cv_config.get('enabled', False)
    
    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        
        cv_config = self.training_config.get('cross_validation', {})
        return {
            'k_folds': cv_config.get('k_folds', 5),
            'stratified': cv_config.get('stratified', True),
            'shuffle': cv_config.get('shuffle', True),
            'random_seed': cv_config.get('random_seed', 42),
            'save_fold_models': cv_config.get('save_fold_models', False)
        }
    
    def should_use_custom_training_loop(self) -> bool:
        """Check if custom training loop should be used."""
        
        training_loop_config = self.training_config.get('training_loop', {})
        selected_strategy = training_loop_config.get('selected_strategy', 'Standard Training')
        
        return selected_strategy.startswith('Custom_')
    
    def get_custom_training_loop_info(self) -> Optional[Dict[str, Any]]:
        """Get custom training loop information."""
        
        training_loop_config = self.training_config.get('training_loop', {})
        selected_strategy = training_loop_config.get('selected_strategy', '')
        
        if not selected_strategy.startswith('Custom_'):
            return None
        
        custom_loops = self.custom_functions.get('training_loops', {})
        return custom_loops.get(selected_strategy)
    
    def estimate_total_steps(self, dataset: tf.data.Dataset, epochs: int) -> Optional[int]:
        """Estimate total training steps for progress tracking."""
        
        try:
            # Try to get dataset cardinality
            cardinality = tf.data.experimental.cardinality(dataset).numpy()
            
            if cardinality > 0:
                return epochs * cardinality
            else:
                # Unknown cardinality, estimate based on batch size
                batch_size = self.config.get('data', {}).get('batch_size', 32)
                estimated_samples = 10000  # Rough estimate
                steps_per_epoch = estimated_samples // batch_size
                return epochs * steps_per_epoch
                
        except Exception:
            # Fallback estimation
            return epochs * 100
