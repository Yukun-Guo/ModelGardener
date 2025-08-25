"""
Enhanced Trainer for ModelGardener

This module provides a comprehensive training system that supports:
1. Dataset loading from files/folders with custom data loaders
2. Model creation with custom models, loss functions, metrics, optimizers, callbacks
3. Training loop with default model.fit() or custom training loops
4. Progress tracking with detailed logging
"""

import os
import sys
import json
import importlib
import importlib.util
import contextlib
import io
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path

import tensorflow as tf
import keras
import numpy as np

try:
    from PySide6.QtCore import QObject, QThread, pyqtSignal
    PYSIDE6_AVAILABLE = True
except ImportError:
    # Fallback for environments without PySide6
    PYSIDE6_AVAILABLE = False
    class QObject:
        def __init__(self): pass
    class QThread:
        def __init__(self): pass
        def start(self): pass
        def isRunning(self): return False
        def wait(self, timeout=0): pass
        def terminate(self): pass
    def pyqtSignal(*args): return lambda: None

from bridge_callback import BRIDGE, QtBridgeCallback


class LogCapture:
    """Context manager to capture stdout/stderr and redirect to BRIDGE logging."""
    
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get captured output
        stdout_output = self.stdout_buffer.getvalue()
        stderr_output = self.stderr_buffer.getvalue()
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Log captured output
        if stdout_output:
            for line in stdout_output.strip().split('\n'):
                if line.strip():
                    BRIDGE.log.emit(f"{self.prefix}{line}")
        
        if stderr_output:
            for line in stderr_output.strip().split('\n'):
                if line.strip():
                    BRIDGE.log.emit(f"{self.prefix}[ERROR] {line}")


class DatasetLoader:
    """Handles dataset loading with custom data loaders and preprocessing."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        self.config = config
        self.custom_functions = custom_functions or {}
        
    def load_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """Load dataset for specified split (train/val)."""
        try:
            # Get data configuration
            data_config = self.config.get('data', {})
            
            # Check for custom data loader
            data_loader_config = data_config.get('data_loader', {})
            selected_loader = data_loader_config.get('selected_data_loader', 'Default')
            
            if selected_loader.startswith('Custom_'):
                return self._load_custom_dataset(split, data_loader_config)
            else:
                return self._load_builtin_dataset(split, data_config, selected_loader)
                
        except Exception as e:
            BRIDGE.log.emit(f"Error loading dataset: {str(e)}")
            raise
    
    def _load_custom_dataset(self, split: str, loader_config: Dict[str, Any]) -> tf.data.Dataset:
        """Load dataset using custom data loader."""
        selected_loader = loader_config.get('selected_data_loader', '')
        custom_loader_info = self.custom_functions.get('data_loaders', {}).get(selected_loader)
        
        if not custom_loader_info:
            raise ValueError(f"Custom data loader {selected_loader} not found")
        
        loader_func = custom_loader_info['loader']
        loader_type = custom_loader_info['type']
        
        # Prepare arguments from configuration
        args = self._prepare_loader_args(loader_config, split)
        
        try:
            if loader_type == 'function':
                dataset = loader_func(**args)
            elif loader_type == 'class':
                loader_instance = loader_func(**args)
                if hasattr(loader_instance, 'get_dataset'):
                    dataset = loader_instance.get_dataset()
                elif hasattr(loader_instance, '__call__'):
                    dataset = loader_instance()
                else:
                    raise ValueError(f"Custom loader class must have 'get_dataset' or '__call__' method")
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")
                
            if not isinstance(dataset, tf.data.Dataset):
                raise ValueError("Custom loader must return tf.data.Dataset")
                
            BRIDGE.log.emit(f"Loaded {split} dataset using custom loader: {selected_loader}")
            return dataset
            
        except Exception as e:
            BRIDGE.log.emit(f"Error in custom data loader: {str(e)}")
            raise
    
    def _load_builtin_dataset(self, split: str, data_config: Dict[str, Any], loader_type: str) -> tf.data.Dataset:
        """Load dataset using built-in loaders."""
        # Get directory path
        dir_key = f"{split}_dir" if split in ['train', 'val'] else 'train_dir'
        data_dir = data_config.get(dir_key, '')
        
        if not data_dir or not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Get batch size and other parameters
        batch_size = data_config.get('batch_size', 32)
        image_size = data_config.get('image_size', [224, 224])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
        BRIDGE.log.emit(f"Loading {split} dataset from: {data_dir}")
        
        if loader_type == 'ImageDataLoader':
            dataset = self._load_image_dataset(data_dir, batch_size, image_size, split)
        elif loader_type == 'TFRecordDataLoader':
            dataset = self._load_tfrecord_dataset(data_dir, batch_size)
        else:
            # Default image loading
            dataset = self._load_image_dataset(data_dir, batch_size, image_size, split)
        
        return dataset
    
    def _load_image_dataset(self, data_dir: str, batch_size: int, image_size: List[int], split: str) -> tf.data.Dataset:
        """Load image dataset from directory structure."""
        try:
            # Use tf.keras.utils.image_dataset_from_directory
            dataset = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=0.0,  # We assume separate train/val dirs
                subset=None,
                seed=42,
                image_size=tuple(image_size[:2]),
                batch_size=batch_size,
                label_mode='categorical'
            )
            
            # Apply preprocessing
            dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
            
            # Apply augmentation for training
            if split == 'train':
                dataset = self._apply_augmentation(dataset)
            
            # Performance optimizations
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            if split == 'train':
                dataset = dataset.shuffle(1000)
                
            return dataset
            
        except Exception as e:
            BRIDGE.log.emit(f"Error loading image dataset: {str(e)}")
            raise
    
    def _load_tfrecord_dataset(self, data_dir: str, batch_size: int) -> tf.data.Dataset:
        """Load TFRecord dataset."""
        try:
            # Find TFRecord files
            tfrecord_files = list(Path(data_dir).glob("*.tfrecord"))
            if not tfrecord_files:
                raise ValueError(f"No TFRecord files found in {data_dir}")
            
            # Create dataset
            dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
            
            # Parse TFRecord (this is a simplified example)
            def parse_tfrecord(example):
                features = {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                }
                parsed = tf.io.parse_single_example(example, features)
                image = tf.io.decode_jpeg(parsed['image'])
                image = tf.cast(image, tf.float32) / 255.0
                label = tf.cast(parsed['label'], tf.int32)
                return image, label
            
            dataset = dataset.map(parse_tfrecord)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
            
        except Exception as e:
            BRIDGE.log.emit(f"Error loading TFRecord dataset: {str(e)}")
            raise
    
    def _apply_augmentation(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply data augmentation to training dataset."""
        aug_config = self.config.get('augmentation', {})
        
        def augment(image, label):
            # Horizontal flip
            hflip = aug_config.get('Horizontal Flip', {})
            if hflip.get('enabled', False):
                prob = hflip.get('probability', 0.5)
                image = tf.cond(tf.random.uniform([]) < prob,
                               lambda: tf.image.flip_left_right(image),
                               lambda: image)
            
            # Random rotation
            rotation = aug_config.get('Random Rotation', {})
            if rotation.get('enabled', False):
                angle_range = rotation.get('angle_range', 15) * np.pi / 180
                angle = tf.random.uniform([], -angle_range, angle_range)
                # Use tf.image for rotation (simplified implementation)
                # For full rotation support, you'd need tfa.image.rotate or custom implementation
                image = tf.image.rot90(image, k=tf.cast(angle / (np.pi/2), tf.int32))  # Simplified
            
            # Brightness adjustment
            brightness = aug_config.get('Brightness Adjustment', {})
            if brightness.get('enabled', False):
                delta = brightness.get('brightness_limit', 0.2)
                image = tf.image.random_brightness(image, delta)
            
            # Ensure image values are in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    def _prepare_loader_args(self, loader_config: Dict[str, Any], split: str) -> Dict[str, Any]:
        """Prepare arguments for custom data loader."""
        args = {}
        
        # Add common parameters
        data_config = self.config.get('data', {})
        dir_key = f"{split}_dir" if split in ['train', 'val'] else 'train_dir'
        data_dir = data_config.get(dir_key, '')
        
        args['data_dir'] = data_dir
        args['split'] = split
        
        # Add loader-specific parameters from config
        for key, value in loader_config.items():
            if key not in ['selected_data_loader', 'use_for_train', 'use_for_val']:
                args[key] = value
        
        return args


class ModelBuilder:
    """Handles model creation with custom models support."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        self.config = config
        self.custom_functions = custom_functions or {}
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build model according to configuration."""
        try:
            model_config = self.config.get('model', {})
            
            # Check for custom model
            if 'custom_model_file_path' in model_config.get('model_parameters', {}):
                model = self._build_custom_model(input_shape, num_classes)
            else:
                model = self._build_builtin_model(input_shape, num_classes, model_config)
            
            # Compile model
            model = self._compile_model(model)
            
            BRIDGE.log.emit(f"Model built successfully: {model.name}")
            BRIDGE.log.emit(f"Model parameters: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            BRIDGE.log.emit(f"Error building model: {str(e)}")
            raise
    
    def _build_custom_model(self, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
        """Build custom model from user file."""
        model_params = self.config.get('model', {}).get('model_parameters', {})
        custom_model_path = model_params.get('custom_model_file_path', '')
        
        if not custom_model_path or not os.path.exists(custom_model_path):
            raise ValueError(f"Custom model file not found: {custom_model_path}")
        
        # Get custom models from loaded functions
        custom_models = self.custom_functions.get('models', {})
        
        if not custom_models:
            raise ValueError("No custom models loaded")
        
        # Use the first available custom model (can be enhanced to allow selection)
        model_name = list(custom_models.keys())[0]
        model_info = custom_models[model_name]
        
        model_func = model_info['function']
        
        # Prepare model arguments
        model_args = {
            'input_shape': input_shape,
            'num_classes': num_classes
        }
        
        # Add other parameters from config
        kwargs_str = model_params.get('kwargs', '{}')
        try:
            additional_kwargs = json.loads(kwargs_str) if kwargs_str else {}
            model_args.update(additional_kwargs)
        except json.JSONDecodeError:
            BRIDGE.log.emit("Warning: Invalid kwargs JSON, ignoring")
        
        # Build model
        if model_info['type'] == 'function':
            model = model_func(**model_args)
        elif model_info['type'] == 'class':
            model = model_func(**model_args)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
        
        if not isinstance(model, keras.Model):
            raise ValueError("Custom model must return keras.Model instance")
        
        BRIDGE.log.emit(f"Built custom model: {model_name}")
        return model
    
    def _build_builtin_model(self, input_shape: Tuple[int, ...], num_classes: int, model_config: Dict[str, Any]) -> keras.Model:
        """Build built-in model."""
        model_name = model_config.get('model_name', 'ResNet-50')
        
        BRIDGE.log.emit(f"Building built-in model: {model_name}")
        
        if 'ResNet' in model_name:
            return self._build_resnet(input_shape, num_classes, model_name)
        elif 'EfficientNet' in model_name:
            return self._build_efficientnet(input_shape, num_classes, model_name)
        elif 'VGG' in model_name:
            return self._build_vgg(input_shape, num_classes, model_name)
        else:
            # Default to ResNet-50
            return self._build_resnet(input_shape, num_classes, 'ResNet-50')
    
    def _build_resnet(self, input_shape: Tuple[int, ...], num_classes: int, model_name: str) -> keras.Model:
        """Build ResNet model."""
        # Map model names to Keras applications
        resnet_models = {
            'ResNet-50': tf.keras.applications.ResNet50,
            'ResNet-101': tf.keras.applications.ResNet101,
            'ResNet-152': tf.keras.applications.ResNet152,
        }
        
        ResNetClass = resnet_models.get(model_name, tf.keras.applications.ResNet50)
        
        base_model = ResNetClass(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        
        # Add custom head
        inputs = base_model.input
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name=model_name.lower().replace('-', '_'))
        return model
    
    def _build_efficientnet(self, input_shape: Tuple[int, ...], num_classes: int, model_name: str) -> keras.Model:
        """Build EfficientNet model."""
        efficientnet_models = {
            'EfficientNet-B0': tf.keras.applications.EfficientNetB0,
            'EfficientNet-B1': tf.keras.applications.EfficientNetB1,
            'EfficientNet-B2': tf.keras.applications.EfficientNetB2,
        }
        
        EfficientNetClass = efficientnet_models.get(model_name, tf.keras.applications.EfficientNetB0)
        
        base_model = EfficientNetClass(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        
        inputs = base_model.input
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name=model_name.lower().replace('-', '_'))
        return model
    
    def _build_vgg(self, input_shape: Tuple[int, ...], num_classes: int, model_name: str) -> keras.Model:
        """Build VGG model."""
        vgg_models = {
            'VGG-16': tf.keras.applications.VGG16,
            'VGG-19': tf.keras.applications.VGG19,
        }
        
        VGGClass = vgg_models.get(model_name, tf.keras.applications.VGG16)
        
        base_model = VGGClass(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        
        inputs = base_model.input
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name=model_name.lower().replace('-', '_'))
        return model
    
    def _compile_model(self, model: keras.Model) -> keras.Model:
        """Compile model with optimizer, loss, and metrics."""
        # Get optimizer
        optimizer = self._build_optimizer()
        
        # Get loss function
        loss_fn = self._build_loss_function()
        
        # Get metrics
        metrics = self._build_metrics()
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer from configuration."""
        optimizer_config = self.config.get('optimizer', {})
        
        # Check for custom optimizer
        custom_optimizers = self.custom_functions.get('optimizers', {})
        selected_optimizer = optimizer_config.get('selected_optimizer', 'Adam')
        
        if selected_optimizer.startswith('Custom_'):
            optimizer_info = custom_optimizers.get(selected_optimizer)
            if optimizer_info:
                return optimizer_info['function']()
        
        # Built-in optimizers
        training_config = self.config.get('training', {})
        learning_rate = training_config.get('initial_learning_rate', 0.001)
        
        if selected_optimizer == 'Adam':
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif selected_optimizer == 'SGD':
            momentum = training_config.get('momentum', 0.9)
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif selected_optimizer == 'RMSprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def _build_loss_function(self):
        """Build loss function from configuration."""
        loss_config = self.config.get('loss_functions', {})
        
        # Check for custom loss
        custom_losses = self.custom_functions.get('loss_functions', {})
        selected_loss = loss_config.get('selected_loss_function', 'categorical_crossentropy')
        
        if selected_loss.startswith('Custom_'):
            loss_info = custom_losses.get(selected_loss)
            if loss_info:
                return loss_info['function']
        
        # Built-in losses
        if selected_loss == 'categorical_crossentropy':
            return 'categorical_crossentropy'
        elif selected_loss == 'sparse_categorical_crossentropy':
            return 'sparse_categorical_crossentropy'
        elif selected_loss == 'binary_crossentropy':
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'
    
    def _build_metrics(self):
        """Build metrics from configuration."""
        metrics_config = self.config.get('metrics', {})
        
        # Check for custom metrics
        custom_metrics = self.custom_functions.get('metrics', {})
        
        metrics_list = []
        
        # Add standard metrics
        if metrics_config.get('accuracy', {}).get('enabled', True):
            metrics_list.append('accuracy')
        
        if metrics_config.get('top_5_accuracy', {}).get('enabled', False):
            metrics_list.append('top_k_categorical_accuracy')
        
        # Add custom metrics
        for metric_name, metric_info in custom_metrics.items():
            if metric_name.startswith('Custom_'):
                metrics_list.append(metric_info['function'])
        
        return metrics_list if metrics_list else ['accuracy']


class TrainingController(QThread):
    """Controls the training process with progress tracking."""
    
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    training_finished = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        super().__init__()
        self.config = config
        self.custom_functions = custom_functions or {}
        self.should_stop = False
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
    
    def stop_training(self):
        """Request training to stop."""
        self.should_stop = True
        BRIDGE.log.emit("Training stop requested...")
    
    def run(self):
        """Main training execution."""
        try:
            BRIDGE.log.emit("=== Starting Enhanced Training Process ===")
            
            # Step 1: Load datasets
            BRIDGE.log.emit("Step 1: Loading datasets...")
            self._load_datasets()
            
            # Step 2: Build model
            BRIDGE.log.emit("Step 2: Building model...")
            self._build_model()
            
            # Step 3: Setup callbacks
            BRIDGE.log.emit("Step 3: Setting up callbacks...")
            callbacks = self._setup_callbacks()
            
            # Step 4: Run training
            BRIDGE.log.emit("Step 4: Starting training loop...")
            self._run_training(callbacks)
            
            BRIDGE.log.emit("=== Training completed successfully ===")
            
        except Exception as e:
            BRIDGE.log.emit(f"Training failed with error: {str(e)}")
            import traceback
            BRIDGE.log.emit(f"Traceback: {traceback.format_exc()}")
        finally:
            BRIDGE.finished.emit()
    
    def _load_datasets(self):
        """Load training and validation datasets."""
        dataset_loader = DatasetLoader(self.config, self.custom_functions)
        
        # Load training dataset
        self.train_dataset = dataset_loader.load_dataset('train')
        BRIDGE.log.emit("Training dataset loaded successfully")
        
        # Load validation dataset if available
        data_config = self.config.get('data', {})
        if data_config.get('val_dir'):
            self.val_dataset = dataset_loader.load_dataset('val')
            BRIDGE.log.emit("Validation dataset loaded successfully")
        else:
            BRIDGE.log.emit("No validation dataset specified")
    
    def _build_model(self):
        """Build and compile model."""
        # Determine input shape and number of classes from data
        input_shape, num_classes = self._infer_data_specs()
        
        # Build model
        model_builder = ModelBuilder(self.config, self.custom_functions)
        self.model = model_builder.build_model(input_shape, num_classes)
        
        # Log model summary with proper capture
        try:
            with LogCapture("[MODEL] "):
                self.model.summary()
        except Exception:
            BRIDGE.log.emit("Could not print model summary")
    
    def _infer_data_specs(self) -> Tuple[Tuple[int, ...], int]:
        """Infer input shape and number of classes from datasets."""
        # Get a sample from training dataset
        for batch in self.train_dataset.take(1):
            if isinstance(batch, tuple) and len(batch) == 2:
                images, labels = batch
                input_shape = images.shape[1:]  # Remove batch dimension
                
                # Infer number of classes
                if len(labels.shape) > 1:
                    # One-hot encoded
                    num_classes = labels.shape[-1]
                else:
                    # Sparse labels
                    num_classes = tf.reduce_max(labels).numpy() + 1
                
                BRIDGE.log.emit(f"Inferred input shape: {input_shape}")
                BRIDGE.log.emit(f"Inferred number of classes: {num_classes}")
                
                return tuple(input_shape), int(num_classes)
        
        # Fallback defaults
        data_config = self.config.get('data', {})
        image_size = data_config.get('image_size', [224, 224])
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        
        input_shape = tuple(image_size + [3])  # Assume RGB
        num_classes = data_config.get('num_classes', 1000)
        
        BRIDGE.log.emit(f"Using default input shape: {input_shape}")
        BRIDGE.log.emit(f"Using default number of classes: {num_classes}")
        
        return input_shape, num_classes
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Add Qt bridge callback for progress tracking
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        # Calculate steps per epoch
        steps_per_epoch = None
        try:
            # Try to get dataset cardinality
            steps_per_epoch = tf.data.experimental.cardinality(self.train_dataset).numpy()
            if steps_per_epoch < 0:  # Unknown cardinality
                steps_per_epoch = None
        except Exception:
            pass
        
        if steps_per_epoch:
            total_steps = epochs * steps_per_epoch
        else:
            total_steps = epochs * 100  # Rough estimate
        
        qt_callback = QtBridgeCallback(total_train_steps=total_steps, log_every_n=10)
        callbacks.append(qt_callback)
        
        # Add model checkpoint callback
        runtime_config = self.config.get('runtime', {})
        model_dir = runtime_config.get('model_dir', './model_dir')
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'checkpoint-{epoch:03d}.keras'),
            save_best_only=True,
            monitor='val_loss' if self.val_dataset else 'loss',
            mode='min',
            save_freq='epoch'
        )
        callbacks.append(checkpoint_callback)
        
        # Add early stopping if validation data is available
        if self.val_dataset:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Add custom callbacks if available
        custom_callbacks = self.custom_functions.get('callbacks', {})
        for callback_name, callback_info in custom_callbacks.items():
            if callback_name.startswith('Custom_'):
                try:
                    custom_callback = callback_info['function']()
                    callbacks.append(custom_callback)
                    BRIDGE.log.emit(f"Added custom callback: {callback_name}")
                except Exception as e:
                    BRIDGE.log.emit(f"Error adding custom callback {callback_name}: {str(e)}")
        
        BRIDGE.log.emit(f"Setup {len(callbacks)} callbacks for training")
        return callbacks
    
    def _run_training(self, callbacks):
        """Run the actual training loop."""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        # Check for custom training loop
        training_loop_config = training_config.get('training_loop', {})
        selected_strategy = training_loop_config.get('selected_strategy', 'Standard Training')
        
        if selected_strategy.startswith('Custom_'):
            self._run_custom_training_loop(callbacks, epochs)
        else:
            self._run_standard_training_loop(callbacks, epochs)
    
    def _run_standard_training_loop(self, callbacks, epochs):
        """Run standard model.fit() training loop."""
        BRIDGE.log.emit(f"Starting standard training for {epochs} epochs")
        
        # Use log capture to redirect training output to logs
        with LogCapture("[TRAINING] "):
            history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        BRIDGE.log.emit("Standard training completed")
        
        # Save final model
        runtime_config = self.config.get('runtime', {})
        model_dir = runtime_config.get('model_dir', './model_dir')
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        self.model.save(final_model_path)
        BRIDGE.log.emit(f"Final model saved to: {final_model_path}")
    
    def _run_custom_training_loop(self, callbacks, epochs):
        """Run custom training loop."""
        selected_strategy = self.config.get('training', {}).get('training_loop', {}).get('selected_strategy', '')
        custom_loops = self.custom_functions.get('training_loops', {})
        
        loop_info = custom_loops.get(selected_strategy)
        if not loop_info:
            BRIDGE.log.emit(f"Custom training loop {selected_strategy} not found, falling back to standard training")
            self._run_standard_training_loop(callbacks, epochs)
            return
        
        BRIDGE.log.emit(f"Starting custom training loop: {selected_strategy}")
        
        try:
            custom_loop_func = loop_info['function']
            
            # Prepare arguments for custom training loop
            loop_args = {
                'model': self.model,
                'train_dataset': self.train_dataset,
                'val_dataset': self.val_dataset,
                'epochs': epochs,
                'callbacks': callbacks,
                'config': self.config
            }
            
            # Run custom training loop
            if loop_info['type'] == 'function':
                custom_loop_func(**loop_args)
            elif loop_info['type'] == 'class':
                loop_instance = custom_loop_func(**loop_args)
                if hasattr(loop_instance, 'run'):
                    loop_instance.run()
                elif hasattr(loop_instance, '__call__'):
                    loop_instance()
                else:
                    raise ValueError("Custom training loop class must have 'run' or '__call__' method")
            
            BRIDGE.log.emit("Custom training loop completed")
            
        except Exception as e:
            BRIDGE.log.emit(f"Error in custom training loop: {str(e)}")
            BRIDGE.log.emit("Falling back to standard training")
            self._run_standard_training_loop(callbacks, epochs)


class EnhancedTrainer(QObject):
    """Main enhanced trainer class that coordinates the training process."""
    
    def __init__(self, config: Dict[str, Any], custom_functions: Dict[str, Any] = None):
        super().__init__()
        self.config = config
        self.custom_functions = custom_functions or {}
        self.training_controller = None
    
    def start_training(self):
        """Start the training process."""
        if self.training_controller and self.training_controller.isRunning():
            BRIDGE.log.emit("Training is already running")
            return
        
        # Create and start training controller
        self.training_controller = TrainingController(self.config, self.custom_functions)
        self.training_controller.start()
    
    def stop_training(self):
        """Stop the training process."""
        if self.training_controller and self.training_controller.isRunning():
            self.training_controller.stop_training()
            self.training_controller.wait(5000)  # Wait up to 5 seconds for graceful shutdown
            if self.training_controller.isRunning():
                self.training_controller.terminate()
                BRIDGE.log.emit("Training forcefully terminated")
        else:
            BRIDGE.log.emit("No training process to stop")
    
    def is_training(self):
        """Check if training is currently running."""
        return self.training_controller and self.training_controller.isRunning()
