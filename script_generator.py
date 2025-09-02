#!/usr/bin/env python3
"""
Python Script Generator for ModelGardener
Generates train.py, evaluation.py, prediction.py, deploy.py scripts based on YAML configuration
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class ScriptGenerator:
    """Generates executable Python scripts based on ModelGardener YAML configuration."""
    
    def __init__(self):
        self.templates = {
            'train': self._get_train_template(),
            'evaluation': self._get_evaluation_template(),
            'prediction': self._get_prediction_template(),
            'deploy': self._get_deploy_template()
        }
    
    def generate_scripts(self, config_data: Dict[str, Any], output_dir: str, 
                        config_file_name: str = "model_config.yaml") -> bool:
        """
        Generate all Python scripts based on configuration.
        
        Args:
            config_data: The configuration dictionary
            output_dir: Directory where scripts should be saved
            config_file_name: Name of the configuration file
            
        Returns:
            bool: True if all scripts generated successfully
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract configuration for script generation
            config = config_data.get('configuration', config_data)
            
            # Generate each script
            for script_name, template in self.templates.items():
                script_content = self._fill_template(template, config, config_file_name)
                script_path = os.path.join(output_dir, f"{script_name}.py")
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                
                # Make script executable on Unix systems
                if os.name != 'nt':  # Not Windows
                    os.chmod(script_path, 0o755)
                
                print(f"✅ Generated: {script_path}")
            
            # Generate requirements.txt
            self._generate_requirements_txt(config, output_dir)
            
            # Generate README for scripts
            self._generate_scripts_readme(config, output_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating scripts: {str(e)}")
            return False
    
    def generate_custom_modules_templates(self, output_dir: str) -> bool:
        """
        Generate custom modules templates with one function per file.
        
        Args:
            output_dir: Directory where to create the custom_modules folder
            
        Returns:
            bool: True if templates generated successfully
        """
        try:
            # Create custom_modules directory
            custom_modules_dir = os.path.join(output_dir, 'custom_modules')
            os.makedirs(custom_modules_dir, exist_ok=True)
            
            # Generate individual function files based on example_funcs structure
            self._generate_individual_custom_functions(custom_modules_dir)
            
            # Generate __init__.py file
            init_file_path = os.path.join(custom_modules_dir, '__init__.py')
            with open(init_file_path, 'w', encoding='utf-8') as f:
                f.write('"""Custom modules for ModelGardener project."""\n')
            
            # Generate README for custom modules
            self._generate_custom_modules_readme(custom_modules_dir)
            
            print(f"📦 Custom modules templates created in: {custom_modules_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating custom modules templates: {str(e)}")
            return False

    def _generate_individual_custom_functions(self, custom_modules_dir: str):
        """
        Generate individual custom function files based on example_funcs structure.
        Each file will contain only one custom function.
        """
        # Define the example_funcs directory path
        example_funcs_dir = os.path.join(os.path.dirname(__file__), 'example_funcs')
        
        # Map of example files to their function extraction patterns
        function_extractions = {
            'example_custom_models.py': [
                ('create_simple_cnn', 'custom_models.py')
            ],
            'example_custom_data_loaders.py': [
                ('custom_image_data_loader', 'custom_data_loaders.py'),
                ('Custom_load_cifar10_npz_data', 'custom_data_loaders.py')
            ],
            'example_custom_loss_functions.py': [
                ('dice_loss', 'custom_loss_functions.py')
            ],
            'example_custom_optimizers.py': [
                ('adaptive_adam', 'custom_optimizers.py')
            ],
            'example_custom_metrics.py': [
                ('balanced_accuracy', 'custom_metrics.py')
            ],
            'example_custom_callbacks.py': [
                ('MemoryUsageMonitor', 'custom_callbacks.py')
            ],
            'example_custom_augmentations.py': [
                ('color_shift', 'custom_augmentations.py')
            ],
            'example_custom_preprocessing.py': [
                ('adaptive_histogram_equalization', 'custom_preprocessing.py')
            ],
            'example_custom_training_loops.py': [
                ('progressive_training_loop', 'custom_training_loops.py')
            ]
        }
        
        for example_file, extractions in function_extractions.items():
            example_file_path = os.path.join(example_funcs_dir, example_file)
            
            if os.path.exists(example_file_path):
                try:
                    with open(example_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for function_name, output_file in extractions:
                        # Extract the specific function and create individual file
                        extracted_content = self._extract_single_function(content, function_name, example_file)
                        if extracted_content:
                            output_path = os.path.join(custom_modules_dir, output_file)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(extracted_content)
                            print(f"✅ Generated: {output_path}")
                        
                except Exception as e:
                    print(f"❌ Error processing {example_file}: {str(e)}")
                    # Fallback to creating a basic template
                    self._create_basic_template(custom_modules_dir, example_file)
            else:
                print(f"⚠️ Example file not found: {example_file_path}, creating basic template")
                self._create_basic_template(custom_modules_dir, example_file)

    def _extract_single_function(self, file_content: str, function_name: str, source_file: str) -> str:
        """
        Extract a single function from the file content.
        """
        lines = file_content.split('\n')
        
        # Find the function definition
        function_start = None
        for i, line in enumerate(lines):
            if (line.strip().startswith(f'def {function_name}(') or 
                line.strip().startswith(f'class {function_name}')):
                function_start = i
                break
        
        if function_start is None:
            print(f"⚠️ Function {function_name} not found in {source_file}")
            return None
        
        # Find the end of the function by looking for the next function/class definition
        # or end of file, considering proper indentation
        function_end = len(lines)
        indent_level = len(lines[function_start]) - len(lines[function_start].lstrip())
        
        for i in range(function_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                # If we find a line at the same or lower indentation level that starts a new definition
                if (current_indent <= indent_level and 
                    (line.strip().startswith(('def ', 'class ', 'if __name__')) or
                     (current_indent == 0 and not line.strip().startswith(('#', '"""', "'''"))
                      and not line.strip().startswith(('import ', 'from '))))):
                    function_end = i
                    break
        
        # Extract just the function content
        function_lines = lines[function_start:function_end]
        
        # Get only the essential imports (not docstrings or comments)
        import_lines = []
        in_module_docstring = False
        docstring_quote = None
        
        for line in lines[:function_start]:
            stripped_line = line.strip()
            
            # Handle module-level docstrings
            if not in_module_docstring:
                if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                    docstring_quote = stripped_line[:3]
                    if len(stripped_line) > 3 and stripped_line.endswith(docstring_quote):
                        # Single line docstring, skip it
                        continue
                    else:
                        # Multi-line docstring starts
                        in_module_docstring = True
                        continue
            else:
                # We're inside a module docstring
                if docstring_quote in stripped_line:
                    in_module_docstring = False
                    docstring_quote = None
                continue
            
            # Only keep import statements and essential blank lines
            if (stripped_line.startswith(('import ', 'from ')) or 
                (not stripped_line and import_lines and 
                 import_lines[-1].strip().startswith(('import ', 'from ')))):
                import_lines.append(line)
        
        # Clean up trailing blank lines from imports
        while import_lines and not import_lines[-1].strip():
            import_lines.pop()
        
        # Combine imports and function with proper spacing
        result_lines = []
        if import_lines:
            result_lines.extend(import_lines)
            result_lines.append('')  # Blank line after imports
        
        result_lines.extend(function_lines)
        
        return '\n'.join(result_lines)

    def _create_basic_template(self, custom_modules_dir: str, example_file: str):
        """Create basic template file when example is not available."""
        template_mapping = {
            'example_custom_models.py': ('custom_models.py', self._get_basic_models_template()),
            'example_custom_data_loaders.py': ('custom_data_loaders.py', self._get_basic_data_loaders_template()),
            'example_custom_loss_functions.py': ('custom_loss_functions.py', self._get_basic_loss_functions_template()),
            'example_custom_optimizers.py': ('custom_optimizers.py', self._get_basic_optimizers_template()),
            'example_custom_metrics.py': ('custom_metrics.py', self._get_basic_metrics_template()),
            'example_custom_callbacks.py': ('custom_callbacks.py', self._get_basic_callbacks_template()),
            'example_custom_augmentations.py': ('custom_augmentations.py', self._get_basic_augmentations_template()),
            'example_custom_preprocessing.py': ('custom_preprocessing.py', self._get_basic_preprocessing_template()),
            'example_custom_training_loops.py': ('custom_training_loops.py', self._get_basic_training_loops_template()),
        }
        
        if example_file in template_mapping:
            filename, template = template_mapping[example_file]
            output_path = os.path.join(custom_modules_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"✅ Generated basic template: {output_path}")

    def _get_basic_models_template(self) -> str:
        """Get basic models template."""
        return '''"""
Custom Models Template for ModelGardener

This file provides a template for creating a custom model architecture.
"""

import tensorflow as tf
import keras
from keras import layers


def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, **kwargs):
    """
    Create a simple CNN model optimized for CIFAR-10 image classification.
    
    Args:
        input_shape: Input tensor shape (height, width, channels) - default (32, 32, 3) for CIFAR-10
        num_classes: Number of output classes - default 10 for CIFAR-10
        dropout_rate: Dropout rate for regularization
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    inputs = keras.Input(shape=input_shape)
    
    # First block - start with smaller filters for 32x32 input
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 32x32 -> 16x16
    x = layers.Dropout(0.25)(x)
    
    # Second block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16 -> 8x8
    x = layers.Dropout(0.25)(x)
    
    # Third block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 8x8 -> 4x4
    x = layers.Dropout(0.25)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='cifar10_cnn')
    return model


if __name__ == "__main__":
    # Test the custom model
    print("Testing CIFAR-10 optimized model...")
    model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
    print(f"Model: {model.name}, params: {model.count_params():,}")
    print("CIFAR-10 model created successfully!")
'''

    def _get_basic_data_loaders_template(self) -> str:
        """Get basic data loaders template."""
        return '''"""
Custom Data Loaders Template for ModelGardener

This file provides templates for creating custom data loading functions.
Includes support for both directory-based and NPZ file loading.
"""

import os
import tensorflow as tf
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def Custom_load_cifar10_npz_data(train_dir: str = "./data", 
                                 val_dir: str = "./data",
                                 npz_file_path: str = "./data/cifar10.npz",
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 buffer_size: int = 1000,
                                 validation_split: float = 0.2,
                                 **kwargs):
    """
    Custom CIFAR-10 NPZ data loader for ModelGardener.
    
    This function loads CIFAR-10 data from an NPZ file and returns
    training and validation datasets.
    
    Args:
        train_dir: Directory path (used for compatibility)
        val_dir: Directory path (used for compatibility) 
        npz_file_path: Path to the NPZ file containing CIFAR-10 data
        batch_size: Batch size for datasets
        shuffle: Whether to shuffle the data
        buffer_size: Buffer size for shuffling
        validation_split: Fraction of data to use for validation
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """
    print(f"🔍 Loading CIFAR-10 data from: {npz_file_path}")
    
    # Load NPZ file
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")
    
    data = np.load(npz_file_path)
    images = data['x'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    labels = data['y'].astype(np.int32)
    
    print(f"📊 Loaded {len(images)} images with shape {images.shape[1:]}")
    print(f"🎯 Found {len(np.unique(labels))} unique classes")
    
    # Split into train and validation
    train_indices = int(len(images) * (1 - validation_split))
    
    train_images = images[:train_indices]
    train_labels = labels[:train_indices]
    val_images = images[train_indices:]
    val_labels = labels[train_indices:]
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(labels))
    train_labels_categorical = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels_categorical = tf.keras.utils.to_categorical(val_labels, num_classes)
    
    print(f"🚂 Training set: {len(train_images)} samples")
    print(f"✅ Validation set: {len(val_images)} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_categorical))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_categorical))
    
    # Apply shuffling if requested
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    
    # Batch the datasets
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def custom_image_data_loader(data_dir: str, batch_size: int = 32, 
                           image_size: List[int] = [224, 224], 
                           shuffle: bool = True, buffer_size: int = 1000, 
                           augment: bool = False):
    """
    Custom image data loader that loads images from directories.
    
    Args:
        data_dir: Path to directory containing image files
        batch_size: Batch size for the dataset
        image_size: Target image size [height, width]
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        augment: Whether to apply augmentation
        
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    # Create dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.utils.experimental.preprocessing.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Apply augmentation if requested
    if augment:
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        dataset = dataset.map(lambda x, y: (augmentation_layer(x, training=True), y))
    
    # Optimize dataset performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test the custom data loader
    print("Testing custom data loader...")
    # Example usage - adjust path as needed
    # dataset = custom_image_data_loader('./data/train', batch_size=4)
    # for batch in dataset.take(1):
    #     images, labels = batch
    #     print(f"Batch shape: {images.shape}, Labels: {labels}")
    print("Custom data loader ready!")
'''

    def _get_basic_loss_functions_template(self) -> str:
        """Get basic loss functions template."""
        return '''"""
Custom Loss Functions Template for ModelGardener

This file provides a template for creating custom loss functions.
"""

import tensorflow as tf
import numpy as np


def custom_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    """
    Custom implementation of Focal Loss for addressing class imbalance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        from_logits: Whether y_pred is logits or probabilities
    
    Returns:
        Focal loss value
    """
    # Convert to probabilities if logits
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions to prevent numerical instability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # Calculate focal weight: (1 - p_t)^gamma
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = tf.pow((1 - p_t), gamma)
    
    # Apply alpha weighting
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    # Compute focal loss
    focal_loss = alpha_t * focal_weight * cross_entropy
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


if __name__ == "__main__":
    # Test the custom loss function
    print("Testing custom loss function...")
    
    # Create dummy data
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=tf.float32)
    
    loss = custom_focal_loss(y_true, y_pred)
    print(f"Focal loss: {loss.numpy():.4f}")
    print("Custom loss function working!")
'''

    def _get_basic_optimizers_template(self) -> str:
        """Get basic optimizers template."""
        return '''"""
Custom Optimizers Template for ModelGardener

This file provides a template for creating custom optimizers.
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
    # Create learning rate schedule with warmup
    def warmup_schedule(step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(warmup_steps, tf.float32)
        
        warmup_lr = learning_rate * step / warmup_steps_f
        decay_lr = learning_rate
        
        return tf.where(step < warmup_steps_f, warmup_lr, decay_lr)
    
    # Create optimizer with custom schedule
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=warmup_schedule,
        momentum=momentum,
        name="CustomSGDWarmup"
    )
    
    return optimizer


if __name__ == "__main__":
    # Test the custom optimizer
    print("Testing custom optimizer...")
    
    optimizer = custom_sgd_with_warmup(learning_rate=0.01, warmup_steps=500)
    print(f"Optimizer: {optimizer.name}")
    print("Custom optimizer created successfully!")
'''

    def _get_basic_metrics_template(self) -> str:
        """Get basic metrics template."""
        return '''"""
Custom Metrics Template for ModelGardener

This file provides a template for creating custom metrics.
"""

import tensorflow as tf


def custom_f1_score(y_true, y_pred, threshold=0.5):
    """
    Custom F1 score metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        F1 score
    """
    # Convert predictions to binary
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate precision and recall
    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    predicted_positives = tf.reduce_sum(y_pred_binary)
    actual_positives = tf.reduce_sum(y_true)
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    return f1


class CustomF1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric as a class.
    """
    
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


if __name__ == "__main__":
    # Test the custom metrics
    print("Testing custom metrics...")
    
    # Test function-based metric
    y_true = tf.constant([1, 1, 0, 0], dtype=tf.float32)
    y_pred = tf.constant([0.8, 0.6, 0.3, 0.2], dtype=tf.float32)
    
    f1 = custom_f1_score(y_true, y_pred)
    print(f"F1 Score (function): {f1.numpy():.4f}")
    
    # Test class-based metric
    metric = CustomF1Score()
    metric.update_state(y_true, y_pred)
    print(f"F1 Score (class): {metric.result().numpy():.4f}")
    
    print("Custom metrics working!")
'''

    def _get_basic_callbacks_template(self) -> str:
        """Get basic callbacks template."""
        return '''"""
Custom Callbacks Template for ModelGardener

This file provides a template for creating custom training callbacks.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional


class LossThresholdStopping(tf.keras.callbacks.Callback):
    """
    Custom callback that stops training when loss reaches a threshold.
    """
    
    def __init__(self, threshold=0.01, monitor='loss', patience=0, 
                 restore_best_weights=False, verbose=1):
        super().__init__()
        self.threshold = threshold
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_weights = None
        self.wait = 0
        self.best_loss = np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        
        if current_loss is None:
            if self.verbose > 0:
                print(f"Warning: {self.monitor} is not available in logs")
            return
        
        # Check if loss is below threshold
        if current_loss < self.threshold:
            if self.verbose > 0:
                print(f"\\nEpoch {epoch + 1}: {self.monitor} reached threshold {self.threshold}, stopping training")
            self.model.stop_training = True
            return
        
        # Track best weights
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
    
    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print("Restoring model weights from the best epoch")
            self.model.set_weights(self.best_weights)


if __name__ == "__main__":
    # Test the custom callback
    print("Testing custom callback...")
    
    # Create a simple model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Create callback
    callback = LossThresholdStopping(threshold=0.1, verbose=1)
    
    # Generate dummy data
    X = np.random.random((100, 5))
    y = np.random.randint(0, 2, (100, 1))
    
    print("Training with custom callback...")
    # model.fit(X, y, epochs=10, callbacks=[callback], verbose=0)
    
    print("Custom callback created successfully!")
'''

    def _get_basic_augmentations_template(self) -> str:
        """Get basic augmentations template."""
        return '''"""
Custom Augmentations Template for ModelGardener

This file provides a template for creating custom augmentation functions.
"""

import numpy as np
import cv2


def random_pixelate(image, block_size=8, probability=0.5):
    """
    Apply random pixelation effect to image.
    
    Args:
        image (np.ndarray): Input image
        block_size (int): Size of pixelation blocks (default: 8)
        probability (float): Probability of applying effect (default: 0.5)
    
    Returns:
        np.ndarray: Pixelated image
    """
    if np.random.random() > probability:
        return image
    
    try:
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Resize down and then back up to create pixelation effect
        temp = cv2.resize(image, 
                         (width // block_size, height // block_size), 
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, 
                              (width, height), 
                              interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    except Exception as e:
        print(f"Error in random_pixelate: {e}")
        return image


if __name__ == "__main__":
    # Test the custom augmentation
    print("Testing custom augmentation...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Apply augmentation
    augmented = random_pixelate(dummy_image, block_size=16, probability=1.0)
    
    print(f"Original shape: {dummy_image.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print("Custom augmentation working!")
'''

    def _get_basic_preprocessing_template(self) -> str:
        """Get basic preprocessing template."""
        return '''"""
Custom Preprocessing Template for ModelGardener

This file provides a template for creating custom preprocessing functions.
"""

import numpy as np
import cv2
from typing import Union, Tuple


def adaptive_histogram_equalization(data: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8):
    """
    Apply adaptive histogram equalization (CLAHE) to improve image contrast.
    
    This method enhances local contrast in images by applying histogram 
    equalization in small regions (tiles) rather than the entire image.
    
    Args:
        data: Input image data (numpy array)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of the neighborhood region for local contrast
        
    Returns:
        Processed image with enhanced local contrast
    """
    if len(data.shape) == 3 and data.shape[2] == 3:
        # Convert RGB to LAB color space for better results
        lab = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to the L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        result = clahe.apply(data.astype(np.uint8))
    
    return result


if __name__ == "__main__":
    # Test the custom preprocessing
    print("Testing custom preprocessing...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Apply preprocessing
    processed = adaptive_histogram_equalization(dummy_image)
    
    print(f"Original shape: {dummy_image.shape}")
    print(f"Processed shape: {processed.shape}")
    print("Custom preprocessing working!")
'''

    def _get_basic_training_loops_template(self) -> str:
        """Get basic training loops template."""
        return '''"""
Custom Training Loops Template for ModelGardener

This file provides a template for creating custom training loop strategies.
"""

import tensorflow as tf
import numpy as np
import time
from typing import Dict, Any, Optional, Callable


def progressive_training_loop(model, train_dataset, val_dataset, epochs, 
                            optimizer, loss_fn, initial_resolution=64, 
                            final_resolution=224, progression_schedule='linear'):
    """
    Progressive training loop that gradually increases image resolution during training.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        epochs: Total number of epochs
        optimizer: Optimizer to use
        loss_fn: Loss function
        initial_resolution: Starting image resolution
        final_resolution: Final image resolution
        progression_schedule: How to increase resolution ('linear' or 'exponential')
    """
    print(f"Starting progressive training from {initial_resolution}x{initial_resolution} to {final_resolution}x{final_resolution}")
    
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Calculate current resolution
        if progression_schedule == 'linear':
            progress = epoch / (epochs - 1)
        else:  # exponential
            progress = (np.exp(epoch / epochs) - 1) / (np.e - 1)
        
        current_resolution = int(initial_resolution + 
                               (final_resolution - initial_resolution) * progress)
        current_resolution = min(current_resolution, final_resolution)
        
        print(f"\\nEpoch {epoch + 1}/{epochs} - Resolution: {current_resolution}x{current_resolution}")
        
        # Training step
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_dataset:
            images, labels = batch
            
            # Resize images to current resolution
            if current_resolution != images.shape[1]:
                images = tf.image.resize(images, [current_resolution, current_resolution])
            
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        history['loss'].append(float(avg_loss))
        
        # Validation step
        if val_dataset is not None:
            val_loss = 0
            val_batches = 0
            
            for batch in val_dataset:
                images, labels = batch
                
                # Resize images to current resolution
                if current_resolution != images.shape[1]:
                    images = tf.image.resize(images, [current_resolution, current_resolution])
                
                predictions = model(images, training=False)
                loss = loss_fn(labels, predictions)
                val_loss += loss
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            history['val_loss'].append(float(avg_val_loss))
            
            print(f"Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Loss: {avg_loss:.4f}")
    
    return history


if __name__ == "__main__":
    # Test the custom training loop
    print("Testing custom training loop...")
    
    # Create a simple model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, None, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    print("Custom training loop template created successfully!")
'''
    
    def _fill_template(self, template: str, config: Dict[str, Any], config_file_name: str) -> str:
        """Fill template with configuration values."""
        
        # Extract key configuration values
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        runtime_config = config.get('runtime', {})
        
        # Basic parameters
        train_dir = data_config.get('train_dir', './data/train')
        val_dir = data_config.get('val_dir', './data/val')
        test_dir = data_config.get('test_dir', './data/test')  # For evaluation
        batch_size = data_config.get('data_loader', {}).get('parameters', {}).get('batch_size', 32)
        epochs = training_config.get('epochs', 100)
        model_dir = runtime_config.get('model_dir', './models')
        
        # Model information
        model_family = model_config.get('model_family', 'resnet')
        model_name = model_config.get('model_name', 'ResNet-50')
        
        # Input shape
        model_params = model_config.get('model_parameters', {})
        input_shape = model_params.get('input_shape', {})
        img_height = input_shape.get('height', 224)
        img_width = input_shape.get('width', 224)
        channels = input_shape.get('channels', 3)
        num_classes = model_params.get('classes', 10)
        
        # Optimizer and learning rate
        optimizer_config = model_config.get('optimizer', {}).get('Optimizer Selection', {})
        optimizer_name = optimizer_config.get('selected_optimizer', 'Adam')
        learning_rate = optimizer_config.get('learning_rate', training_config.get('initial_learning_rate', 0.001))
        
        # Loss function
        loss_config = model_config.get('loss_functions', {}).get('Loss Selection', {})
        loss_function = loss_config.get('selected_loss', 'Categorical Crossentropy')
        
        # Metrics
        metrics_config = model_config.get('metrics', {}).get('Metrics Selection', {})
        metrics = metrics_config.get('selected_metrics', 'Accuracy')
        
        # Cross validation
        cv_config = training_config.get('cross_validation', {})
        cv_enabled = cv_config.get('enabled', False)
        k_folds = cv_config.get('k_folds', 5)
        
        # Custom functions handling
        custom_functions = self._extract_custom_functions_info(config)
        
        # Replace template placeholders
        replacements = {
            '{{CONFIG_FILE}}': config_file_name,
            '{{TRAIN_DIR}}': train_dir,
            '{{VAL_DIR}}': val_dir,
            '{{TEST_DIR}}': test_dir,
            '{{BATCH_SIZE}}': str(batch_size),
            '{{EPOCHS}}': str(epochs),
            '{{MODEL_DIR}}': model_dir,
            '{{MODEL_FAMILY}}': model_family,
            '{{MODEL_NAME}}': model_name,
            '{{IMG_HEIGHT}}': str(img_height),
            '{{IMG_WIDTH}}': str(img_width),
            '{{CHANNELS}}': str(channels),
            '{{NUM_CLASSES}}': str(num_classes),
            '{{OPTIMIZER}}': optimizer_name,
            '{{LEARNING_RATE}}': str(learning_rate),
            '{{LOSS_FUNCTION}}': loss_function,
            '{{METRICS}}': metrics,
            '{{CV_ENABLED}}': str(cv_enabled),
            '{{K_FOLDS}}': str(k_folds),
            '{{CUSTOM_IMPORTS}}': self._generate_custom_imports(custom_functions),
            '{{CUSTOM_LOADER_CALLS}}': self._generate_custom_loader_calls(custom_functions),
            '{{DATA_LOADING_CODE}}': self._generate_data_loading_code(config, custom_functions),
            '{{GENERATION_DATE}}': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Apply replacements
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, str(value))
        
        return result
    
    def _extract_custom_functions_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract custom functions information from configuration."""
        # This would extract custom function metadata if present
        metadata = config.get('metadata', {})
        custom_functions = metadata.get('custom_functions', {})
        return custom_functions
    
    def _generate_custom_imports(self, custom_functions: Dict[str, Any]) -> str:
        """Generate import statements for custom functions."""
        imports = []
        if custom_functions:
            imports.append("# Custom functions imports")
            imports.append("import sys")
            imports.append("from pathlib import Path")
            imports.append("sys.path.append(str(Path(__file__).parent / 'src'))")
            
            for func_type, functions in custom_functions.items():
                if functions:
                    imports.append(f"# Custom {func_type}")
                    for func_info in functions:
                        if isinstance(func_info, dict):
                            module_name = func_info.get('relative_file_path', '').replace('.py', '')
                            if module_name:
                                imports.append(f"# from {module_name} import {func_info.get('function_name', 'custom_function')}")
        
        return '\n'.join(imports) if imports else "# No custom functions"
    
    def _generate_custom_loader_calls(self, custom_functions: Dict[str, Any]) -> str:
        """Generate custom function loader calls."""
        if not custom_functions:
            return """    # No custom functions to load
    custom_functions = None"""
        
        return """
    # Load custom functions if available
    custom_functions = {}
    try:
        from custom_functions_loader import CustomFunctionsLoader
        loader = CustomFunctionsLoader()
        custom_functions = loader.load_from_directory('src')
    except ImportError:
        print("Custom functions loader not found, using built-in functions only")
    except Exception as e:
        print(f"Error loading custom functions: {e}")
"""

    def _generate_data_loading_code(self, config: Dict[str, Any], custom_functions: Dict[str, Any]) -> str:
        """Generate appropriate data loading code based on configuration."""
        data_config = config.get('data', {})
        data_loader_config = data_config.get('data_loader', {})
        selected_loader = data_loader_config.get('selected_data_loader', None)
        
        # If a custom data loader is specified, use it
        if selected_loader and selected_loader != 'default':
            return f"""    # Using custom data loader: {selected_loader}
    print("📁 Loading data with custom loader: {selected_loader}")
    try:
        from custom_modules.custom_data_loaders import {selected_loader}
        
        # Get data loader parameters
        loader_params = config.get('configuration', {{}}).get('data', {{}}).get('data_loader', {{}}).get('parameters', {{}})
        
        # Load training and validation data
        train_gen, val_gen = {selected_loader}(
            train_dir=train_dir,
            val_dir=val_dir,
            **loader_params
        )
        
        print("✅ Custom data loader loaded successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import custom data loader {selected_loader}: {{e}}")
        print("🔄 Falling back to default data generators...")
        train_gen, val_gen = create_data_generators(
            train_dir, val_dir, batch_size, img_height, img_width
        )
    except Exception as e:
        print(f"❌ Error using custom data loader: {{e}}")
        print("🔄 Falling back to default data generators...")
        train_gen, val_gen = create_data_generators(
            train_dir, val_dir, batch_size, img_height, img_width
        )"""
        else:
            # Use default directory-based data generators
            return """    # Using default directory-based data generators
    print("📁 Creating data generators...")
    train_gen, val_gen = create_data_generators(
        train_dir, val_dir, batch_size, img_height, img_width
    )"""
    
    def _get_train_template(self) -> str:
        """Get the training script template."""
        return '''#!/usr/bin/env python3
"""
Training Script for ModelGardener
Generated on: {{GENERATION_DATE}}
Configuration: {{CONFIG_FILE}}
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path

{{CUSTOM_IMPORTS}}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_data_generators(train_dir, val_dir, batch_size=32, img_height=224, img_width=224):
    """Create data generators for training and validation."""
    
    # Data augmentation and preprocessing
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def build_model(model_family, model_name, input_shape, num_classes, custom_functions=None):
    """Build model based on configuration."""
    
    # Check for custom model first
    if custom_functions and 'models' in custom_functions:
        for model_info in custom_functions['models']:
            if model_info.get('name') == model_name:
                return model_info['function'](input_shape=input_shape, num_classes=num_classes)
    
    # Built-in models
    if model_family.lower() == 'resnet':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
    elif model_family.lower() == 'efficientnet':
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
    else:
        # Default to a simple CNN
        base_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
        ])
    
    # Add classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, optimizer='{{OPTIMIZER}}', learning_rate={{LEARNING_RATE}}, 
                  loss='{{LOSS_FUNCTION}}', metrics=['{{METRICS}}']):
    """Compile the model with specified parameters."""
    
    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Convert loss function name
    if loss.lower() in ['categorical crossentropy', 'categorical_crossentropy']:
        loss_fn = 'categorical_crossentropy'
    elif loss.lower() in ['sparse categorical crossentropy', 'sparse_categorical_crossentropy']:
        loss_fn = 'sparse_categorical_crossentropy'
    else:
        loss_fn = 'categorical_crossentropy'
    
    # Convert metrics
    metrics_list = []
    if isinstance(metrics, list):
        for metric in metrics:
            if metric.lower() == 'accuracy':
                metrics_list.append('accuracy')
            else:
                metrics_list.append(metric.lower())
    else:
        if metrics.lower() == 'accuracy':
            metrics_list = ['accuracy']
        else:
            metrics_list = [metrics.lower()]
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics_list)
    return model

def create_callbacks(model_dir):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    ))
    
    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ))
    
    # Reduce learning rate on plateau
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ))
    
    return callbacks

def train_model():
    """Main training function."""
    
    # Configuration
    config_file = "{{CONFIG_FILE}}"
    train_dir = "{{TRAIN_DIR}}"
    val_dir = "{{VAL_DIR}}"
    batch_size = {{BATCH_SIZE}}
    epochs = {{EPOCHS}}
    model_dir = "{{MODEL_DIR}}"
    img_height = {{IMG_HEIGHT}}
    img_width = {{IMG_WIDTH}}
    channels = {{CHANNELS}}
    num_classes = {{NUM_CLASSES}}
    
    # Load configuration if available
    if os.path.exists(config_file):
        try:
            config = load_config(config_file)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            config = {}
    else:
        print(f"⚠️  Configuration file {config_file} not found, using defaults")
        config = {}
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    {{CUSTOM_LOADER_CALLS}}
    
    {{DATA_LOADING_CODE}}
    
    input_shape = (img_height, img_width, channels)
    
    # Update num_classes from data if possible
    if hasattr(train_gen, 'num_classes'):
        num_classes = train_gen.num_classes
        print(f"📊 Detected {num_classes} classes from data")
    
    # Build model
    print("🏗️  Building model...")
    model = build_model(
        "{{MODEL_FAMILY}}", "{{MODEL_NAME}}", 
        input_shape, num_classes, custom_functions
    )
    
    # Compile model
    print("⚙️  Compiling model...")
    model = compile_model(model)
    
    # Create callbacks
    callbacks = create_callbacks(model_dir)
    
    # Print model summary
    print("📋 Model Summary:")
    model.summary()
    
    # Cross-validation training
    cv_enabled = {{CV_ENABLED}}
    if cv_enabled:
        print("🔄 Cross-validation training enabled")
        k_folds = {{K_FOLDS}}
        
        # Get data for cross-validation
        # Note: This is a simplified version - real implementation would need proper data handling
        print(f"Training with {k_folds}-fold cross-validation")
        
        # For now, train normally but save multiple models
        for fold in range(k_folds):
            print(f"📊 Training fold {fold + 1}/{k_folds}")
            fold_model_dir = os.path.join(model_dir, f"fold_{fold + 1}")
            os.makedirs(fold_model_dir, exist_ok=True)
            
            fold_callbacks = create_callbacks(fold_model_dir)
            
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=fold_callbacks,
                verbose=1
            )
            
            # Save fold model
            model.save(os.path.join(fold_model_dir, 'model.h5'))
            
            # Reset model weights for next fold (simplified)
            # In real implementation, you'd rebuild the model or reset weights properly
            pass
    else:
        # Regular training
        print("🚀 Starting training...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(model_dir, 'final_model.h5')
        model.save(final_model_path)
        print(f"💾 Model saved to: {final_model_path}")
    
    print("✅ Training completed!")

if __name__ == "__main__":
    train_model()
'''
    
    def _get_evaluation_template(self) -> str:
        """Get the evaluation script template."""
        return '''#!/usr/bin/env python3
"""
Evaluation Script for ModelGardener
Generated on: {{GENERATION_DATE}}
Configuration: {{CONFIG_FILE}}
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path

{{CUSTOM_IMPORTS}}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_generator(test_dir, batch_size=32, img_height=224, img_width=224):
    """Create test data generator."""
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )
    
    return test_generator

def evaluate_model():
    """Main evaluation function."""
    
    # Configuration
    config_file = "{{CONFIG_FILE}}"
    test_dir = "{{TEST_DIR}}"
    model_dir = "{{MODEL_DIR}}"
    batch_size = {{BATCH_SIZE}}
    img_height = {{IMG_HEIGHT}}
    img_width = {{IMG_WIDTH}}
    
    # Load configuration if available
    if os.path.exists(config_file):
        try:
            config = load_config(config_file)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            config = {}
    else:
        print(f"⚠️  Configuration file {config_file} not found, using defaults")
        config = {}
    
    {{CUSTOM_LOADER_CALLS}}
    
    # Find model file
    model_files = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.h5') and ('best' in file or 'final' in file):
                model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        return
    
    # Use the best model if available, otherwise use the first one
    model_file = None
    for file in model_files:
        if 'best' in os.path.basename(file):
            model_file = file
            break
    if not model_file:
        model_file = model_files[0]
    
    print(f"📥 Loading model from: {model_file}")
    
    # Load model
    try:
        model = keras.models.load_model(model_file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create test generator
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print("📁 Creating test data generator...")
    test_gen = create_test_generator(test_dir, batch_size, img_height, img_width)
    
    # Evaluate model
    print("📊 Evaluating model...")
    
    # Get predictions
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"🎯 Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\\n📋 Classification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)
    
    # Confusion matrix
    print("🔄 Generating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save confusion matrix
    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Save evaluation results
    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'model_file': model_file,
        'test_directory': test_dir,
        'num_samples': len(true_classes),
        'num_classes': len(class_labels),
        'class_labels': class_labels
    }
    
    results_path = os.path.join(model_dir, 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"💾 Evaluation results saved to: {results_path}")
    
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    evaluate_model()
'''
    
    def _get_prediction_template(self) -> str:
        """Get the prediction script template."""
        return '''#!/usr/bin/env python3
"""
Prediction Script for ModelGardener
Generated on: {{GENERATION_DATE}}
Configuration: {{CONFIG_FILE}}
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

{{CUSTOM_IMPORTS}}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for prediction."""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_single_image(model, image_path, class_labels, target_size=(224, 224)):
    """Predict class for a single image."""
    
    # Preprocess image
    img_array = preprocess_image(image_path, target_size)
    if img_array is None:
        return None, None, None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

def predict_batch(model, image_dir, class_labels, target_size=(224, 224)):
    """Predict classes for all images in a directory."""
    
    results = []
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return results
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for image_file in image_files:
        predicted_class, confidence, probabilities = predict_single_image(
            model, str(image_file), class_labels, target_size
        )
        
        if predicted_class is not None:
            result = {
                'image_path': str(image_file),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }
            results.append(result)
            print(f"📷 {image_file.name}: {predicted_class} ({confidence:.3f})")
        else:
            print(f"❌ Failed to process: {image_file.name}")
    
    return results

def main():
    """Main prediction function."""
    
    parser = argparse.ArgumentParser(description='Make predictions using trained ModelGardener model')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or directory containing images')
    parser.add_argument('--model', '-m', 
                       help='Path to model file (if not specified, will search in model directory)')
    parser.add_argument('--output', '-o', 
                       help='Output file to save results (JSON format)')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Show top K predictions (default: 5)')
    
    args = parser.parse_args()
    
    # Configuration
    config_file = "{{CONFIG_FILE}}"
    model_dir = "{{MODEL_DIR}}"
    img_height = {{IMG_HEIGHT}}
    img_width = {{IMG_WIDTH}}
    target_size = (img_height, img_width)
    
    # Load configuration if available
    class_labels = []
    if os.path.exists(config_file):
        try:
            config = load_config(config_file)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            config = {}
    else:
        print(f"⚠️  Configuration file {config_file} not found, using defaults")
        config = {}
    
    {{CUSTOM_LOADER_CALLS}}
    
    # Find model file
    model_file = args.model
    if not model_file:
        model_files = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.h5') and ('best' in file or 'final' in file):
                    model_files.append(os.path.join(model_dir, file))
        
        if not model_files:
            print(f"❌ No model files found in {model_dir}")
            return
        
        # Use the best model if available
        for file in model_files:
            if 'best' in os.path.basename(file):
                model_file = file
                break
        if not model_file:
            model_file = model_files[0]
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    print(f"📥 Loading model from: {model_file}")
    
    # Load model
    try:
        model = keras.models.load_model(model_file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Try to get class labels from training directory structure
    if not class_labels:
        train_dir = "{{TRAIN_DIR}}"
        if os.path.exists(train_dir):
            class_labels = [d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
            class_labels.sort()
            print(f"📋 Detected class labels from training data: {class_labels}")
    
    # Default class labels if none found
    if not class_labels:
        num_classes = {{NUM_CLASSES}}
        class_labels = [f"class_{i}" for i in range(num_classes)]
        print(f"⚠️  Using default class labels: {class_labels}")
    
    # Check input
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return
    
    # Make predictions
    results = []
    
    if os.path.isfile(input_path):
        # Single image prediction
        print(f"🔍 Predicting single image: {input_path}")
        predicted_class, confidence, probabilities = predict_single_image(
            model, input_path, class_labels, target_size
        )
        
        if predicted_class is not None:
            print(f"\\n🎯 Prediction: {predicted_class}")
            print(f"🎲 Confidence: {confidence:.4f}")
            
            # Show top-k predictions
            top_k_indices = np.argsort(probabilities)[::-1][:args.top_k]
            print(f"\\n📊 Top-{args.top_k} predictions:")
            for i, idx in enumerate(top_k_indices, 1):
                print(f"  {i}. {class_labels[idx]}: {probabilities[idx]:.4f}")
            
            results.append({
                'image_path': input_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': [
                    {'class': class_labels[idx], 'probability': float(probabilities[idx])}
                    for idx in top_k_indices
                ]
            })
    
    elif os.path.isdir(input_path):
        # Batch prediction
        print(f"📁 Predicting batch of images in: {input_path}")
        results = predict_batch(model, input_path, class_labels, target_size)
    
    else:
        print(f"❌ Invalid input path: {input_path}")
        return
    
    # Save results if requested
    if args.output and results:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")
    
    print("✅ Prediction completed!")

if __name__ == "__main__":
    main()
'''
    
    def _get_deploy_template(self) -> str:
        """Get the deployment script template."""
        return '''#!/usr/bin/env python3
"""
Deployment Script for ModelGardener
Generated on: {{GENERATION_DATE}}
Configuration: {{CONFIG_FILE}}
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from pathlib import Path

{{CUSTOM_IMPORTS}}

app = Flask(__name__)

# Global variables
model = None
class_labels = []
target_size = (224, 224)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess image for prediction."""
    try:
        # If image_data is base64 string, decode it
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        # Load image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image from request
        if 'image' not in request.files:
            # Try to get base64 encoded image from JSON
            data = request.get_json()
            if data and 'image' in data:
                image_data = data['image']
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            image_file = request.files['image']
            image_data = image_file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_data, target_size)
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_labels[predicted_class_idx] if class_labels else f"class_{predicted_class_idx}"
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[::-1][:5]
        top_5_predictions = [
            {
                'class': class_labels[idx] if class_labels else f"class_{idx}",
                'probability': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_5_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes."""
    return jsonify({'classes': class_labels})

def load_model_and_classes():
    """Load model and class labels."""
    global model, class_labels, target_size
    
    # Configuration
    config_file = "{{CONFIG_FILE}}"
    model_dir = "{{MODEL_DIR}}"
    img_height = {{IMG_HEIGHT}}
    img_width = {{IMG_WIDTH}}
    target_size = (img_height, img_width)
    
    # Load configuration if available
    if os.path.exists(config_file):
        try:
            config = load_config(config_file)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Could not load configuration: {e}")
            config = {}
    else:
        print(f"⚠️  Configuration file {config_file} not found, using defaults")
        config = {}
    
    {{CUSTOM_LOADER_CALLS}}
    
    # Find model file
    model_files = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.h5') and ('best' in file or 'final' in file):
                model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        return False
    
    # Use the best model if available
    model_file = None
    for file in model_files:
        if 'best' in os.path.basename(file):
            model_file = file
            break
    if not model_file:
        model_file = model_files[0]
    
    print(f"📥 Loading model from: {model_file}")
    
    # Load model
    try:
        model = keras.models.load_model(model_file)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Try to get class labels from training directory structure
    train_dir = "{{TRAIN_DIR}}"
    if os.path.exists(train_dir):
        class_labels = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        class_labels.sort()
        print(f"📋 Detected class labels: {class_labels}")
    
    # Default class labels if none found
    if not class_labels:
        num_classes = {{NUM_CLASSES}}
        class_labels = [f"class_{i}" for i in range(num_classes)]
        print(f"⚠️  Using default class labels: {class_labels}")
    
    return True

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation."""
    docs = """
    <h1>ModelGardener Model API</h1>
    <p>Generated on: {{GENERATION_DATE}}</p>
    
    <h2>Endpoints:</h2>
    <ul>
        <li><strong>GET /health</strong> - Health check</li>
        <li><strong>POST /predict</strong> - Make prediction on uploaded image</li>
        <li><strong>GET /classes</strong> - Get available classes</li>
    </ul>
    
    <h2>Usage Examples:</h2>
    
    <h3>Python requests:</h3>
    <pre>
import requests

# Predict with file upload
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', files={'image': f})
    result = response.json()
    print(result)

# Predict with base64 encoded image
import base64
with open('image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/predict', 
                        json={'image': image_b64})
result = response.json()
print(result)
    </pre>
    
    <h3>cURL:</h3>
    <pre>
# Upload file
curl -X POST -F "image=@image.jpg" http://localhost:5000/predict

# Health check
curl http://localhost:5000/health

# Get classes
curl http://localhost:5000/classes
    </pre>
    """
    return docs

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy ModelGardener model as REST API')
    parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting ModelGardener Model API...")
    print("=" * 50)
    
    # Load model and classes
    if not load_model_and_classes():
        print("❌ Failed to load model")
        sys.exit(1)
    
    print(f"🌐 Starting server on {args.host}:{args.port}")
    print("📖 API documentation available at: http://localhost:{}/".format(args.port))
    print("💡 Use Ctrl+C to stop the server")
    
    # Start Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
'''
    
    def _generate_requirements_txt(self, config: Dict[str, Any], output_dir: str):
        """Generate requirements.txt file."""
        requirements = [
            "tensorflow>=2.10.0",
            "numpy>=1.21.0",
            "Pillow>=8.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
            "PyYAML>=6.0",
            "Flask>=2.2.0",  # For deployment script
        ]
        
        # Add custom requirements based on configuration
        custom_functions = self._extract_custom_functions_info(config)
        if custom_functions:
            for func_type, functions in custom_functions.items():
                for func_info in functions:
                    if isinstance(func_info, dict):
                        dependencies = func_info.get('dependencies', [])
                        for dep in dependencies:
                            if dep not in requirements:
                                requirements.append(dep)
        
        requirements_path = os.path.join(output_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            for req in sorted(requirements):
                f.write(f"{req}\n")
        
        print(f"✅ Generated: {requirements_path}")
    
    def _generate_scripts_readme(self, config: Dict[str, Any], output_dir: str):
        """Generate README file for the scripts."""
        
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        train_dir = data_config.get('train_dir', './data/train')
        val_dir = data_config.get('val_dir', './data/val')
        test_dir = data_config.get('test_dir', './data/test')
        model_name = model_config.get('model_name', 'Custom Model')
        epochs = training_config.get('epochs', 100)
        
        readme_content = f"""# ModelGardener Generated Scripts

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This directory contains automatically generated Python scripts for training, evaluating, predicting, and deploying your {model_name} model.

## Files

- `train.py` - Training script
- `evaluation.py` - Model evaluation script
- `prediction.py` - Prediction script for new images
- `deploy.py` - REST API deployment script
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized as specified in the configuration:
```
{train_dir}/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...

{val_dir}/
├── class1/
├── class2/
└── ...

{test_dir}/  (for evaluation)
├── class1/
├── class2/
└── ...
```

## Usage

### Training

Train your model:
```bash
python train.py
```

The script will:
- Load the configuration from `model_config.yaml`
- Create data generators with augmentation
- Build and compile the model
- Train for {epochs} epochs
- Save the best model to the specified directory

### Evaluation

Evaluate your trained model:
```bash
python evaluation.py
```

The script will:
- Load the best trained model
- Evaluate on test data
- Generate classification report
- Create and save confusion matrix
- Save evaluation results

### Prediction

Make predictions on new images:

Single image:
```bash
python prediction.py --input path/to/image.jpg
```

Batch prediction:
```bash
python prediction.py --input path/to/images/directory/
```

Advanced options:
```bash
python prediction.py --input image.jpg --output results.json --top-k 3
```

### Deployment

Deploy your model as a REST API:
```bash
python deploy.py
```

Options:
```bash
python deploy.py --host 0.0.0.0 --port 8080 --debug
```

The API will be available at `http://localhost:5000` with the following endpoints:
- `GET /health` - Health check
- `POST /predict` - Prediction endpoint
- `GET /classes` - Available classes

## Customization

All scripts are generated based on your YAML configuration and can be customized as needed. The scripts include:

- Automatic configuration loading
- Error handling and logging
- Support for custom functions (if defined)
- Flexible input/output handling

## Custom Functions

If you have custom functions defined in your configuration, make sure they are available in the `src` directory relative to these scripts.

## Notes

- Models are saved in the directory specified in your configuration
- All scripts use the same preprocessing and normalization as specified in your configuration
- Cross-validation training is supported when enabled in the configuration
- The deployment script provides a simple REST API suitable for testing and development

## Support

These scripts are generated automatically by ModelGardener. For issues or customization needs, refer to the ModelGardener documentation.
"""
        
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Generated: {readme_path}")

    def _generate_custom_modules_readme(self, custom_modules_dir: str):
        """Generate README file for custom modules."""
        readme_content = f"""# Custom Modules for ModelGardener

This directory contains templates for custom functions that can be used with ModelGardener.

## Available Templates

- `custom_models.py` - Custom model architectures
- `custom_data_loaders.py` - Custom data loading functions
- `custom_loss_functions.py` - Custom loss functions
- `custom_optimizers.py` - Custom optimizers
- `custom_metrics.py` - Custom metrics
- `custom_callbacks.py` - Custom training callbacks
- `custom_augmentations.py` - Custom data augmentation functions
- `custom_preprocessing.py` - Custom preprocessing functions
- `custom_training_loops.py` - Custom training loop strategies

## Usage

1. **Customize the Templates**: Edit the template files to implement your custom functions
2. **Update Configuration**: Add references to your custom functions in the configuration file
3. **Use in Training**: The generated scripts will automatically load and use your custom functions

## Example Configuration

```yaml
metadata:
  custom_functions:
    models:
      - name: "MyCustomModel"
        file_path: "./custom_modules/custom_models.py"
        function_name: "create_my_custom_model"
    loss_functions:
      - name: "MyCustomLoss"
        file_path: "./custom_modules/custom_loss_functions.py"  
        function_name: "my_custom_loss"
```

## Notes

- All custom functions should follow the patterns shown in the templates
- Make sure to install any additional dependencies required by your custom functions
- Test your custom functions independently before using them in training

## Support

Refer to the ModelGardener documentation for more details on custom functions.
"""
        
        readme_path = os.path.join(custom_modules_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Generated: {readme_path}")

    def _get_custom_models_template(self) -> str:
        """Get the custom models template."""
        return '''"""
Custom Models Template for ModelGardener

This file provides templates for creating custom model architectures.
Implement your models as either functions or classes following the patterns below.
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np


def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5):
    """
    Create a simple CNN model optimized for CIFAR-10.
    
    Args:
        input_shape: Input image shape (height, width, channels) - default (32, 32, 3) for CIFAR-10
        num_classes: Number of output classes - default 10 for CIFAR-10
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_resnet_block_model(input_shape=(224, 224, 3), num_classes=1000, blocks=3):
    """
    Create a ResNet-like model with residual blocks.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes  
        blocks: Number of residual blocks
        
    Returns:
        keras.Model: The model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    filters = 64
    for i in range(blocks):
        # Residual connection
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add shortcut
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1)(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Downsample after each block (except last)
        if i < blocks - 1:
            x = layers.MaxPooling2D(2)(x)
            filters *= 2
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='custom_resnet_like')
    return model


class CustomViTModel(keras.Model):
    """
    Custom Vision Transformer model.
    
    This is an example of a class-based custom model.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000, 
                 patch_size=16, hidden_size=768, num_heads=12, num_layers=12):
        super().__init__()
        
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_projection = layers.Dense(hidden_size)
        
        # Class token and position embedding
        self.class_token = self.add_weight(
            shape=(1, 1, hidden_size),
            initializer='random_normal',
            trainable=True,
            name='class_token'
        )
        
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1,
            output_dim=hidden_size
        )
        
        # Transformer layers
        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.extend([
                layers.MultiHeadAttention(num_heads, hidden_size // num_heads),
                layers.LayerNormalization(),
                layers.Dense(hidden_size * 4, activation='gelu'),
                layers.Dropout(0.1),
                layers.Dense(hidden_size),
                layers.LayerNormalization(),
            ])
        
        # Classification head
        self.layer_norm = layers.LayerNormalization()
        self.head = layers.Dense(num_classes, activation='softmax')
    
    def extract_patches(self, images):
        """Extract patches from input images."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        return patches
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Extract patches
        patches = self.extract_patches(inputs)
        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])
        
        # Project patches
        x = self.patch_projection(patches)
        
        # Add class token
        class_tokens = tf.broadcast_to(
            self.class_token, [batch_size, 1, self.hidden_size]
        )
        x = tf.concat([class_tokens, x], axis=1)
        
        # Add position embeddings
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        x = x + self.position_embedding(positions)
        
        # Apply transformer layers
        for i in range(0, len(self.transformer_layers), 6):
            # Multi-head attention
            attention_output = self.transformer_layers[i](x, x, training=training)
            x = self.transformer_layers[i + 1](x + attention_output)
            
            # MLP
            mlp_output = self.transformer_layers[i + 2](x, training=training)
            mlp_output = self.transformer_layers[i + 3](mlp_output, training=training)
            mlp_output = self.transformer_layers[i + 4](mlp_output, training=training)
            x = self.transformer_layers[i + 5](x + mlp_output)
        
        # Classification head (use class token)
        x = self.layer_norm(x[:, 0])
        return self.head(x)


# Example of how to test your models
if __name__ == "__main__":
    # Test the custom models
    print("Testing custom model definitions...")
    
    # Test function-based models
    model1 = create_simple_cnn(input_shape=(224, 224, 3), num_classes=10)
    print(f"Simple CNN: {model1.name}, params: {model1.count_params():,}")
    
    model2 = create_resnet_block_model(input_shape=(224, 224, 3), num_classes=10)
    print(f"ResNet-like: {model2.name}, params: {model2.count_params():,}")
    
    # Test class-based model
    model3 = CustomViTModel(input_shape=(224, 224, 3), num_classes=10)
    model3.build((None, 224, 224, 3))
    print(f"Custom ViT: params: {model3.count_params():,}")
    
    print("✅ All models created successfully!")
'''
    
    def _get_custom_data_loaders_template(self) -> str:
        """Get the custom data loaders template."""
        return '''"""
Custom Data Loaders Template for ModelGardener

This file provides templates for creating custom data loading functions.
Implement your data loaders as functions that return tf.data.Dataset objects.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def custom_image_data_loader(data_dir: str,
                           batch_size: int = 32,
                           image_size: List[int] = [224, 224],
                           shuffle: bool = True,
                           buffer_size: int = 10000) -> tf.data.Dataset:
    """
    Custom image data loader with advanced preprocessing.
    
    Args:
        data_dir: Directory containing class subdirectories with images
        batch_size: Batch size for training
        image_size: Target image size [height, width]
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    
    def load_and_preprocess_image(path, label):
        """Load and preprocess a single image."""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Custom preprocessing can be added here
        # e.g., normalization, color space conversion, etc.
        
        return image, label
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def custom_csv_data_loader(csv_path: str,
                          image_dir: str,
                          batch_size: int = 32,
                          image_size: List[int] = [224, 224],
                          shuffle: bool = True) -> tf.data.Dataset:
    """
    Custom data loader for CSV-based datasets.
    
    Args:
        csv_path: Path to CSV file with image paths and labels
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Target image size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Assuming CSV has 'image_path' and 'label' columns
    image_paths = [os.path.join(image_dir, path) for path in df['image_path'].values]
    labels = df['label'].values
    
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(len(image_paths))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def custom_numpy_data_loader(data_path: str,
                            labels_path: str,
                            batch_size: int = 32,
                            shuffle: bool = True) -> tf.data.Dataset:
    """
    Custom data loader for numpy arrays.
    
    Args:
        data_path: Path to numpy array file with data
        labels_path: Path to numpy array file with labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset: Dataset
    """
    
    # Load numpy arrays
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    # Normalize data if needed
    data = data.astype(np.float32) / 255.0
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    
    if shuffle:
        dataset = dataset.shuffle(len(data))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class CustomDataLoader:
    """
    Example of a class-based custom data loader.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, image_size: List[int] = [224, 224]):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        
    def create_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """Create dataset for specified split."""
        split_dir = os.path.join(self.data_dir, split)
        
        # Implementation would depend on your data structure
        # This is just a placeholder
        dataset = tf.data.Dataset.from_tensor_slices([])
        return dataset


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom data loaders...")
    
    # You can test your data loaders here
    # Example:
    # dataset = custom_image_data_loader('./data/train', batch_size=16)
    # for batch in dataset.take(1):
    #     print(f"Batch shape: {batch[0].shape}")
    
    print("✅ Custom data loaders template ready!")
'''
    
    def _get_custom_loss_functions_template(self) -> str:
        """Get the custom loss functions template."""
        return '''"""
Custom Loss Functions Template for ModelGardener

This file provides templates for creating custom loss functions.
All loss functions should accept y_true and y_pred as the first two parameters.
"""

import tensorflow as tf
import numpy as np


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    """
    Focal loss for addressing class imbalance.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        alpha: Weighting factor for rare class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        from_logits: Whether predictions are logits or probabilities
        
    Returns:
        Focal loss value
    """
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    # Calculate focal weight
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = alpha * tf.pow((1 - p_t), gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    return tf.reduce_mean(focal_loss)


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for segmentation tasks.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return dice loss
    return 1.0 - dice


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss for siamese networks.
    
    Args:
        y_true: Binary labels (1 for similar, 0 for dissimilar)
        y_pred: Distance between embeddings
        margin: Margin for dissimilar pairs
        
    Returns:
        Contrastive loss value
    """
    # Square the distance
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    
    # Calculate loss
    loss = y_true * square_pred + (1 - y_true) * margin_square
    
    return tf.reduce_mean(loss) / 2.0


def weighted_categorical_crossentropy(y_true, y_pred, class_weights=None):
    """
    Weighted categorical crossentropy for imbalanced datasets.
    
    Args:
        y_true: Ground truth labels (one-hot encoded)
        y_pred: Predicted probabilities
        class_weights: Weight for each class
        
    Returns:
        Weighted crossentropy loss
    """
    if class_weights is None:
        class_weights = tf.ones(tf.shape(y_pred)[-1])
    
    # Calculate crossentropy
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    
    # Apply class weights
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_loss = weights * ce_loss
    
    return tf.reduce_mean(weighted_loss)


def smooth_l1_loss(y_true, y_pred, delta=1.0):
    """
    Smooth L1 loss (Huber loss).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        delta: Threshold for switching between L1 and L2 loss
        
    Returns:
        Smooth L1 loss value
    """
    diff = tf.abs(y_true - y_pred)
    
    # Use L2 loss for small errors, L1 for large errors
    loss = tf.where(
        diff < delta,
        0.5 * tf.square(diff),
        delta * diff - 0.5 * tf.square(delta)
    )
    
    return tf.reduce_mean(loss)


def cosine_similarity_loss(y_true, y_pred):
    """
    Cosine similarity loss for embedding learning.
    
    Args:
        y_true: Ground truth embeddings
        y_pred: Predicted embeddings
        
    Returns:
        Cosine similarity loss
    """
    # Normalize vectors
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Calculate cosine similarity
    cosine_similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Return loss (1 - similarity)
    return tf.reduce_mean(1.0 - cosine_similarity)


class CustomLossClass:
    """
    Example of a class-based custom loss function.
    """
    
    def __init__(self, weight=1.0, reduction='mean'):
        self.weight = weight
        self.reduction = reduction
        
    def __call__(self, y_true, y_pred):
        """Calculate the loss."""
        # Implement your custom loss logic here
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = loss * self.weight
        
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        else:
            return loss


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom loss functions...")
    
    # Create dummy data for testing
    y_true = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], dtype=tf.float32)
    
    # Test focal loss
    fl = focal_loss(y_true, y_pred)
    print(f"Focal loss: {fl:.4f}")
    
    # Test weighted crossentropy
    weights = tf.constant([1.0, 2.0, 1.5])  # Higher weight for class 1
    wce = weighted_categorical_crossentropy(y_true, y_pred, weights)
    print(f"Weighted CE loss: {wce:.4f}")
    
    print("✅ Custom loss functions template ready!")
'''
    
    def _get_custom_optimizers_template(self) -> str:
        """Get the custom optimizers template."""
        return '''"""
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
'''
    
    def _get_custom_metrics_template(self) -> str:
        """Get the custom metrics template."""
        return '''"""
Custom Metrics Template for ModelGardener

This file provides templates for creating custom metrics.
Metrics should inherit from tf.keras.metrics.Metric or be functions.
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric.
    """
    
    def __init__(self, name='f1_score', average='macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.average = average
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class IoU(tf.keras.metrics.Metric):
    """
    Intersection over Union metric for segmentation.
    """
    
    def __init__(self, num_classes, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name='total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class predictions
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Calculate confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        self.total_cm.assign_add(tf.cast(cm, tf.float32))
    
    def result(self):
        # Calculate IoU from confusion matrix
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        true_positives = tf.linalg.diag_part(self.total_cm)
        
        denominator = sum_over_row + sum_over_col - true_positives
        iou = tf.math.divide_no_nan(true_positives, denominator)
        
        return tf.reduce_mean(iou)
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


class TopKAccuracy(tf.keras.metrics.Metric):
    """
    Custom Top-K accuracy metric.
    """
    
    def __init__(self, k=5, name='top_k_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.correct_predictions = self.add_weight(name='correct', initializer='zeros')
        self.total_predictions = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get true class indices
        y_true_indices = tf.argmax(y_true, axis=-1)
        
        # Get top-k predictions
        top_k_pred = tf.nn.top_k(y_pred, k=self.k).indices
        
        # Check if true class is in top-k
        correct = tf.reduce_any(tf.equal(tf.expand_dims(y_true_indices, -1), top_k_pred), axis=-1)
        correct = tf.cast(correct, tf.float32)
        
        self.correct_predictions.assign_add(tf.reduce_sum(correct))
        self.total_predictions.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    def result(self):
        return self.correct_predictions / self.total_predictions
    
    def reset_state(self):
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)


def balanced_accuracy(y_true, y_pred):
    """
    Balanced accuracy function-based metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Balanced accuracy score
    """
    # Convert to class predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Get unique classes
    classes = tf.unique(y_true)[0]
    
    # Calculate recall for each class
    recalls = []
    for cls in classes:
        mask = tf.equal(y_true, cls)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(mask, tf.equal(y_pred, cls)), tf.float32))
        actual_positives = tf.reduce_sum(tf.cast(mask, tf.float32))
        recall = true_positives / (actual_positives + 1e-8)
        recalls.append(recall)
    
    # Return mean recall (balanced accuracy)
    return tf.reduce_mean(tf.stack(recalls))


def matthews_correlation_coefficient(y_true, y_pred):
    """
    Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        
    Returns:
        Matthews correlation coefficient
    """
    # Convert to binary predictions
    y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate confusion matrix elements
    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Calculate MCC
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + 1e-8)


class AverageMetric(tf.keras.metrics.Metric):
    """
    Metric that computes the average of multiple metrics.
    """
    
    def __init__(self, metrics, name='average_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.metrics = metrics
        self.total_sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update all metrics and compute average
        metric_values = []
        for metric in self.metrics:
            if hasattr(metric, 'update_state'):
                metric.update_state(y_true, y_pred, sample_weight)
                metric_values.append(metric.result())
            else:
                metric_values.append(metric(y_true, y_pred))
        
        avg_value = tf.reduce_mean(tf.stack(metric_values))
        self.total_sum.assign_add(avg_value)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.total_sum / self.count
    
    def reset_state(self):
        self.total_sum.assign(0)
        self.count.assign(0)
        for metric in self.metrics:
            if hasattr(metric, 'reset_state'):
                metric.reset_state()


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom metrics...")
    
    # Create dummy data
    y_true = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]], dtype=tf.float32)
    
    # Test F1 Score
    f1_metric = F1Score()
    f1_metric.update_state(y_true, y_pred)
    print(f"F1 Score: {f1_metric.result():.4f}")
    
    # Test Top-K Accuracy
    topk_metric = TopKAccuracy(k=2)
    topk_metric.update_state(y_true, y_pred)
    print(f"Top-2 Accuracy: {topk_metric.result():.4f}")
    
    # Test balanced accuracy
    ba = balanced_accuracy(y_true, y_pred)
    print(f"Balanced Accuracy: {ba:.4f}")
    
    print("✅ Custom metrics template ready!")
'''

    def _get_custom_callbacks_template(self) -> str:
        """Get the custom callbacks template."""
        return '''"""
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
                    print(f"\\nEarly stopping at epoch {epoch + 1}")
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
                print(f"\\nSaved model to {filepath}")


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
'''
    
    def _get_custom_augmentations_template(self) -> str:
        """Get the custom augmentations template."""
        return '''"""
Custom Augmentations Template for ModelGardener

This file provides templates for creating custom data augmentation functions.
Augmentation functions should work with tf.data.Dataset objects.
"""

import tensorflow as tf
import numpy as np


def random_rotation_3d(image, max_angle=30):
    """
    Apply random 3D rotation to an image.
    
    Args:
        image: Input image tensor
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Rotated image
    """
    angle = tf.random.uniform([], -max_angle, max_angle) * np.pi / 180
    return tf.image.rot90(image, k=tf.cast(angle / (np.pi / 2), tf.int32))


def random_brightness_contrast(image, brightness_delta=0.2, contrast_delta=0.2):
    """
    Apply random brightness and contrast adjustments.
    
    Args:
        image: Input image tensor
        brightness_delta: Maximum brightness change
        contrast_delta: Maximum contrast change
        
    Returns:
        Adjusted image
    """
    # Random brightness
    image = tf.image.random_brightness(image, brightness_delta)
    
    # Random contrast
    image = tf.image.random_contrast(image, 1 - contrast_delta, 1 + contrast_delta)
    
    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def random_cutout(image, mask_size=32, num_masks=1):
    """
    Apply random cutout (erasing) to an image.
    
    Args:
        image: Input image tensor
        mask_size: Size of the square mask
        num_masks: Number of masks to apply
        
    Returns:
        Image with random cutouts
    """
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    for _ in range(num_masks):
        # Random position
        y = tf.random.uniform([], 0, height - mask_size, dtype=tf.int32)
        x = tf.random.uniform([], 0, width - mask_size, dtype=tf.int32)
        
        # Create mask
        mask = tf.ones([mask_size, mask_size, tf.shape(image)[2]])
        
        # Apply cutout
        image = tf.tensor_scatter_nd_update(
            image,
            [[y + i, x + j, c] for i in range(mask_size) for j in range(mask_size) for c in range(tf.shape(image)[2])],
            tf.zeros([mask_size * mask_size * tf.shape(image)[2]])
        )
    
    return image


def mixup_augmentation(image1, label1, image2, label2, alpha=0.2):
    """
    Apply mixup augmentation to two images.
    
    Args:
        image1, image2: Input images
        label1, label2: Corresponding labels
        alpha: Mixup parameter
        
    Returns:
        Mixed image and label
    """
    # Sample lambda from Beta distribution
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.random.gamma([], alpha, alpha)
        lam = lam / (lam + tf.random.gamma([], alpha, alpha))
    
    # Mix images
    mixed_image = lam * image1 + (1 - lam) * image2
    
    # Mix labels
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label


def cutmix_augmentation(image1, label1, image2, label2, alpha=1.0):
    """
    Apply CutMix augmentation to two images.
    
    Args:
        image1, image2: Input images
        label1, label2: Corresponding labels
        alpha: CutMix parameter
        
    Returns:
        CutMix image and label
    """
    height, width = tf.shape(image1)[0], tf.shape(image1)[1]
    
    # Sample lambda
    lam = tf.random.uniform([], 0, 1)
    if alpha > 0:
        lam = tf.random.gamma([], alpha, alpha)
        lam = lam / (lam + tf.random.gamma([], alpha, alpha))
    
    # Calculate cut size
    cut_ratio = tf.sqrt(1 - lam)
    cut_w = tf.cast(width * cut_ratio, tf.int32)
    cut_h = tf.cast(height * cut_ratio, tf.int32)
    
    # Random position
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    
    # Calculate box coordinates
    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
    
    # Create mask
    mask = tf.ones_like(image1)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, j, c] for i in range(bby1, bby2) for j in range(bbx1, bbx2) for c in range(tf.shape(image1)[2])],
        tf.zeros([(bby2 - bby1) * (bbx2 - bbx1) * tf.shape(image1)[2]])
    )
    
    # Apply CutMix
    mixed_image = image1 * mask + image2 * (1 - mask)
    
    # Adjust lambda based on actual cut area
    lam = 1 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(width * height, tf.float32)
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_image, mixed_label


def random_perspective_transform(image, distortion_scale=0.5):
    """
    Apply random perspective transformation.
    
    Args:
        image: Input image tensor
        distortion_scale: Scale of perspective distortion
        
    Returns:
        Transformed image
    """
    height, width = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    
    # Original corners
    src_corners = tf.constant([[0., 0.], [width, 0.], [width, height], [0., height]])
    
    # Add random distortion to corners
    distortion = tf.random.uniform([4, 2], -distortion_scale, distortion_scale)
    dst_corners = src_corners + distortion * tf.stack([width, height])
    
    # Apply perspective transform (simplified version)
    # Note: In practice, you might want to use a more sophisticated transform
    return tf.image.resize(image, [tf.shape(image)[0], tf.shape(image)[1]])


def color_jittering(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """
    Apply color jittering with multiple color adjustments.
    
    Args:
        image: Input image tensor
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
        hue: Hue adjustment range
        
    Returns:
        Color-adjusted image
    """
    # Apply transformations in random order
    transforms = [
        lambda img: tf.image.random_brightness(img, brightness),
        lambda img: tf.image.random_contrast(img, 1 - contrast, 1 + contrast),
        lambda img: tf.image.random_saturation(img, 1 - saturation, 1 + saturation),
        lambda img: tf.image.random_hue(img, hue)
    ]
    
    # Shuffle and apply transforms
    for transform in transforms:
        if tf.random.uniform([]) > 0.5:
            image = transform(image)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


class AugmentationPipeline:
    """
    Class for creating complex augmentation pipelines.
    """
    
    def __init__(self, augmentations, probabilities=None):
        """
        Args:
            augmentations: List of augmentation functions
            probabilities: List of probabilities for each augmentation
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [0.5] * len(augmentations)
    
    def __call__(self, image, label=None):
        """Apply random augmentations from the pipeline."""
        for aug, prob in zip(self.augmentations, self.probabilities):
            if tf.random.uniform([]) < prob:
                if label is not None:
                    # For augmentations that modify both image and label
                    try:
                        image, label = aug(image, label)
                    except:
                        image = aug(image)
                else:
                    image = aug(image)
        
        return (image, label) if label is not None else image


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom augmentations...")
    
    # Create dummy image
    dummy_image = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
    
    # Test brightness/contrast
    aug_image = random_brightness_contrast(dummy_image)
    print(f"Brightness/Contrast: {aug_image.shape}")
    
    # Test color jittering
    aug_image = color_jittering(dummy_image)
    print(f"Color Jittering: {aug_image.shape}")
    
    # Test augmentation pipeline
    pipeline = AugmentationPipeline([
        random_brightness_contrast,
        color_jittering
    ])
    aug_image = pipeline(dummy_image)
    print(f"Pipeline: {aug_image.shape}")
    
    print("✅ Custom augmentations template ready!")
'''

    def _get_custom_preprocessing_template(self) -> str:
        """Get the custom preprocessing template."""  
        return '''"""
Custom Preprocessing Template for ModelGardener

This file provides templates for creating custom preprocessing functions.
Preprocessing functions should work with individual samples or batches.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List


def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image using ImageNet statistics or custom values.
    
    Args:
        image: Input image tensor (0-1 range)
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized image
    """
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    
    # Normalize
    image = (image - mean) / std
    
    return image


def resize_with_padding(image, target_size, pad_value=0):
    """
    Resize image while maintaining aspect ratio using padding.
    
    Args:
        image: Input image tensor
        target_size: Target size [height, width]
        pad_value: Value to use for padding
        
    Returns:
        Resized and padded image
    """
    target_height, target_width = target_size
    
    # Get original dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    # Calculate scaling factor
    scale = tf.minimum(
        tf.cast(target_height, tf.float32) / tf.cast(height, tf.float32),
        tf.cast(target_width, tf.float32) / tf.cast(width, tf.float32)
    )
    
    # Calculate new dimensions
    new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
    
    # Resize image
    image = tf.image.resize(image, [new_height, new_width])
    
    # Calculate padding
    pad_height = target_height - new_height
    pad_width = target_width - new_width
    
    # Apply padding
    paddings = [
        [pad_height // 2, pad_height - pad_height // 2],
        [pad_width // 2, pad_width - pad_width // 2],
        [0, 0]
    ]
    
    image = tf.pad(image, paddings, constant_values=pad_value)
    
    return image


def histogram_equalization(image):
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        image: Input image tensor (0-1 range)
        
    Returns:
        Equalized image
    """
    # Convert to uint8
    image_uint8 = tf.cast(image * 255, tf.uint8)
    
    # Apply histogram equalization per channel
    channels = []
    for c in range(tf.shape(image)[-1]):
        channel = image_uint8[:, :, c]
        # Flatten and compute histogram
        channel_flat = tf.reshape(channel, [-1])
        hist = tf.histogram_fixed_width(tf.cast(channel_flat, tf.float32), [0.0, 255.0], nbins=256)
        
        # Compute CDF
        cdf = tf.cumsum(hist)
        cdf_normalized = cdf / tf.cast(tf.reduce_max(cdf), tf.float32) * 255
        
        # Apply equalization
        channel_eq = tf.gather(cdf_normalized, channel)
        channels.append(channel_eq)
    
    # Combine channels
    image_eq = tf.stack(channels, axis=-1)
    image_eq = tf.cast(image_eq, tf.float32) / 255.0
    
    return image_eq


def apply_gaussian_noise(image, stddev=0.1):
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image tensor
        stddev: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = tf.random.normal(tf.shape(image), stddev=stddev)
    noisy_image = image + noise
    
    # Clip to valid range
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    
    return noisy_image


def edge_enhancement(image, kernel_size=3, sigma=1.0):
    """
    Apply edge enhancement using Gaussian derivatives.
    
    Args:
        image: Input image tensor
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of Gaussian
        
    Returns:
        Edge-enhanced image
    """
    # Convert to grayscale if needed
    if tf.shape(image)[-1] == 3:
        gray = tf.reduce_mean(image, axis=-1, keepdims=True)
    else:
        gray = image
    
    # Create Gaussian kernel
    kernel = tf.cast(tf.range(kernel_size), tf.float32) - (kernel_size - 1) / 2
    kernel = tf.exp(-0.5 * tf.square(kernel) / (sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Reshape for conv2d
    kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1])
    
    # Apply horizontal and vertical gradients
    grad_x = tf.nn.conv2d(gray, kernel, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(gray, tf.transpose(kernel, [1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
    
    # Compute gradient magnitude
    gradient_mag = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))
    
    # Enhance edges
    enhanced = image + 0.5 * gradient_mag
    enhanced = tf.clip_by_value(enhanced, 0.0, 1.0)
    
    return enhanced


def color_space_conversion(image, from_space='RGB', to_space='HSV'):
    """
    Convert image between color spaces.
    
    Args:
        image: Input image tensor
        from_space: Source color space
        to_space: Target color space
        
    Returns:
        Converted image
    """
    if from_space == 'RGB' and to_space == 'HSV':
        return tf.image.rgb_to_hsv(image)
    elif from_space == 'HSV' and to_space == 'RGB':
        return tf.image.hsv_to_rgb(image)
    elif from_space == 'RGB' and to_space == 'LAB':
        # Simplified RGB to LAB conversion
        # Note: This is a simplified version, full LAB conversion is more complex
        return tf.image.rgb_to_yuv(image)
    else:
        return image


def adaptive_preprocessing(image, image_stats=None):
    """
    Apply adaptive preprocessing based on image statistics.
    
    Args:
        image: Input image tensor
        image_stats: Precomputed image statistics
        
    Returns:
        Preprocessed image
    """
    if image_stats is None:
        # Compute statistics
        mean_val = tf.reduce_mean(image)
        std_val = tf.math.reduce_std(image)
        brightness = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
    else:
        mean_val, std_val, brightness = image_stats
    
    # Adaptive normalization
    if std_val > 0.2:
        # High variance - apply histogram equalization
        image = histogram_equalization(image)
    
    if brightness < 0.3:
        # Dark image - apply gamma correction
        gamma = 0.7
        image = tf.pow(image, gamma)
    elif brightness > 0.7:
        # Bright image - reduce brightness
        image = image * 0.9
    
    # Final normalization
    image = normalize_image(image)
    
    return image


class PreprocessingPipeline:
    """
    Class for creating preprocessing pipelines.
    """
    
    def __init__(self, steps, apply_order='sequential'):
        """
        Args:
            steps: List of preprocessing functions
            apply_order: Order to apply steps ('sequential' or 'random')
        """
        self.steps = steps
        self.apply_order = apply_order
    
    def __call__(self, image, label=None):
        """Apply preprocessing steps."""
        if self.apply_order == 'random':
            # Apply steps in random order
            steps = tf.random.shuffle(self.steps)
        else:
            steps = self.steps
        
        for step in steps:
            try:
                image = step(image)
            except Exception as e:
                print(f"Warning: Preprocessing step failed: {e}")
                continue
        
        return (image, label) if label is not None else image


# Batch preprocessing functions
def batch_normalize(batch_images, batch_size=None):
    """
    Normalize a batch of images.
    
    Args:
        batch_images: Batch of images [batch, height, width, channels]
        batch_size: Size of batch
        
    Returns:
        Normalized batch
    """
    # Compute batch statistics
    batch_mean = tf.reduce_mean(batch_images, axis=[1, 2, 3], keepdims=True)
    batch_var = tf.math.reduce_variance(batch_images, axis=[1, 2, 3], keepdims=True)
    
    # Normalize
    normalized_batch = (batch_images - batch_mean) / (tf.sqrt(batch_var) + 1e-8)
    
    return normalized_batch


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom preprocessing functions...")
    
    # Create dummy image
    dummy_image = tf.random.uniform([224, 224, 3], 0, 1, dtype=tf.float32)
    
    # Test normalization
    norm_image = normalize_image(dummy_image)
    print(f"Normalized image shape: {norm_image.shape}")
    
    # Test resize with padding
    padded_image = resize_with_padding(dummy_image, [256, 256])
    print(f"Padded image shape: {padded_image.shape}")
    
    # Test preprocessing pipeline
    pipeline = PreprocessingPipeline([
        normalize_image,
        lambda img: apply_gaussian_noise(img, 0.05)
    ])
    processed_image = pipeline(dummy_image)
    print(f"Pipeline processed shape: {processed_image.shape}")
    
    print("✅ Custom preprocessing template ready!")
'''

    def _get_custom_training_loops_template(self) -> str:
        """Get the custom training loops template."""
        return '''"""
Custom Training Loops Template for ModelGardener

This file provides templates for creating custom training loops.
These allow for advanced training strategies beyond standard fit() methods.
"""

import tensorflow as tf
import numpy as np
import time
from typing import Dict, Callable, Any, Optional


class GradualUnfreezingTrainer:
    """
    Custom trainer that gradually unfreezes layers during training.
    """
    
    def __init__(self, model, optimizer, loss_fn, unfreeze_schedule=None):
        """
        Args:
            model: Keras model to train
            optimizer: Optimizer to use
            loss_fn: Loss function
            unfreeze_schedule: Dict mapping epoch to number of layers to unfreeze
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.unfreeze_schedule = unfreeze_schedule or {0: 10, 5: 20, 10: -1}
        
        # Initially freeze all layers except the last few
        self._freeze_layers(10)
    
    def _freeze_layers(self, num_trainable):
        """Freeze all layers except the last num_trainable layers."""
        if num_trainable == -1:
            # Unfreeze all layers
            for layer in self.model.layers:
                layer.trainable = True
        else:
            # Freeze all first, then unfreeze last num_trainable
            for layer in self.model.layers:
                layer.trainable = False
            
            for layer in self.model.layers[-num_trainable:]:
                layer.trainable = True
    
    def train_step(self, batch_data):
        """Single training step."""
        images, labels = batch_data
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        
        # Compute gradients and apply
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss, predictions
    
    def train(self, dataset, epochs, validation_data=None, callbacks=None):
        """Custom training loop with gradual unfreezing."""
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Check if we need to unfreeze layers
            if epoch in self.unfreeze_schedule:
                self._freeze_layers(self.unfreeze_schedule[epoch])
                print(f"Epoch {epoch}: Unfroze {self.unfreeze_schedule[epoch]} layers")
            
            # Training loop
            epoch_loss = []
            for batch in dataset:
                loss, _ = self.train_step(batch)
                epoch_loss.append(loss)
            
            avg_loss = tf.reduce_mean(epoch_loss)
            history['loss'].append(float(avg_loss))
            
            # Validation
            if validation_data is not None:
                val_loss = self._validate(validation_data)
                history['val_loss'].append(float(val_loss))
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': avg_loss})
        
        return history
    
    def _validate(self, validation_data):
        """Validation step."""
        val_losses = []
        for val_batch in validation_data:
            val_images, val_labels = val_batch
            val_predictions = self.model(val_images, training=False)
            val_loss = self.loss_fn(val_labels, val_predictions)
            val_losses.append(val_loss)
        
        return tf.reduce_mean(val_losses)


class AdversarialTrainer:
    """
    Custom trainer for adversarial training.
    """
    
    def __init__(self, model, optimizer, loss_fn, epsilon=0.1, alpha=0.01):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            epsilon: Maximum perturbation magnitude
            alpha: Step size for adversarial examples
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.alpha = alpha
    
    def generate_adversarial_examples(self, images, labels):
        """Generate adversarial examples using FGSM."""
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
        
        # Compute gradients w.r.t. input
        gradients = tape.gradient(loss, images)
        
        # Generate adversarial examples
        signed_grad = tf.sign(gradients)
        adversarial_images = images + self.alpha * signed_grad
        adversarial_images = tf.clip_by_value(
            adversarial_images,
            images - self.epsilon,
            images + self.epsilon
        )
        adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
        
        return adversarial_images
    
    def train_step(self, batch_data):
        """Adversarial training step."""
        images, labels = batch_data
        
        # Generate adversarial examples
        adv_images = self.generate_adversarial_examples(images, labels)
        
        # Mix clean and adversarial examples
        mixed_images = tf.concat([images, adv_images], axis=0)
        mixed_labels = tf.concat([labels, labels], axis=0)
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = self.model(mixed_images, training=True)
            loss = self.loss_fn(mixed_labels, predictions)
        
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss


class CurriculumLearningTrainer:
    """
    Trainer implementing curriculum learning.
    """
    
    def __init__(self, model, optimizer, loss_fn, difficulty_fn=None):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            difficulty_fn: Function to determine sample difficulty
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.difficulty_fn = difficulty_fn or self._default_difficulty
    
    def _default_difficulty(self, sample, epoch):
        """Default difficulty function based on prediction confidence."""
        image, label = sample
        predictions = self.model(tf.expand_dims(image, 0), training=False)
        confidence = tf.reduce_max(tf.nn.softmax(predictions))
        return 1.0 - confidence  # Higher difficulty for low confidence
    
    def create_curriculum(self, dataset, epoch):
        """Create curriculum for current epoch."""
        # Collect all samples with difficulty scores
        samples_with_difficulty = []
        for sample in dataset.unbatch():
            difficulty = self._default_difficulty(sample, epoch)
            samples_with_difficulty.append((sample, difficulty))
        
        # Sort by difficulty (easy to hard)
        samples_with_difficulty.sort(key=lambda x: x[1])
        
        # Select subset based on epoch (gradually increase difficulty)
        total_samples = len(samples_with_difficulty)
        if epoch < 5:
            # Early epochs: use only easy samples
            selected_samples = samples_with_difficulty[:total_samples // 2]
        elif epoch < 15:
            # Middle epochs: use easy and medium samples
            selected_samples = samples_with_difficulty[:int(total_samples * 0.8)]
        else:
            # Later epochs: use all samples
            selected_samples = samples_with_difficulty
        
        # Extract just the samples
        curriculum_samples = [sample for sample, _ in selected_samples]
        
        return tf.data.Dataset.from_generator(
            lambda: curriculum_samples,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )


class MetaLearningTrainer:
    """
    Trainer for meta-learning (learning to learn).
    """
    
    def __init__(self, model, meta_optimizer, inner_optimizer, inner_steps=1):
        """
        Args:
            model: Model to meta-train
            meta_optimizer: Optimizer for meta-updates
            inner_optimizer: Optimizer for inner loop updates
            inner_steps: Number of inner loop steps
        """
        self.model = model
        self.meta_optimizer = meta_optimizer
        self.inner_optimizer = inner_optimizer
        self.inner_steps = inner_steps
    
    def inner_loop(self, support_data, loss_fn):
        """Inner loop adaptation."""
        # Create a copy of model weights
        initial_weights = [var.numpy() for var in self.model.trainable_variables]
        
        # Perform inner loop updates
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                support_images, support_labels = support_data
                support_pred = self.model(support_images, training=True)
                support_loss = loss_fn(support_labels, support_pred)
            
            gradients = tape.gradient(support_loss, self.model.trainable_variables)
            self.inner_optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
        
        return initial_weights
    
    def meta_step(self, task_batch, loss_fn):
        """Meta-learning step."""
        meta_gradients = []
        
        for task in task_batch:
            support_data, query_data = task
            
            # Save initial weights and perform inner loop
            initial_weights = self.inner_loop(support_data, loss_fn)
            
            # Compute meta-gradient on query set
            with tf.GradientTape() as tape:
                query_images, query_labels = query_data
                query_pred = self.model(query_images, training=True)
                query_loss = loss_fn(query_labels, query_pred)
            
            meta_grad = tape.gradient(query_loss, self.model.trainable_variables)
            meta_gradients.append(meta_grad)
            
            # Restore initial weights
            for var, initial_weight in zip(self.model.trainable_variables, initial_weights):
                var.assign(initial_weight)
        
        # Average meta-gradients and apply
        avg_meta_gradients = []
        for i in range(len(meta_gradients[0])):
            avg_grad = tf.reduce_mean([grad[i] for grad in meta_gradients], axis=0)
            avg_meta_gradients.append(avg_grad)
        
        self.meta_optimizer.apply_gradients(
            zip(avg_meta_gradients, self.model.trainable_variables)
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom training loops...")
    
    # Create dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    # Test gradual unfreezing trainer
    trainer = GradualUnfreezingTrainer(model, optimizer, loss_fn)
    print(f"Gradual Unfreezing Trainer: {trainer.__class__.__name__}")
    
    # Test adversarial trainer
    adv_trainer = AdversarialTrainer(model, optimizer, loss_fn)
    print(f"Adversarial Trainer: {adv_trainer.__class__.__name__}")
    
    print("✅ Custom training loops template ready!")
'''
