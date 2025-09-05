#!/usr/bin/env python3
"""
Training Script for ModelGardener
Generated on: 2025-09-04 17:04:12
Configuration: config.yaml
"""

import os
import sys
import yaml
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path

# No custom functions

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

def compile_model(model, optimizer='Adam', learning_rate=0.001, 
                  loss='Categorical Crossentropy', metrics=['Accuracy']):
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
    config_file = "config.yaml"
    train_dir = "./data"
    val_dir = "./data"
    batch_size = 32
    epochs = 100
    model_dir = "./logs"
    img_height = 32
    img_width = 32
    channels = 3
    num_classes = 10
    
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
    
        # No custom functions to load
    custom_functions = None
    
        # Using custom data loader: Custom_load_cifar10_npz_data
    print("📁 Loading data with custom loader: Custom_load_cifar10_npz_data")
    try:
        from custom_modules.custom_data_loaders import Custom_load_cifar10_npz_data
        
        # Get data loader parameters
        loader_params = config.get('configuration', {}).get('data', {}).get('data_loader', {}).get('parameters', {})
        
        # Load training and validation data
        train_gen, val_gen = Custom_load_cifar10_npz_data(
            train_dir=train_dir,
            val_dir=val_dir,
            **loader_params
        )
        
        print("✅ Custom data loader loaded successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import custom data loader Custom_load_cifar10_npz_data: {e}")
        print("🔄 Falling back to default data generators...")
        train_gen, val_gen = create_data_generators(
            train_dir, val_dir, batch_size, img_height, img_width
        )
    except Exception as e:
        print(f"❌ Error using custom data loader: {e}")
        print("🔄 Falling back to default data generators...")
        train_gen, val_gen = create_data_generators(
            train_dir, val_dir, batch_size, img_height, img_width
        )
    
    input_shape = (img_height, img_width, channels)
    
    # Update num_classes from data if possible
    if hasattr(train_gen, 'num_classes'):
        num_classes = train_gen.num_classes
        print(f"📊 Detected {num_classes} classes from data")
    
    # Build model
    print("🏗️  Building model...")
    model = build_model(
        "custom_model", "create_simple_cnn", 
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
    cv_enabled = False
    if cv_enabled:
        print("🔄 Cross-validation training enabled")
        k_folds = 5
        
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
