"""
CIFAR-10 optimized CNN model for ModelGardener.

This module contains custom CNN architectures optimized for CIFAR-10 dataset
(32x32x3 images, 10 classes).
"""

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



def create_simple_cnn_two_outputs(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, **kwargs):
    """
    Create a simple CNN model optimized for CIFAR-10 image classification.
    
    Args:
        input_shape: Input tensor shape (height, width, channels) - default (32, 32, 3) for CIFAR-10
        num_classes: Number of output classes - default 10 for CIFAR-10
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
    outputs_aux = layers.Dense(5, activation='softmax', name='aux_output')(x)

    model = keras.Model(inputs, [outputs, outputs_aux], name='cifar10_cnn')
    return model
