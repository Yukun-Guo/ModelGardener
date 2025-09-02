"""
Example custom model definitions for ModelGardener.

This file demonstrates how to create custom models that can be loaded
into ModelGardener. Models can be defined as:

1. Functions that return a keras model
2. Classes that inherit from keras.models.Model

The function signature should accept common parameters like:
- input_shape: tuple of input dimensions
- num_classes: number of output classes
- **kwargs: additional model-specific parameters
"""

import keras
from keras import layers


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=1000, dropout_rate=0.5, **kwargs):
    """
    Create a simple CNN model for image classification.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction layers
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='simple_cnn')
    return model

if __name__ == "__main__":
    # Test the custom models
    print("Testing custom model definitions...")
    
    # Test function-based model
    model1 = create_simple_cnn(input_shape=(224, 224, 3), num_classes=10)
    print(f"Simple CNN: {model1.name}, params: {model1.count_params():,}")
    
    
    print("Custom models created successfully!")
