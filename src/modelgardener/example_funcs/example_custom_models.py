"""
Enhanced custom model architectures for ModelGardener.

This module contains custom model architectures that can be dynamically loaded
into the ModelGardener application.

the returned models should be Keras Model instances.
"""

import keras

def example_model(input_shape=(32, 32, 3), num_classes=10, **kwargs):
    """
    Example custom model architecture.
    
    Args:
        input_shape: Shape of the input tensor (height, width, channels)
        num_classes: Number of output classes
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: keras Model instance
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Simple CNN architecture
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = keras.Model(inputs, outputs, name="example_custom_model")
    return model

def example_model_two_outputs(input_shape=(32, 32, 3), num_classes=10, **kwargs):
    """
    Another example custom model architecture.
    
    Args:
        input_shape: Shape of the input tensor (height, width)
        num_classes: Number of output classes
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: keras Model instance with two outputs
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)    
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    outputs2 = keras.layers.Dense(1, activation='sigmoid', name='output_layer_2')(x)
    model = keras.Model(inputs, [outputs, outputs2], name="example_custom_model_2")
    return model