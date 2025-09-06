"""
Example multi-output model for testing loss function configuration.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_multi_output_model(input_shape=(224, 224, 3), num_classes_main=10, num_classes_aux=5):
    """
    Create a multi-output CNN model with main and auxiliary outputs.
    
    Args:
        input_shape: Input shape for the model
        num_classes_main: Number of classes for main output
        num_classes_aux: Number of classes for auxiliary output
        
    Returns:
        Multi-output Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_layer")
    
    # Shared feature extraction layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Shared dense layers
    shared_dense = layers.Dense(512, activation='relu', name="shared_features")(x)
    shared_dropout = layers.Dropout(0.5)(shared_dense)
    
    # Main output branch
    main_branch = layers.Dense(256, activation='relu', name="main_branch")(shared_dropout)
    main_dropout = layers.Dropout(0.3)(main_branch)
    main_output = layers.Dense(num_classes_main, activation='softmax', name="main_output")(main_dropout)
    
    # Auxiliary output branch
    aux_branch = layers.Dense(128, activation='relu', name="aux_branch")(shared_dropout)
    aux_dropout = layers.Dropout(0.3)(aux_branch)
    auxiliary_output = layers.Dense(num_classes_aux, activation='softmax', name="auxiliary_output")(aux_dropout)
    
    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=[main_output, auxiliary_output],
        name="multi_output_cnn"
    )
    
    return model

def create_simple_multi_output_classifier(input_shape=(224, 224, 3), classes=10):
    """
    Create a simple multi-output classifier for testing.
    
    Args:
        input_shape: Input image shape
        classes: Number of classes
        
    Returns:
        Multi-output model with main and auxiliary outputs
    """
    return create_multi_output_model(
        input_shape=input_shape, 
        num_classes_main=classes,
        num_classes_aux=classes // 2  # Auxiliary output has half the classes
    )

if __name__ == "__main__":
    # Test the model creation
    model = create_multi_output_model()
    model.summary()
    
    print(f"\nModel has {len(model.outputs)} outputs:")
    for i, output in enumerate(model.outputs):
        print(f"  Output {i+1}: {output.name} - shape: {output.shape}")
