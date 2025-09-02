"""
Custom Models Template for ModelGardener

This file provides templates for creating custom model architectures.
Implement your models as either functions or classes following the patterns below.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=1000, dropout_rate=0.5):
    """
    Create a simple CNN model.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
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
