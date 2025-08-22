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

import tensorflow as tf
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


def create_residual_block_model(input_shape=(224, 224, 3), num_classes=1000, 
                               num_blocks=3, filters=64, **kwargs):
    """
    Create a model with residual blocks.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        num_blocks: Number of residual blocks
        filters: Base number of filters
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: Model with residual connections
    """
    def residual_block(x, filters, kernel_size=3):
        """Create a residual block."""
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(filters, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    for i in range(num_blocks):
        x = residual_block(x, filters * (2 ** i))
        x = layers.MaxPooling2D()(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='residual_block_model')
    return model


class CustomViTModel(keras.Model):
    """
    Custom Vision Transformer model class.
    
    This demonstrates how to create a custom model by inheriting from keras.Model.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000, 
                 patch_size=16, num_layers=6, hidden_size=256, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Calculate number of patches
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        
        # Build layers
        self._build_model()
    
    def _build_model(self):
        """Build the model layers."""
        # Patch extraction
        self.patch_projection = layers.Dense(self.hidden_size)
        
        # Position embeddings
        self.position_embedding = self.add_weight(
            shape=(1, self.num_patches + 1, self.hidden_size),
            initializer='random_normal',
            trainable=True,
            name='position_embedding'
        )
        
        # Class token
        self.class_token = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer='random_normal', 
            trainable=True,
            name='class_token'
        )
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            self.transformer_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_size // self.num_heads,
                    name=f'attention_{i}'
                )
            )
            self.transformer_layers.append(
                layers.LayerNormalization(name=f'norm1_{i}')
            )
            self.transformer_layers.append(
                layers.Dense(self.hidden_size * 4, activation='gelu', name=f'mlp1_{i}')
            )
            self.transformer_layers.append(
                layers.Dense(self.hidden_size, name=f'mlp2_{i}')
            )
            self.transformer_layers.append(
                layers.LayerNormalization(name=f'norm2_{i}')
            )
        
        # Classification head
        self.layer_norm = layers.LayerNormalization()
        self.head = layers.Dense(self.num_classes, activation='softmax')
    
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
        x = x + self.position_embedding
        
        # Apply transformer layers
        for i in range(0, len(self.transformer_layers), 5):
            # Multi-head attention
            attention_output = self.transformer_layers[i](x, x, training=training)
            x = self.transformer_layers[i + 1](x + attention_output)
            
            # MLP
            mlp_output = self.transformer_layers[i + 2](x, training=training)
            mlp_output = self.transformer_layers[i + 3](mlp_output, training=training)
            x = self.transformer_layers[i + 4](x + mlp_output)
        
        # Classification head (use class token)
        x = self.layer_norm(x[:, 0])
        return self.head(x)
    
    def extract_patches(self, images):
        """Extract patches from images."""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return patches


def create_segmentation_unet(input_shape=(256, 256, 3), num_classes=21, 
                           filters=64, num_layers=4, dropout_rate=0.5, **kwargs):
    """
    Create a U-Net model for semantic segmentation.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of segmentation classes
        filters: Base number of filters
        num_layers: Number of encoder/decoder layers
        dropout_rate: Dropout rate
        **kwargs: Additional parameters
        
    Returns:
        keras.Model: U-Net segmentation model
    """
    def conv_block(x, filters, dropout=False):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        if dropout:
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (downsampling path)
    encoder_outputs = []
    x = inputs
    
    for i in range(num_layers):
        x = conv_block(x, filters * (2 ** i), dropout=(i >= 2))
        encoder_outputs.append(x)
        x = layers.MaxPooling2D()(x)
    
    # Bridge
    x = conv_block(x, filters * (2 ** num_layers), dropout=True)
    
    # Decoder (upsampling path)
    for i in range(num_layers - 1, -1, -1):
        x = layers.Conv2DTranspose(
            filters * (2 ** i), 2, strides=2, padding='same'
        )(x)
        x = layers.Concatenate()([x, encoder_outputs[i]])
        x = conv_block(x, filters * (2 ** i), dropout=(i >= 2))
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='custom_unet')
    return model


if __name__ == "__main__":
    # Test the custom models
    print("Testing custom model definitions...")
    
    # Test function-based model
    model1 = create_simple_cnn(input_shape=(224, 224, 3), num_classes=10)
    print(f"Simple CNN: {model1.name}, params: {model1.count_params():,}")
    
    # Test residual model
    model2 = create_residual_block_model(input_shape=(224, 224, 3), num_classes=10)
    print(f"Residual Model: {model2.name}, params: {model2.count_params():,}")
    
    # Test class-based model
    model3 = CustomViTModel(input_shape=(224, 224, 3), num_classes=10)
    model3.build((None, 224, 224, 3))
    print(f"Custom ViT: {model3.name}, params: {model3.count_params():,}")
    
    # Test segmentation model
    model4 = create_segmentation_unet(input_shape=(256, 256, 3), num_classes=5)
    print(f"U-Net: {model4.name}, params: {model4.count_params():,}")
    
    print("All custom models created successfully!")
