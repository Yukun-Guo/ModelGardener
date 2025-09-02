
"""
Sample custom model file for testing the enhanced model loading feature.
"""

import tensorflow as tf
import keras
from keras import layers

def create_simple_cnn(input_shape=(224, 224, 3), num_classes=1000, dropout_rate=0.5):
    """
    Create a simple CNN model for image classification.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    """
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='simple_cnn')
    return model

def build_resnet_like(input_shape=(224, 224, 3), num_classes=1000, 
                     filters=64, blocks=3, use_batch_norm=True):
    """
    Build a ResNet-like architecture.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of classes for classification
        filters: Number of filters in first conv layer
        blocks: Number of residual blocks
        use_batch_norm: Whether to use batch normalization
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial conv layer
    x = layers.Conv2D(filters, 7, strides=2, padding='same')(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    for i in range(blocks):
        residual = x
        x = layers.Conv2D(filters, 3, padding='same')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        filters *= 2
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name='resnet_like')

class CustomTransformer(keras.Model):
    """Custom transformer-based model class."""
    
    def __init__(self, vocab_size=10000, embed_dim=128, num_heads=8, 
                 num_layers=4, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Build layers
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = [
            layers.MultiHeadAttention(num_heads, embed_dim)
            for _ in range(num_layers)
        ]
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, x)
        
        x = self.global_pool(x)
        return self.classifier(x)

# Non-model functions (should be ignored)
def helper_function():
    """This is just a helper function, not a model."""
    pass

def preprocess_data(data):
    """Data preprocessing function."""
    return data

# Non-model class (should be ignored)
class DataLoader:
    """Custom data loader class."""
    pass
