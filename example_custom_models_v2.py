"""
Example custom models for testing the enhanced trainer
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_simple_cnn_v2(input_shape=(224, 224, 3), num_classes=1000, dropout_rate=0.3):
    """
    Create a simple CNN model with batch normalization.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes  
        dropout_rate: Dropout rate for regularization
    """
    inputs = keras.Input(shape=input_shape, name='input_images')
    
    # First convolutional block
    x = layers.Conv2D(64, 3, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Second convolutional block
    x = layers.Conv2D(128, 3, padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    # Third convolutional block
    x = layers.Conv2D(256, 3, padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Classification head
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='simple_cnn_v2')
    return model

def build_efficientnet_custom(input_shape=(224, 224, 3), num_classes=1000, width_coefficient=1.0):
    """
    Build a custom EfficientNet-style model.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of classes for classification
        width_coefficient: Width scaling coefficient
    """
    inputs = keras.Input(shape=input_shape, name='input_images')
    
    # Initial conv layer
    base_filters = int(32 * width_coefficient)
    x = layers.Conv2D(base_filters, 3, strides=2, padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_activation')(x)
    
    # MBConv blocks (simplified)
    block_configs = [
        (base_filters, 16, 1, 1),     # Stage 1
        (base_filters * 2, 24, 2, 2), # Stage 2  
        (base_filters * 4, 40, 2, 2), # Stage 3
        (base_filters * 6, 80, 3, 2), # Stage 4
    ]
    
    for i, (filters_in, filters_out, repeats, stride) in enumerate(block_configs):
        for j in range(repeats):
            block_stride = stride if j == 0 else 1
            x = _mb_conv_block(x, filters_in, filters_out, block_stride, 
                             f'block{i+1}{chr(97+j)}')
    
    # Head
    x = layers.Conv2D(int(1280 * width_coefficient), 1, padding='same', name='top_conv')(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation('swish', name='top_activation')(x)
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if num_classes > 0:
        x = layers.Dropout(0.2, name='top_dropout')(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    else:
        outputs = x
    
    model = keras.Model(inputs, outputs, name='efficientnet_custom')
    return model

def _mb_conv_block(inputs, filters_in, filters_out, stride, block_name):
    """MobileNet V2 inverted residual block."""
    # Expansion
    expand_filters = filters_in * 6
    x = layers.Conv2D(expand_filters, 1, padding='same', name=f'{block_name}_expand_conv')(inputs)
    x = layers.BatchNormalization(name=f'{block_name}_expand_bn')(x)
    x = layers.Activation('swish', name=f'{block_name}_expand_activation')(x)
    
    # Depthwise
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', name=f'{block_name}_dwconv')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn')(x)
    x = layers.Activation('swish', name=f'{block_name}_activation')(x)
    
    # Squeeze and excitation (simplified)
    se_filters = max(1, filters_in // 4)
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, expand_filters))(se)
    se = layers.Conv2D(se_filters, 1, activation='swish', name=f'{block_name}_se_reduce')(se)
    se = layers.Conv2D(expand_filters, 1, activation='sigmoid', name=f'{block_name}_se_expand')(se)
    x = layers.multiply([x, se], name=f'{block_name}_se_excite')
    
    # Output
    x = layers.Conv2D(filters_out, 1, padding='same', name=f'{block_name}_project_conv')(x)
    x = layers.BatchNormalization(name=f'{block_name}_project_bn')(x)
    
    # Residual connection
    if stride == 1 and filters_in == filters_out:
        x = layers.add([inputs, x], name=f'{block_name}_add')
    
    return x

class CustomVisionTransformer(keras.Model):
    """Custom Vision Transformer implementation."""
    
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = layers.Conv2D(
            embed_dim, patch_size, strides=patch_size, name='patch_embed'
        )
        
        # Position embedding
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.num_patches + 1, embed_dim),
            initializer='random_normal'
        )
        
        # Class token
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, embed_dim),
            initializer='random_normal'
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate, 
                           name=f'transformer_block_{i}')
            for i in range(num_layers)
        ]
        
        # Classification head
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name='layer_norm')
        self.head = layers.Dense(num_classes, name='head')
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Patch embedding
        x = self.patch_embed(inputs)
        x = tf.reshape(x, (batch_size, self.num_patches, self.embed_dim))
        
        # Add class token
        class_tokens = tf.broadcast_to(self.class_token, (batch_size, 1, self.embed_dim))
        x = tf.concat([class_tokens, x], axis=1)
        
        # Add position embedding
        x = x + self.pos_embedding
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Classification
        x = self.layer_norm(x)
        class_token_output = x[:, 0]  # Use class token for classification
        return self.head(class_token_output)

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head self-attention and MLP."""
    
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        # Multi-head self-attention
        attention_output = self.attention(inputs, inputs, training=training)
        attention_output = self.dropout(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # MLP
        mlp_output = self.mlp(out1, training=training)
        return self.layernorm2(out1 + mlp_output)

# Non-model functions that should be ignored
def helper_function():
    """This is just a helper function, not a model."""
    pass

def preprocess_image(image):
    """Image preprocessing function."""
    return tf.cast(image, tf.float32) / 255.0
