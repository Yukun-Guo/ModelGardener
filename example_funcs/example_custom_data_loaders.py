"""
Example custom data loader functions for ModelGardener

This file demonstrates how to create custom data loader functions and classes that can be
dynamically loaded into the ModelGardener application. Data loaders can be either:

1. Functions that return a tf.data.Dataset
2. Classes that provide data loading functionality

For functions:
- Should return a tf.data.Dataset instance
- Can accept configuration parameters like batch_size, shuffle, etc.
- Should handle both training and validation data paths

For classes:
- Should provide methods to create datasets
- Can have __init__ method with configuration parameters
- Should implement data loading logic
"""

import os
import tensorflow as tf
from typing import List

def custom_image_data_loader(data_dir: str,
                           batch_size: int = 32,
                           image_size: List[int] = [224, 224],
                           shuffle: bool = True,
                           buffer_size: int = 10000,
                           augment: bool = False) -> tf.data.Dataset:
    """
    Custom image data loader that loads images from directories.
    
    Args:
        data_dir: Path to directory containing image files
        batch_size: Batch size for the dataset
        image_size: Target image size [height, width]
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        augment: Whether to apply data augmentation
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        raise ValueError(f"No image files found in {data_dir}")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    
    # Load and preprocess images
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Simple augmentation if requested
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
        
        return image
    
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
