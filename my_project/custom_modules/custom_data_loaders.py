"""
Custom Data Loaders Template for ModelGardener

This file provides templates for creating custom data loading functions.
Implement your data loaders as functions that return tf.data.Dataset objects.
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def custom_image_data_loader(data_dir: str,
                           batch_size: int = 32,
                           image_size: List[int] = [224, 224],
                           shuffle: bool = True,
                           buffer_size: int = 10000) -> tf.data.Dataset:
    """
    Custom image data loader with advanced preprocessing.
    
    Args:
        data_dir: Directory containing class subdirectories with images
        batch_size: Batch size for training
        image_size: Target image size [height, width]
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    
    def load_and_preprocess_image(path, label):
        """Load and preprocess a single image."""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Custom preprocessing can be added here
        # e.g., normalization, color space conversion, etc.
        
        return image, label
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def custom_csv_data_loader(csv_path: str,
                          image_dir: str,
                          batch_size: int = 32,
                          image_size: List[int] = [224, 224],
                          shuffle: bool = True) -> tf.data.Dataset:
    """
    Custom data loader for CSV-based datasets.
    
    Args:
        csv_path: Path to CSV file with image paths and labels
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Target image size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Assuming CSV has 'image_path' and 'label' columns
    image_paths = [os.path.join(image_dir, path) for path in df['image_path'].values]
    labels = df['label'].values
    
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(len(image_paths))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def custom_numpy_data_loader(data_path: str,
                            labels_path: str,
                            batch_size: int = 32,
                            shuffle: bool = True) -> tf.data.Dataset:
    """
    Custom data loader for numpy arrays.
    
    Args:
        data_path: Path to numpy array file with data
        labels_path: Path to numpy array file with labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset: Dataset
    """
    
    # Load numpy arrays
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    # Normalize data if needed
    data = data.astype(np.float32) / 255.0
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    
    if shuffle:
        dataset = dataset.shuffle(len(data))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class CustomDataLoader:
    """
    Example of a class-based custom data loader.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32, image_size: List[int] = [224, 224]):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        
    def create_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """Create dataset for specified split."""
        split_dir = os.path.join(self.data_dir, split)
        
        # Implementation would depend on your data structure
        # This is just a placeholder
        dataset = tf.data.Dataset.from_tensor_slices([])
        return dataset


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom data loaders...")
    
    # You can test your data loaders here
    # Example:
    # dataset = custom_image_data_loader('./data/train', batch_size=16)
    # for batch in dataset.take(1):
    #     print(f"Batch shape: {batch[0].shape}")
    
    print("✅ Custom data loaders template ready!")
