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
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


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


def custom_csv_data_loader(csv_path: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         buffer_size: int = 10000,
                         label_column: str = 'label',
                         feature_columns: Optional[List[str]] = None) -> tf.data.Dataset:
    """
    Custom CSV data loader for tabular data.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        label_column: Name of the label column
        feature_columns: List of feature column names (None for all except label)
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV")
    
    # Get feature columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != label_column]
    
    # Extract features and labels
    features = df[feature_columns].values.astype(np.float32)
    labels = df[label_column].values
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def custom_sequence_data_loader(data_file: str,
                              sequence_length: int = 100,
                              batch_size: int = 32,
                              shuffle: bool = True,
                              vocab_size: int = 10000,
                              overlap: float = 0.5) -> tf.data.Dataset:
    """
    Custom sequence data loader for text or time series data.
    
    Args:
        data_file: Path to data file
        sequence_length: Length of each sequence
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        vocab_size: Vocabulary size (for text data)
        overlap: Overlap ratio between consecutive sequences
    
    Returns:
        tf.data.Dataset: Sequence dataset
    """
    # Read data
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple tokenization (replace with proper tokenizer in production)
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    
    # Convert text to indices
    data = [char_to_idx[char] for char in text]
    
    # Create sequences
    step = int(sequence_length * (1 - overlap))
    sequences = []
    targets = []
    
    for i in range(0, len(data) - sequence_length, step):
        seq = data[i:i + sequence_length]
        target = data[i + 1:i + sequence_length + 1]
        sequences.append(seq)
        targets.append(target)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences, targets))
    
    if shuffle:
        dataset = dataset.shuffle(len(sequences))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
