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


class CustomTFRecordLoader:
    """
    Custom TFRecord data loader class with advanced features.
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 buffer_size: int = 10000,
                 num_parallel_calls: int = tf.data.AUTOTUNE,
                 prefetch_size: int = tf.data.AUTOTUNE,
                 compression_type: str = '',
                 parse_fn: Optional[callable] = None):
        """
        Initialize the TFRecord loader.
        
        Args:
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            buffer_size: Buffer size for shuffling
            num_parallel_calls: Number of parallel calls for processing
            prefetch_size: Prefetch buffer size
            compression_type: Compression type for TFRecord files
            parse_fn: Custom parse function for TFRecord examples
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_size = prefetch_size
        self.compression_type = compression_type
        self.parse_fn = parse_fn or self._default_parse_fn
    
    def _default_parse_fn(self, example_proto):
        """Default parse function for TFRecord examples."""
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode image
        image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Get label
        label = tf.cast(parsed_features['image/class/label'], tf.int32)
        
        return image, label
    
    def load_dataset(self, file_pattern: str) -> tf.data.Dataset:
        """
        Load dataset from TFRecord files.
        
        Args:
            file_pattern: Pattern for TFRecord files (e.g., '/path/to/*.tfrecord')
        
        Returns:
            tf.data.Dataset: Processed dataset
        """
        # Get list of TFRecord files
        files = tf.data.Dataset.list_files(file_pattern, shuffle=self.shuffle)
        
        # Create dataset from TFRecord files
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=self.compression_type),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=self.num_parallel_calls
        )
        
        # Parse examples
        dataset = dataset.map(self.parse_fn, num_parallel_calls=self.num_parallel_calls)
        
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_size)
        
        return dataset


class CustomMultiModalLoader:
    """
    Custom multi-modal data loader for handling multiple types of input data.
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 image_size: List[int] = [224, 224],
                 text_max_length: int = 256,
                 vocab_size: int = 10000):
        """
        Initialize multi-modal loader.
        
        Args:
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            image_size: Target size for images [height, width]
            text_max_length: Maximum length for text sequences
            vocab_size: Vocabulary size for text tokenization
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.vocab_size = vocab_size
    
    def create_dataset(self, data_dir: str) -> tf.data.Dataset:
        """
        Create multi-modal dataset from directory structure.
        
        Expected structure:
        data_dir/
        ├── images/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── texts/
        │   ├── text1.txt
        │   └── text2.txt
        └── labels.csv
        
        Args:
            data_dir: Path to data directory
        
        Returns:
            tf.data.Dataset: Multi-modal dataset
        """
        labels_path = os.path.join(data_dir, 'labels.csv')
        images_dir = os.path.join(data_dir, 'images')
        texts_dir = os.path.join(data_dir, 'texts')
        
        # Read labels
        labels_df = pd.read_csv(labels_path)
        
        def load_multimodal_sample(image_file, text_file, label):
            # Load and process image
            image_path = tf.strings.join([images_dir, '/', image_file])
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Load and process text
            text_path = tf.strings.join([texts_dir, '/', text_file])
            text = tf.io.read_file(text_path)
            # Simple text processing (in real scenario, use proper tokenizer)
            text = tf.strings.substr(text, 0, self.text_max_length)
            
            return {'image': image, 'text': text}, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'image_file': labels_df['image_file'].values,
            'text_file': labels_df['text_file'].values,
            'label': labels_df['label'].values
        })
        
        dataset = dataset.map(
            lambda x: load_multimodal_sample(x['image_file'], x['text_file'], x['label']),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if self.shuffle:
            dataset = dataset.shuffle(10000)
        
        dataset = dataset.batch(self.batch_size)
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
