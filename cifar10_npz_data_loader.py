"""
CIFAR-10 NPZ Data Loader for ModelGardener

This module provides a custom data loader for loading CIFAR-10 subset data from .npz files.
The data loader is designed to work with the ModelGardener framework and supports
train/validation splits, batch processing, and data augmentation.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

def load_cifar10_npz(npz_path: str,
                     batch_size: int = 32,
                     validation_split: float = 0.2,
                     shuffle: bool = True,
                     buffer_size: int = 10000,
                     split: str = 'train',
                     seed: int = 42) -> tf.data.Dataset:
    """
    Load CIFAR-10 subset data from NPZ file.
    
    Args:
        npz_path: Path to the .npz file containing CIFAR-10 data
        batch_size: Batch size for the dataset
        validation_split: Fraction of data to use for validation (0.0-1.0)
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        split: Which split to return ('train' or 'val')
        seed: Random seed for reproducibility
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    # Load data from NPZ file
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    data = np.load(npz_path)
    x_data = data['x']  # Shape: (1000, 32, 32, 3)
    y_data = data['y']  # Shape: (1000,)
    
    print(f"Loaded data shape: {x_data.shape}, labels shape: {y_data.shape}")
    print(f"Unique classes: {np.unique(y_data)}")
    
    # Normalize images to [0, 1]
    x_data = x_data.astype(np.float32) / 255.0
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(y_data))
    y_data_categorical = tf.keras.utils.to_categorical(y_data, num_classes)
    
    # Split data into train and validation
    if validation_split > 0.0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data_categorical, 
            test_size=validation_split, 
            random_state=seed,
            stratify=y_data  # Ensure balanced split
        )
    else:
        x_train, y_train = x_data, y_data_categorical
        x_val, y_val = x_data, y_data_categorical  # Use same data if no split
    
    # Select the appropriate split
    if split == 'train':
        x_split, y_split = x_train, y_train
        print(f"Using training data: {x_split.shape[0]} samples")
    elif split == 'val' or split == 'validation':
        x_split, y_split = x_val, y_val
        print(f"Using validation data: {x_split.shape[0]} samples")
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_split, y_split))
    
    # Shuffle if requested
    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size, seed=seed)
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class CIFAR10NPZDataLoader:
    """
    Class-based CIFAR-10 NPZ data loader.
    
    This class provides a convenient interface for loading CIFAR-10 data
    with configurable parameters.
    """
    
    def __init__(self, 
                 npz_path: str,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 shuffle: bool = True,
                 buffer_size: int = 10000,
                 seed: int = 42):
        """
        Initialize the CIFAR-10 NPZ data loader.
        
        Args:
            npz_path: Path to the .npz file
            batch_size: Batch size for the dataset
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle the training data
            buffer_size: Buffer size for shuffling
            seed: Random seed for reproducibility
        """
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        
        # Load and preprocess data once
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data from NPZ file."""
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")
        
        data = np.load(self.npz_path)
        self.x_data = data['x'].astype(np.float32) / 255.0
        self.y_data = data['y']
        
        # Convert to categorical
        self.num_classes = len(np.unique(self.y_data))
        self.y_data_categorical = tf.keras.utils.to_categorical(self.y_data, self.num_classes)
        
        # Split data
        if self.validation_split > 0.0:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_data, self.y_data_categorical,
                test_size=self.validation_split,
                random_state=self.seed,
                stratify=self.y_data
            )
        else:
            self.x_train = self.x_val = self.x_data
            self.y_train = self.y_val = self.y_data_categorical
    
    def get_train_dataset(self) -> tf.data.Dataset:
        """Get training dataset."""
        return self._create_dataset(self.x_train, self.y_train, shuffle=self.shuffle)
    
    def get_val_dataset(self) -> tf.data.Dataset:
        """Get validation dataset."""
        return self._create_dataset(self.x_val, self.y_val, shuffle=False)
    
    def get_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """
        Get dataset for specified split.
        
        Args:
            split: Which split to return ('train' or 'val')
            
        Returns:
            tf.data.Dataset: The requested dataset
        """
        if split == 'train':
            return self.get_train_dataset()
        elif split in ['val', 'validation']:
            return self.get_val_dataset()
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
    
    def _create_dataset(self, x_data: np.ndarray, y_data: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from numpy arrays."""
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        
        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=self.seed)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Get input shape of the data."""
        return self.x_data.shape[1:]
    
    @property
    def num_samples_train(self) -> int:
        """Get number of training samples."""
        return len(self.x_train)
    
    @property
    def num_samples_val(self) -> int:
        """Get number of validation samples."""
        return len(self.x_val)


# For compatibility with ModelGardener's custom function loading
def create_cifar10_dataset(data_dir: str = None,
                          npz_file: str = "cifar10.npz",
                          batch_size: int = 32,
                          validation_split: float = 0.2,
                          shuffle: bool = True,
                          split: str = 'train',
                          **kwargs) -> tf.data.Dataset:
    """
    Compatibility function for ModelGardener.
    
    Args:
        data_dir: Directory containing the NPZ file (if None, uses npz_file path directly)
        npz_file: Name or path of the NPZ file
        batch_size: Batch size for the dataset
        validation_split: Fraction of data for validation
        shuffle: Whether to shuffle the data
        split: Which split to return ('train' or 'val')
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    if data_dir is not None:
        npz_path = os.path.join(data_dir, npz_file)
    else:
        npz_path = npz_file
    
    return load_cifar10_npz(
        npz_path=npz_path,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=shuffle,
        split=split
    )
