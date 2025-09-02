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
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.model_selection import train_test_split

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


def Custom_load_cifar10_npz_data(train_dir: str = "./data", 
                                 val_dir: str = "./data",
                                 npz_file_path: str = "./data/cifar10.npz",
                                 batch_size: int = 32,
                                 shuffle: bool = True,
                                 buffer_size: int = 1000,
                                 validation_split: float = 0.2,
                                 **kwargs):
    """
    Custom CIFAR-10 NPZ data loader for ModelGardener.
    
    This function loads CIFAR-10 data from an NPZ file and returns
    training and validation datasets.
    
    Args:
        train_dir: Directory path (used for compatibility)
        val_dir: Directory path (used for compatibility) 
        npz_file_path: Path to the NPZ file containing CIFAR-10 data
        batch_size: Batch size for datasets
        shuffle: Whether to shuffle the data
        buffer_size: Buffer size for shuffling
        validation_split: Fraction of data to use for validation
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets
    """
    print(f"🔍 Loading CIFAR-10 data from: {npz_file_path}")
    
    # Load NPZ file
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")
    
    data = np.load(npz_file_path)
    images = data['x'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    labels = data['y'].astype(np.int32)
    
    print(f"📊 Loaded {len(images)} images with shape {images.shape[1:]}")
    print(f"🎯 Found {len(np.unique(labels))} unique classes")
    
    # Split into train and validation
    train_indices = int(len(images) * (1 - validation_split))
    
    train_images = images[:train_indices]
    train_labels = labels[:train_indices]
    val_images = images[train_indices:]
    val_labels = labels[train_indices:]
    
    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(labels))
    train_labels_categorical = tf.keras.utils.to_categorical(train_labels, num_classes)
    val_labels_categorical = tf.keras.utils.to_categorical(val_labels, num_classes)
    
    print(f"🚂 Training set: {len(train_images)} samples")
    print(f"✅ Validation set: {len(val_images)} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_categorical))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_categorical))
    
    # Apply shuffling if requested
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    
    # Batch the datasets
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def load_cifar10_npz_data(data_dir: str,
                          npz_file: str = "cifar10.npz", 
                          batch_size: int = 32,
                          validation_split: float = 0.2,
                          shuffle: bool = True,
                          buffer_size: int = 10000,
                          split: str = 'train',
                          seed: int = 42,
                          **kwargs) -> tf.data.Dataset:
    """
    Load CIFAR-10 subset data from NPZ file for ModelGardener.
    
    This data loader is specifically designed for the CIFAR-10 subset dataset
    generated by test_generate_subset.py. It handles train/validation splits,
    normalization, and categorical encoding automatically.
    
    Args:
        data_dir: Directory containing the NPZ file
        npz_file: Name of the NPZ file (default: "cifar10.npz")
        batch_size: Batch size for the dataset
        validation_split: Fraction of data to use for validation (0.0-1.0)
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        split: Which split to return ('train' or 'val')
        seed: Random seed for reproducibility
    
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
        
    Example:
        # Load training data
        train_ds = load_cifar10_npz_data(
            data_dir="./example_data",
            batch_size=32,
            split='train'
        )
        
        # Load validation data
        val_ds = load_cifar10_npz_data(
            data_dir="./example_data", 
            batch_size=32,
            split='val'
        )
    """
    # Construct full path to NPZ file
    npz_path = os.path.join(data_dir, npz_file)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load data from NPZ file
    data = np.load(npz_path)
    x_data = data['x']  # Shape: (1000, 32, 32, 3)
    y_data = data['y']  # Shape: (1000,)
    
    print(f"Loaded CIFAR-10 data: {x_data.shape}, labels: {y_data.shape}")
    print(f"Classes: {np.unique(y_data)}")
    
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
    
    # Shuffle if requested (only for training)
    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size, seed=seed)
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class CIFAR10NPZDataLoader:
    """
    Class-based CIFAR-10 NPZ data loader for ModelGardener.
    
    This class provides a convenient interface for loading CIFAR-10 subset data
    with configurable parameters and methods for accessing train/val splits.
    """
    
    def __init__(self, 
                 data_dir: str,
                 npz_file: str = "cifar10.npz",
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 shuffle: bool = True,
                 buffer_size: int = 10000,
                 seed: int = 42):
        """
        Initialize the CIFAR-10 NPZ data loader.
        
        Args:
            data_dir: Directory containing the NPZ file
            npz_file: Name of the NPZ file
            batch_size: Batch size for the dataset
            validation_split: Fraction of data for validation
            shuffle: Whether to shuffle the training data
            buffer_size: Buffer size for shuffling
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.npz_file = npz_file
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        
        # Load and preprocess data once
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data from NPZ file."""
        npz_path = os.path.join(self.data_dir, self.npz_file)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        data = np.load(npz_path)
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
    
    def get_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """
        Get dataset for specified split.
        
        Args:
            split: Which split to return ('train' or 'val')
            
        Returns:
            tf.data.Dataset: The requested dataset
        """
        if split == 'train':
            return self._create_dataset(self.x_train, self.y_train, shuffle=self.shuffle)
        elif split in ['val', 'validation']:
            return self._create_dataset(self.x_val, self.y_val, shuffle=False)
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


# Alternative simple function for basic usage
def simple_cifar10_loader(data_dir: str,
                         batch_size: int = 32,
                         split: str = 'train',
                         **kwargs) -> tf.data.Dataset:
    """
    Simple CIFAR-10 NPZ loader with minimal configuration.
    
    Args:
        data_dir: Directory containing cifar10.npz file
        batch_size: Batch size for the dataset
        split: Which split to return ('train' or 'val')
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        tf.data.Dataset: Dataset ready for training/validation
    """
    return load_cifar10_npz_data(
        data_dir=data_dir,
        npz_file="cifar10.npz",
        batch_size=batch_size,
        split=split,
        validation_split=0.2,
        shuffle=True
    )
