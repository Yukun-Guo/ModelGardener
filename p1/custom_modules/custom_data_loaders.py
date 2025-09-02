import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.model_selection import train_test_split

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

